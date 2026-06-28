import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from preprocess import VietnameseTextPreprocessor
from error_detector import VietnameseErrorDetector
from candidate_generator import TwoStreamCandidateGenerator
from safety_gate import AlignmentAndSafetyGate

def compute_cer(ref, pred):
    import edlib
    if not ref: return 0.0 if not pred else 1.0
    res = edlib.align(pred, ref)
    return res['editDistance'] / max(len(ref), 1)

def compute_wer(ref, pred):
    import edlib
    ref_words = ref.split()
    pred_words = pred.split()
    if not ref_words: return 0.0 if not pred_words else 1.0
    word_map = {}
    def to_chars(words):
        chars = []
        for w in words:
            if w not in word_map:
                word_map[w] = chr(len(word_map) + 1000)
            chars.append(word_map[w])
        return "".join(chars)
    res = edlib.align(to_chars(pred_words), to_chars(ref_words))
    return res['editDistance'] / max(len(ref_words), 1)

class ProductionMultiStagePipeline:
    """
    Complete 5-Stage Multi-stage Pipeline Integration
    Stage 1: Sanitization & Masking
    Stage 2: Error Detection Gate
    Stage 3: Two-Stream Candidate Generation
    Stage 4: Neural Contextual Reranking
    Stage 5: Alignment & Safety Gate
    """
    def __init__(self, model_dir="outputs/models/best_model"):
        self.preprocessor = VietnameseTextPreprocessor()
        self.detector = VietnameseErrorDetector()
        self.cand_gen = TwoStreamCandidateGenerator(self.detector.vocab)
        self.safety_gate = AlignmentAndSafetyGate(edit_distance_threshold=3)
        
        self.has_neural = os.path.exists(model_dir)
        if self.has_neural:
            print(f"[Pipeline] Loading Neural Transformer from {model_dir} on GPU/CPU...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        import unicodedata
        def remove_accents(s): return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').replace('đ','d').replace('Đ','D')
        self.accentless_vocab = set(remove_accents(w.lower()) for w in self.detector.vocab if len(w)>=2)
        self.vowels = set('aeiouyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ')

    def split_stuck_words(self, text: str) -> str:
        import re, unicodedata
        def remove_accents(s): return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').replace('đ','d').replace('Đ','D')
        vocab_lower = set(w.lower() for w in self.detector.vocab)
        def valid_syl(s):
            sl = s.lower()
            if len(sl) < 2 or not any(c in self.vowels for c in sl): return False
            return sl in vocab_lower or remove_accents(sl) in self.accentless_vocab

        words = text.split()
        new_words = []
        for w in words:
            if len(w) < 4 or w.startswith("__ENT_") or any(char.isdigit() for char in w):
                new_words.append(w)
                continue
            punct = re.sub(r'[\w\s]', '', w)
            w_clean = re.sub(r'[^\w\s]', '', w)
            if w_clean.lower() in vocab_lower or len(w_clean) < 4:
                new_words.append(w)
                continue
            
            split_res = None
            # Pass 1: Strict exact accented matches
            for i in range(len(w_clean)-2, 1, -1):
                p1, p2 = w_clean[:i], w_clean[i:]
                if p1.lower() in vocab_lower and p2.lower() in vocab_lower:
                    split_res = p1 + ' ' + p2 + punct
                    break
                for j in range(len(p2)-2, 1, -1):
                    p2a, p2b = p2[:j], p2[j:]
                    if p1.lower() in vocab_lower and p2a.lower() in vocab_lower and p2b.lower() in vocab_lower:
                        split_res = p1 + ' ' + p2a + ' ' + p2b + punct
                        break
                if split_res: break
            
            # Pass 2: Accentless fallback
            if not split_res:
                for i in range(len(w_clean)-2, 1, -1):
                    p1, p2 = w_clean[:i], w_clean[i:]
                    if valid_syl(p1):
                        if valid_syl(p2):
                            split_res = p1 + ' ' + p2 + punct
                            break
                        for j in range(len(p2)-2, 1, -1):
                            p2a, p2b = p2[:j], p2[j:]
                            if valid_syl(p2a) and valid_syl(p2b):
                                split_res = p1 + ' ' + p2a + ' ' + p2b + punct
                                break
                    if split_res: break
            
            new_words.append(split_res if split_res else w)
        return " ".join(new_words)

    def process_sentence(self, text: str) -> str:
        if not text.strip():
            return text
        split_input = self.split_stuck_words(text)
        # Stage 1: Preprocess & Mask entities (URLs, Emails, Acronyms)
        masked_text, entities = self.preprocessor.process(split_input)
        
        if self.has_neural:
            # Stage 4: Neural Contextual Seq2Seq Correction (ViT5)
            inputs = self.tokenizer(f"correct: {masked_text}", return_tensors="pt", max_length=256, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=256, num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.2)
            raw_pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            unmasked_pred = self.preprocessor.unmask_entities(raw_pred, entities)
        else:
            # Stage 2 & 3: Error Detection & Candidate fallback
            is_clean, err_indices = self.detector.check_sentence(masked_text)
            if is_clean:
                return split_input
            words = masked_text.split()
            corr_words = [self.cand_gen.generate_candidates(w)[0] if idx in err_indices else w for idx, w in enumerate(words)]
            unmasked_pred = self.preprocessor.unmask_entities(" ".join(corr_words), entities)
            
        # Stage 5: Alignment & Safety Gate Filter
        aligned = self.safety_gate.align_tokens(split_input, unmasked_pred)
        safe_words = [corr if status != 'DELETE' else '' for orig, corr, status in aligned]
        res = " ".join([w for w in safe_words if w])
        import re
        res = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', res, flags=re.IGNORECASE)
        res = re.sub(r'\b(nhé|nha|ạ)\.[\s\.]*(?:nhé|nha|ạ)\.?\b', r'\1.', res, flags=re.IGNORECASE)
        return res

    def process(self, text: str) -> str:
        import re
        if not text:
            return ""
        # Split text by sentence boundaries preserving punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 1:
            res = self.process_sentence(text)
        else:
            corrected_sentences = [self.process_sentence(s) for s in sentences if s.strip()]
            res = " ".join(corrected_sentences)
            
        # Global Deduplication Filter across full paragraph
        res = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', res, flags=re.IGNORECASE)
        res = re.sub(r'(\b\w+\b[\.!\?])(?:\s+\1)+', r'\1', res, flags=re.IGNORECASE)
        return res

def run_evaluation(num_samples=500):
    evaluator = ProductionMultiStagePipeline("outputs/models/best_model")
    df_test = pd.read_parquet("data/processed/test.parquet")
    if num_samples and len(df_test) > num_samples:
        df_test = df_test.head(num_samples).copy()
        
    print(f"[Pipeline Eval] Running evaluation on {len(df_test)} test samples...")
    preds = [evaluator.process(text) for text in df_test["noisy_text"]]
    df_test["predicted_text"] = preds
    
    cers = [compute_cer(r, p) for r, p in zip(df_test["clean_text"], df_test["predicted_text"])]
    wers = [compute_wer(r, p) for r, p in zip(df_test["clean_text"], df_test["predicted_text"])]
    exact_match = (df_test["predicted_text"] == df_test["clean_text"]).mean() * 100
    
    clean_mask = (df_test["noisy_text"] == df_test["clean_text"])
    over_corr = ((df_test["predicted_text"] != df_test["clean_text"]) & clean_mask).sum() / max(clean_mask.sum(), 1) * 100
    
    print("\n=== COMPLETE MULTI-STAGE PIPELINE EVALUATION ===")
    print(f"Exact Match (Accuracy): {exact_match:.2f}%")
    print(f"Character Error Rate (CER): {np.mean(cers)*100:.2f}%")
    print(f"Word Error Rate (WER): {np.mean(wers)*100:.2f}%")
    print(f"Over-Correction Rate: {over_corr:.2f}%")
    
    os.makedirs("outputs/predictions", exist_ok=True)
    df_test.to_csv("outputs/predictions/test_predictions.csv", index=False)

if __name__ == "__main__":
    run_evaluation(num_samples=300)
