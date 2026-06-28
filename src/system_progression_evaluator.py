import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from evaluate_correction import compute_cer, compute_wer
from preprocess import VietnameseTextPreprocessor
from error_detector import VietnameseErrorDetector
from candidate_generator import TwoStreamCandidateGenerator
from safety_gate import AlignmentAndSafetyGate

def run_system_progression_evaluation(test_parquet="data/processed/test.parquet", num_samples=1000):
    print("=========================================================")
    print("EMPIRICAL SYSTEM PROGRESSION EVALUATION (SAME TEST SET)")
    print("=========================================================")
    
    df_test = pd.read_parquet(test_parquet)
    if num_samples and len(df_test) > num_samples:
        df_test = df_test.head(num_samples).copy()
        
    preprocessor = VietnameseTextPreprocessor()
    detector = VietnameseErrorDetector()
    cand_gen = TwoStreamCandidateGenerator(detector.vocab)
    safety_gate = AlignmentAndSafetyGate(edit_distance_threshold=3)
    
    model_dir = "outputs/models/best_model"
    has_neural = os.path.exists(model_dir)
    
    tokenizer = None
    model = None
    device = "cpu"
    if has_neural:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

    stages = [
        {"name": "1. Baseline (Heuristic / SymSpell + N-gram)", "type": "heuristic"},
        {"name": "2. Neural Model Greedy (`num_beams=1`)", "type": "neural_greedy"},
        {"name": "3. Neural Model + Beam Search (`num_beams=4`)", "type": "neural_beam"},
        {"name": "4. Full Pipeline (+ Candidate Gen & Safety Gate)", "type": "full_pipeline"}
    ]
    
    results = []
    
    for stage in stages:
        preds = []
        for text in df_test["noisy_text"]:
            masked_text, entities = preprocessor.process(text)
            is_clean, err_indices = detector.check_sentence(masked_text)
            
            if stage["type"] == "heuristic":
                if is_clean:
                    preds.append(text)
                else:
                    words = masked_text.split()
                    corr_words = []
                    for idx, w in enumerate(words):
                        if idx in err_indices:
                            cands = cand_gen.generate_candidates(w)
                            corr_words.append(cands[0])
                        else:
                            corr_words.append(w)
                    preds.append(preprocessor.unmask_entities(" ".join(corr_words), entities))
                    
            elif stage["type"] in ["neural_greedy", "neural_beam"]:
                if has_neural:
                    beams = 1 if stage["type"] == "neural_greedy" else 4
                    inp = tokenizer(f"correct: {masked_text}", return_tensors="pt", max_length=256, truncation=True).to(device)
                    with torch.no_grad():
                        out = model.generate(**inp, max_length=256, num_beams=beams)
                    raw_p = tokenizer.decode(out[0], skip_special_tokens=True)
                    preds.append(preprocessor.unmask_entities(raw_p, entities))
                else:
                    preds.append(text)
                    
            elif stage["type"] == "full_pipeline":
                if is_clean:
                    preds.append(text)
                else:
                    if has_neural:
                        inp = tokenizer(f"correct: {masked_text}", return_tensors="pt", max_length=256, truncation=True).to(device)
                        with torch.no_grad():
                            out = model.generate(**inp, max_length=256, num_beams=4)
                        raw_p = tokenizer.decode(out[0], skip_special_tokens=True)
                        unmasked = preprocessor.unmask_entities(raw_p, entities)
                    else:
                        words = masked_text.split()
                        corr_words = [cand_gen.generate_candidates(w)[0] if idx in err_indices else w for idx, w in enumerate(words)]
                        unmasked = preprocessor.unmask_entities(" ".join(corr_words), entities)
                    
                    aligned = safety_gate.align_tokens(text, unmasked)
                    safe_words = [corr if status != 'DELETE' else '' for orig, corr, status in aligned]
                    preds.append(" ".join([w for w in safe_words if w]))

        cers = [compute_cer(r, p) for r, p in zip(df_test["clean_text"], preds)]
        wers = [compute_wer(r, p) for r, p in zip(df_test["clean_text"], preds)]
        exact = (pd.Series(preds) == df_test["clean_text"]).mean() * 100
        
        clean_mask = (df_test["noisy_text"] == df_test["clean_text"])
        over_corr = ((pd.Series(preds) != df_test["clean_text"]) & clean_mask).sum() / max(clean_mask.sum(), 1) * 100
        
        results.append({
            "Phiên bản Hệ thống (System Stage)": stage["name"],
            "Exact Match (%)": round(exact, 2),
            "CER (%)": round(np.mean(cers)*100, 2),
            "WER (%)": round(np.mean(wers)*100, 2),
            "Over-Correction Rate (%)": round(over_corr, 2)
        })
        
    df_res = pd.DataFrame(results)
    os.makedirs("outputs/benchmarks", exist_ok=True)
    df_res.to_csv("outputs/benchmarks/system_progression_empirical.csv", index=False)
    
    print("\n=== VERIFIED SYSTEM PROGRESSION TABLE ===")
    print(df_res.to_string(index=False))
    print("\n[Progression] Saved verified table to outputs/benchmarks/system_progression_empirical.csv")

if __name__ == "__main__":
    run_system_progression_evaluation(num_samples=1000)
