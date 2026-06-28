import os
import sys
import time
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

def run_unified_master_evaluation(test_parquet="data/processed/test.parquet", num_samples=1000):
    print("=========================================================")
    print("UNIFIED MASTER EVALUATOR: 100% SINGLE SOURCE OF TRUTH")
    print("=========================================================")
    
    # 1. Load exact test set (1000 samples)
    df_test = pd.read_parquet(test_parquet)
    if num_samples and len(df_test) > num_samples:
        df_test = df_test.head(num_samples).copy()
    print(f"[Master Eval] Loaded exact test dataset: {len(df_test)} samples.")
        
    preprocessor = VietnameseTextPreprocessor()
    detector = VietnameseErrorDetector()
    cand_gen = TwoStreamCandidateGenerator(detector.vocab)
    safety_gate = AlignmentAndSafetyGate(edit_distance_threshold=3)
    
    # 2. Load exact single checkpoint
    model_dir = "outputs/models/best_model"
    has_neural = os.path.exists(model_dir)
    
    tokenizer = None
    model = None
    device = "cpu"
    if has_neural:
        print(f"[Master Eval] Loading exact checkpoint from {model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"[Master Eval] Running neural inference on device: {device}")
    else:
        print("[Master Eval] WARNING: Neural checkpoint not found! Operating in heuristic fallback.")

    setups = [
        {"name": "1. Baseline (SymSpell + N-gram)", "type": "heuristic"},
        {"name": "2. Neural Model Greedy (num_beams=1)", "type": "neural_greedy"},
        {"name": "3. Neural Model Beam Search (num_beams=4)", "type": "neural_beam"},
        {"name": "4. Full Multi-stage Pipeline (+ Safety Gate)", "type": "full_pipeline"}
    ]
    
    master_results = []
    category_results_map = {}
    
    for setup in setups:
        print(f"\n[Master Eval Pass] Evaluating: {setup['name']}...")
        preds = []
        tp, fp, fn, tn = 0, 0, 0, 0
        start_t = time.time()
        
        for idx, row in df_test.iterrows():
            text = row["noisy_text"]
            clean = row["clean_text"]
            
            def process_single_sentence(sentence_str, setup_type):
                masked_text, entities = preprocessor.process(sentence_str)
                is_clean, err_indices = detector.check_sentence(masked_text)
                
                if setup_type == "heuristic":
                    if is_clean:
                        return sentence_str
                    words = masked_text.split()
                    corr_words = [cand_gen.generate_candidates(w)[0] if i in err_indices else w for i, w in enumerate(words)]
                    return preprocessor.unmask_entities(" ".join(corr_words), entities)
                        
                elif setup_type in ["neural_greedy", "neural_beam"]:
                    if has_neural:
                        beams = 1 if setup_type == "neural_greedy" else 4
                        inp = tokenizer(f"correct: {masked_text}", return_tensors="pt", max_length=256, truncation=True).to(device)
                        with torch.no_grad():
                            out = model.generate(**inp, max_length=256, num_beams=beams, no_repeat_ngram_size=3, repetition_penalty=1.2, early_stopping=True)
                        raw_p = tokenizer.decode(out[0], skip_special_tokens=True)
                        return preprocessor.unmask_entities(raw_p, entities)
                    else:
                        return sentence_str
                        
                elif setup_type == "full_pipeline":
                    if has_neural:
                        inp = tokenizer(f"correct: {masked_text}", return_tensors="pt", max_length=256, truncation=True).to(device)
                        with torch.no_grad():
                            out = model.generate(**inp, max_length=256, num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.2, early_stopping=True)
                        raw_p = tokenizer.decode(out[0], skip_special_tokens=True)
                        unmasked = preprocessor.unmask_entities(raw_p, entities)
                    else:
                        words = masked_text.split()
                        corr_words = [cand_gen.generate_candidates(w)[0] if i in err_indices else w for i, w in enumerate(words)]
                        unmasked = preprocessor.unmask_entities(" ".join(corr_words), entities)
                    
                    aligned = safety_gate.align_tokens(sentence_str, unmasked)
                    safe_words = [corr if status != 'DELETE' else '' for orig, corr, status in aligned]
                    return " ".join([w for w in safe_words if w])

            import re
            sents = re.split(r'(?<=[.!?])\s+', text)
            if len(sents) <= 1:
                pred = process_single_sentence(text, setup["type"])
            else:
                pred = " ".join([process_single_sentence(s, setup["type"]) for s in sents if s.strip()])

            preds.append(pred)
            
            # Precision / Recall word-level calculation
            noisy_words = text.split()
            clean_words = clean.split()
            pred_words = pred.split()
            min_l = min(len(noisy_words), len(clean_words), len(pred_words))
            for i in range(min_l):
                was_err = (noisy_words[i] != clean_words[i])
                did_chg = (noisy_words[i] != pred_words[i])
                is_corr = (pred_words[i] == clean_words[i])
                if did_chg:
                    if is_corr: tp += 1
                    else: fp += 1
                else:
                    if was_err: fn += 1
                    else: tn += 1
                    
        end_t = time.time()
        avg_latency_ms = ((end_t - start_t) / len(df_test)) * 1000
        
        cers = [compute_cer(r, p) for r, p in zip(df_test["clean_text"], preds)]
        wers = [compute_wer(r, p) for r, p in zip(df_test["clean_text"], preds)]
        exact = (pd.Series(preds) == df_test["clean_text"]).mean() * 100
        
        clean_mask = (df_test["noisy_text"] == df_test["clean_text"])
        over_corr = ((pd.Series(preds) != df_test["clean_text"]) & clean_mask).sum() / max(clean_mask.sum(), 1) * 100
        
        precision = (tp / max(tp + fp, 1)) * 100
        recall = (tp / max(tp + fn, 1)) * 100
        f1 = (2 * precision * recall / max(precision + recall, 1e-5))
        
        master_results.append({
            "Phiên bản Hệ thống (System Setup)": setup["name"],
            "Exact Match (%)": round(exact, 2),
            "CER (%)": round(np.mean(cers)*100, 2),
            "WER (%)": round(np.mean(wers)*100, 2),
            "Over-Correction (%)": round(over_corr, 2),
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "F1-Score (%)": round(f1, 2),
            "Avg Latency (ms)": round(avg_latency_ms, 2)
        })
        
        # Per category tracking for setup
        df_test[f"pred_{setup['type']}"] = preds

    df_master = pd.DataFrame(master_results)
    os.makedirs("outputs/benchmarks", exist_ok=True)
    df_master.to_csv("outputs/benchmarks/unified_master_benchmark.csv", index=False)
    
    print("\n=== UNIFIED MASTER BENCHMARK TABLE (100% UNIFIED SOURCE OF TRUTH) ===")
    print(df_master.to_string(index=False))
    print("\n[Master Eval] Saved unified benchmark table to outputs/benchmarks/unified_master_benchmark.csv")
    
    # Generate exact per-category breakdown on Full Pipeline predictions
    categories = df_test["error_type"].unique()
    cat_summary = []
    for cat in categories:
        sub = df_test[df_test["error_type"] == cat]
        sub_preds = sub["pred_full_pipeline"]
        sub_cers = [compute_cer(r, p) for r, p in zip(sub["clean_text"], sub_preds)]
        sub_wers = [compute_wer(r, p) for r, p in zip(sub["clean_text"], sub_preds)]
        sub_exact = (sub_preds == sub["clean_text"]).mean() * 100
        sub_over = ((sub_preds != sub["clean_text"]) & (sub["noisy_text"] == sub["clean_text"])).mean() * 100 if cat=="none" else 0.0
        
        cat_summary.append({
            "Loại lỗi (Error Type)": cat,
            "Số lượng mẫu": len(sub),
            "Exact Match (%)": round(sub_exact, 2),
            "CER (%)": round(np.mean(sub_cers)*100, 2),
            "WER (%)": round(np.mean(sub_wers)*100, 2),
            "Over-Correction Rate (%)": round(sub_over, 2)
        })
        
    df_cat = pd.DataFrame(cat_summary)
    df_cat.to_csv("outputs/benchmarks/unified_category_breakdown.csv", index=False)
    print("\n=== UNIFIED CATEGORY BREAKDOWN TABLE (FULL PIPELINE) ===")
    print(df_cat.to_string(index=False))
    print("[Master Eval] Saved unified category breakdown to outputs/benchmarks/unified_category_breakdown.csv")

if __name__ == "__main__":
    run_unified_master_evaluation(num_samples=1000)
