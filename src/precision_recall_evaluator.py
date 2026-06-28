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
from safety_gate import AlignmentAndSafetyGate

def measure_precision_recall(test_parquet="data/processed/test.parquet", num_samples=300):
    print("=========================================================")
    print("MEASURING PRECISION, RECALL & F1 FOR ABLATION SETUPS")
    print("=========================================================")
    
    df_test = pd.read_parquet(test_parquet).head(num_samples)
    model_dir = "outputs/models/best_model"
    
    preprocessor = VietnameseTextPreprocessor()
    detector = VietnameseErrorDetector()
    safety_gate = AlignmentAndSafetyGate(edit_distance_threshold=3)
    
    if not os.path.exists(model_dir):
        print("[Precision/Recall] Model dir not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    setups = [
        {"name": "Greedy Decoding (num_beams=1)", "beams": 1, "use_gate": False},
        {"name": "Beam Search (num_beams=4)", "beams": 4, "use_gate": False},
        {"name": "Full Multi-stage Pipeline", "beams": 4, "use_gate": True}
    ]
    
    metrics_summary = []
    
    for setup in setups:
        tp, fp, fn, tn = 0, 0, 0, 0
        preds = []
        
        for idx, row in df_test.iterrows():
            text = row["noisy_text"]
            clean = row["clean_text"]
            
            masked, ents = preprocessor.process(text)
            is_clean, err_indices = detector.check_sentence(masked)
            
            if setup["use_gate"] and is_clean:
                pred = text
            else:
                inp = tokenizer(f"correct: {masked}", return_tensors="pt", max_length=256, truncation=True).to(device)
                with torch.no_grad():
                    out = model.generate(**inp, max_length=256, num_beams=setup["beams"])
                raw_p = tokenizer.decode(out[0], skip_special_tokens=True)
                pred = preprocessor.unmask_entities(raw_p, ents)
                
                if setup["use_gate"]:
                    aligned = safety_gate.align_tokens(text, pred)
                    words = [corr if status != 'DELETE' else '' for orig, corr, status in aligned]
                    pred = " ".join([w for w in words if w])
                    
            preds.append(pred)
            
            noisy_words = text.split()
            clean_words = clean.split()
            pred_words = pred.split()
            
            min_len = min(len(noisy_words), len(clean_words), len(pred_words))
            for i in range(min_len):
                was_error = (noisy_words[i] != clean_words[i])
                did_change = (noisy_words[i] != pred_words[i])
                is_correct_change = (pred_words[i] == clean_words[i])
                
                if did_change:
                    if is_correct_change:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if was_error:
                        fn += 1
                    else:
                        tn += 1
                        
        precision = (tp / max(tp + fp, 1)) * 100
        recall = (tp / max(tp + fn, 1)) * 100
        f1 = (2 * precision * recall / max(precision + recall, 1e-5))
        
        cers = [compute_cer(r, p) for r, p in zip(df_test["clean_text"], preds)]
        wers = [compute_wer(r, p) for r, p in zip(df_test["clean_text"], preds)]
        exact = (pd.Series(preds) == df_test["clean_text"]).mean() * 100
        clean_mask = (df_test["noisy_text"] == df_test["clean_text"])
        over_corr = ((pd.Series(preds) != df_test["clean_text"]) & clean_mask).sum() / max(clean_mask.sum(), 1) * 100
        
        metrics_summary.append({
            "Mô hình / Phiên bản": setup["name"],
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "F1-Score (%)": round(f1, 2),
            "CER (%)": round(np.mean(cers)*100, 2),
            "WER (%)": round(np.mean(wers)*100, 2),
            "Over-Correction Rate (%)": round(over_corr, 2)
        })
        
    df_res = pd.DataFrame(metrics_summary)
    os.makedirs("outputs/benchmarks", exist_ok=True)
    df_res.to_csv("outputs/benchmarks/precision_recall_tradeoff.csv", index=False)
    print("\n=== PRECISION / RECALL TRADE-OFF TABLE ===")
    print(df_res.to_string(index=False))
    print("\n[Trade-off] Saved metrics to outputs/benchmarks/precision_recall_tradeoff.csv")

if __name__ == "__main__":
    measure_precision_recall()
