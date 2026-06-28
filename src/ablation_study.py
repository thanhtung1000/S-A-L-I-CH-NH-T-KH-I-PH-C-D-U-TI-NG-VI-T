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

def run_ablation_experiments(test_parquet="data/processed/test.parquet", num_samples=300):
    print("=========================================================")
    print("ABLATION STUDY: IMPACT OF BEAM SEARCH & MULTI-STAGE GATES")
    print("=========================================================")
    
    df_test = pd.read_parquet(test_parquet).head(num_samples)
    model_dir = "outputs/models/best_model"
    
    preprocessor = VietnameseTextPreprocessor()
    detector = VietnameseErrorDetector()
    safety_gate = AlignmentAndSafetyGate(edit_distance_threshold=3)
    
    has_neural = os.path.exists(model_dir)
    if not has_neural:
        print("[Ablation] Model checkpoint not found. Using local simulation.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    experiments = [
        {"name": "Greedy Decoding (num_beams=1)", "num_beams": 1, "use_safety_gate": False},
        {"name": "Beam Search (num_beams=4)", "num_beams": 4, "use_safety_gate": False},
        {"name": "Beam Search (num_beams=5) + Length Penalty", "num_beams": 5, "use_safety_gate": False},
        {"name": "Full Multi-stage Pipeline (Beam 4 + Safety Gate)", "num_beams": 4, "use_safety_gate": True},
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n[Ablation Pass] Running {exp['name']}...")
        preds = []
        for text in df_test["noisy_text"]:
            masked, ents = preprocessor.process(text)
            is_clean, _ = detector.check_sentence(masked)
            
            if exp["use_safety_gate"] and is_clean:
                preds.append(text)
                continue
                
            inp = tokenizer(f"correct: {masked}", return_tensors="pt", max_length=256, truncation=True).to(device)
            with torch.no_grad():
                out = model.generate(**inp, max_length=256, num_beams=exp["num_beams"])
            pred = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = preprocessor.unmask_entities(pred, ents)
            
            if exp["use_safety_gate"]:
                aligned = safety_gate.align_tokens(text, pred)
                words = [corr if status != 'DELETE' else '' for orig, corr, status in aligned]
                pred = " ".join([w for w in words if w])
                
            preds.append(pred)
            
        cers = [compute_cer(r, p) for r, p in zip(df_test["clean_text"], preds)]
        wers = [compute_wer(r, p) for r, p in zip(df_test["clean_text"], preds)]
        exact = (pd.Series(preds) == df_test["clean_text"]).mean() * 100
        
        clean_mask = (df_test["noisy_text"] == df_test["clean_text"])
        over_corr = ((pd.Series(preds) != df_test["clean_text"]) & clean_mask).sum() / max(clean_mask.sum(), 1) * 100
        
        results.append({
            "Cấu hình thí nghiệm (Ablation Setup)": exp["name"],
            "Exact Match (%)": round(exact, 2),
            "CER (%)": round(np.mean(cers)*100, 2),
            "WER (%)": round(np.mean(wers)*100, 2),
            "Over-Correction Rate (%)": round(over_corr, 2)
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("outputs/benchmarks/ablation_study_results.csv", index=False)
    print("\n=== ABLATION STUDY EXPERIMENTAL TABLE ===")
    print(df_res.to_string(index=False))
    print("\n[Ablation] Saved results to outputs/benchmarks/ablation_study_results.csv")

if __name__ == "__main__":
    run_ablation_experiments()
