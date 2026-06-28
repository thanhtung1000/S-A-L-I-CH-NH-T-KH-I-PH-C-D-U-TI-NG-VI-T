import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from evaluate_correction import ProductionMultiStagePipeline, compute_cer, compute_wer

def analyze_per_category_errors(test_parquet="data/processed/test.parquet", num_samples=1000):
    print("=========================================================")
    print("PER-CATEGORY ERROR BREAKDOWN ANALYSIS (TEST SET)")
    print("=========================================================")
    
    df_test = pd.read_parquet(test_parquet)
    if num_samples and len(df_test) > num_samples:
        df_test = df_test.head(num_samples).copy()
        
    pipeline = ProductionMultiStagePipeline("outputs/models/best_model")
    
    print(f"[Analysis] Evaluating {len(df_test)} samples across error types...")
    preds = [pipeline.process(text) for text in df_test["noisy_text"]]
    df_test["predicted"] = preds
    
    df_test["cer"] = [compute_cer(r, p) for r, p in zip(df_test["clean_text"], df_test["predicted"])]
    df_test["wer"] = [compute_wer(r, p) for r, p in zip(df_test["clean_text"], df_test["predicted"])]
    df_test["exact_match"] = (df_test["predicted"] == df_test["clean_text"])
    
    # Category aggregation
    categories = df_test["error_type"].unique()
    breakdown = []
    
    for cat in categories:
        sub = df_test[df_test["error_type"] == cat]
        em_rate = sub["exact_match"].mean() * 100
        mean_cer = sub["cer"].mean() * 100
        mean_wer = sub["wer"].mean() * 100
        
        # Over-correction check for clean samples
        if cat == "none":
            over_corr_rate = ((sub["predicted"] != sub["clean_text"]).mean()) * 100
        else:
            over_corr_rate = 0.0
            
        breakdown.append({
            "Loại lỗi (Error Type)": cat,
            "Số lượng mẫu": len(sub),
            "Exact Match (%)": round(em_rate, 2),
            "CER (%)": round(mean_cer, 2),
            "WER (%)": round(mean_wer, 2),
            "Over-Correction Rate (%)": round(over_corr_rate, 2)
        })
        
    df_res = pd.DataFrame(breakdown)
    os.makedirs("outputs/benchmarks", exist_ok=True)
    df_res.to_csv("outputs/benchmarks/per_category_error_breakdown.csv", index=False)
    
    print("\n=== PER-CATEGORY PERFORMANCE TABLE ===")
    print(df_res.to_string(index=False))
    print("\n[Analysis] Detailed report saved to outputs/benchmarks/per_category_error_breakdown.csv")

if __name__ == "__main__":
    analyze_per_category_errors(num_samples=1000)
