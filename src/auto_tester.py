import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from evaluate_correction import ProductionMultiStagePipeline, compute_cer, compute_wer

def run_automated_test_suite():
    print("=========================================================")
    print("AUTOMATED TEST SUITE: VIETNAMESE SPELL CORRECTION SYSTEM")
    print("=========================================================")
    
    pipeline = ProductionMultiStagePipeline("outputs/models/best_model")
    
    # Test cases representing specific Vietnamese linguistic challenges
    unit_tests = [
        {"category": "Diacritic Removal", "noisy": "Hoc deep learning tai Dai hoc Nam Can Tho", "expected": "Học deep learning tại Đại học Nam Cần Thơ"},
        {"category": "Keyboard Typo", "noisy": "Điển ình là Sony, IBM vq̀ một số công tt công nghệ", "expected": "Điển hình là Sony, IBM và một số công ty công nghệ"},
        {"category": "Consonant Substitution (l/n)", "noisy": "Em lói cô ấy đừng lấu cơm tối nay nhé", "expected": "Em nói cô ấy đừng nấu cơm tối nay nhé"},
        {"category": "Consonant Substitution (s/x)", "noisy": "Sanh điệu và trẻ trung", "expected": "Sành điệu và trẻ trung"},
        {"category": "Consonant Substitution (ch/tr)", "noisy": "Chung thành với tổ quốc", "expected": "Trung thành với tổ quốc"},
        {"category": "Entity Protection (URL/Email)", "noisy": "Truy cập https://univ.edu.vn hoặc lh admin@univ.edu.vn", "expected": "Truy cập https://univ.edu.vn hoặc lh admin@univ.edu.vn"},
        {"category": "Clean Sentence (Over-correction test)", "noisy": "Tôi đi học bằng xe máy điện hàng ngày.", "expected": "Tôi đi học bằng xe máy điện hàng ngày."}
    ]
    
    print("\n--- 1. UNIT TEST RESULTS ---")
    passed = 0
    for test in unit_tests:
        pred = pipeline.process(test["noisy"])
        # Check basic similarity
        cer = compute_cer(test["expected"], pred)
        is_pass = cer < 0.2
        if is_pass: passed += 1
        print(f"[{'PASS' if is_pass else 'FAIL'}] [{test['category']}]")
        print(f"   Input:    {test['noisy']}")
        print(f"   Output:   {pred}")
        print(f"   Expected: {test['expected']}\n")
        
    print(f"Unit Tests Passed: {passed}/{len(unit_tests)} ({passed/len(unit_tests)*100:.1f}%)\n")
    
    print("--- 2. BULK TEST SET EVALUATION (300 Samples) ---")
    df_test = pd.read_parquet("data/processed/test.parquet").head(300)
    preds = [pipeline.process(t) for t in df_test["noisy_text"]]
    df_test["predicted"] = preds
    
    exact_match = (df_test["predicted"] == df_test["clean_text"]).mean() * 100
    cers = [compute_cer(r, p) for r, p in zip(df_test["clean_text"], df_test["predicted"])]
    wers = [compute_wer(r, p) for r, p in zip(df_test["clean_text"], df_test["predicted"])]
    
    clean_mask = (df_test["noisy_text"] == df_test["clean_text"])
    over_corr = ((df_test["predicted"] != df_test["clean_text"]) & clean_mask).sum() / max(clean_mask.sum(), 1) * 100
    
    print(f"Exact Match Accuracy: {exact_match:.2f}%")
    print(f"Character Error Rate (CER): {np.mean(cers)*100:.2f}%")
    print(f"Word Error Rate (WER): {np.mean(wers)*100:.2f}%")
    print(f"Over-Correction Rate: {over_corr:.2f}%")
    print("=========================================================")

if __name__ == "__main__":
    run_automated_test_suite()
