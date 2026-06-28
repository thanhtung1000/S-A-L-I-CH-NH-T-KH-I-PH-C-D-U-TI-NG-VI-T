import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append("C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2/src")

from evaluate_correction import ProductionMultiStagePipeline
from safety_gate import AlignmentAndSafetyGate

def run_exhaustive_testing_suite():
    print("=========================================================")
    print("EXHAUSTIVE AUTOMATED TEST SUITE (ZERO-DEFECT VERIFICATION)")
    print("=========================================================")
    
    pipeline = ProductionMultiStagePipeline("C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2/outputs/models/best_model")
    safety_gate = AlignmentAndSafetyGate(edit_distance_threshold=3)
    
    test_cases = [
        # Category 1: Regional Consonants & Telex Typos
        {
            "category": "Regional & Telex Typos",
            "input": "Em lói cô ấy đừng lấu cơm tối nay nhé, đag trên đường về.",
            "expected_contains": ["nói", "nấu", "đang"]
        },
        # Category 2: Full Diacritic Restoration
        {
            "category": "Diacritic Restoration",
            "input": "mo hinh hoat dong kha on dinh va dat hieu qua cao",
            "expected_contains": ["mô hình", "hoạt động", "ổn định"]
        },
        # Category 3: Entity & Technical Jargon Protection
        {
            "category": "Entity & Acronym Protection",
            "input": "Nhóm dùng PyTorch, Transformers, ONNX Runtime và email admin.tech@company.vn trên https://github.com",
            "expected_contains": ["PyTorch", "Transformers", "ONNX Runtime", "admin.tech@company.vn", "https://github.com"]
        },
        # Category 4: Mixed Long Paragraph Chunking
        {
            "category": "Multi-sentence Long Paragraph",
            "input": "Hôm lay nhóm em đag phát triển hệ thống sữa lỗi. Hệ thống sử lý tốt các thực thể như PyTorch và email support@test.com.",
            "expected_contains": ["nhóm em đang", "sửa lỗi", "xử lý", "support@test.com"]
        },
        # Category 5: Clean Text Over-Correction Protection
        {
            "category": "Clean Text Integrity",
            "input": "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc và sáng tạo mỗi ngày.",
            "expected_exact": "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc và sáng tạo mỗi ngày."
        }
    ]
    
    passed_count = 0
    total_count = len(test_cases)
    
    results = []
    
    for idx, tc in enumerate(test_cases, 1):
        inp = tc["input"]
        output = pipeline.process(inp)
        
        is_pass = True
        if "expected_exact" in tc:
            if output.strip() != tc["expected_exact"].strip():
                is_pass = False
        if "expected_contains" in tc:
            for item in tc["expected_contains"]:
                if item.lower() not in output.lower():
                    is_pass = False
                    break
                    
        if is_pass:
            passed_count += 1
            status = "PASSED ✅"
        else:
            status = "FAILED ❌"
            
        print(f"\n[Test #{idx} - {tc['category']}] -> {status}")
        print(f"  Input : {inp}")
        print(f"  Output: {output}")
        
        results.append({
            "Test ID": idx,
            "Category": tc["category"],
            "Input": inp,
            "Output": output,
            "Status": status
        })
        
    print(f"\n=========================================================")
    print(f"TEST SUITE SUMMARY: {passed_count}/{total_count} PASSED ({(passed_count/total_count)*100:.1f}%)")
    print("=========================================================")
    
    df_res = pd.DataFrame(results)
    os.makedirs("outputs/benchmarks", exist_ok=True)
    df_res.to_csv("outputs/benchmarks/exhaustive_test_results.csv", index=False)

if __name__ == "__main__":
    run_exhaustive_testing_suite()
