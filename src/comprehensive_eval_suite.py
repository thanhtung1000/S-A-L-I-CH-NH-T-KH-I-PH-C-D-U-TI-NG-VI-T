import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.append("C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2/src")

from evaluate_correction import ProductionMultiStagePipeline

def run_multi_scenario_benchmark():
    print("=========================================================")
    print("COMPREHENSIVE 30-SAMPLE MULTI-SCENARIO BENCHMARK SUITE")
    print("=========================================================")
    
    pipeline = ProductionMultiStagePipeline("C:/Users/Lenovo/Desktop/Nam4_HK3/NLP/BTL/BTL_Tung2/outputs/models/best_model")
    
    scenarios = [
        # Scenario 1: Short Sentences (Câu ngắn)
        {"cat": "1. Short Sentences", "input": "Hôm lay trời đẹp quá.", "target": "Hôm nay trời đẹp quá."},
        {"cat": "1. Short Sentences", "input": "Tôi đag làm bài tập.", "target": "Tôi đang làm bài tập."},
        {"cat": "1. Short Sentences", "input": "Mô hìng hoạt động ổn.", "target": "Mô hình hoạt động ổn."},
        {"cat": "1. Short Sentences", "input": "Ăn cơm xong đi chơi.", "target": "Ăn cơm xong đi chơi."},
        
        # Scenario 2: Long Sentences (Câu dài)
        {"cat": "2. Long Sentences", "input": "Trong quá trình học tập tại đại học, chúng em đã cùng nhau nghiên cứu các thuật toán xử lý ngôn ngữ tự nhiên hiện đại.", "target": "Trong quá trình học tập tại đại học, chúng em đã cùng nhau nghiên cứu các thuật toán xử lý ngôn ngữ tự nhiên hiện đại."},
        {"cat": "2. Long Sentences", "input": "Hệ thống có khả năng tự động phát hiện lỗi gõ nhầm và gợi ý từ thay thế chuẩn xác dựa trên ngữ cảnh toàn câu.", "target": "Hệ thống có khả năng tự động phát hiện lỗi gõ nhầm và gợi ý từ thay thế chuẩn xác dựa trên ngữ cảnh toàn câu."},
        
        # Scenario 3: Spelling & Telex Typos Only (Chỉ sai chính tả)
        {"cat": "3. Spelling Typos Only", "input": "Em lói cô ấy đừng lấu cơm tối nay nhé.", "target": "Em nói cô ấy đừng nấu cơm tối nay nhé."},
        {"cat": "3. Spelling Typos Only", "input": "Học sinh trung học đag sữa bài kiểm tra.", "target": "Học sinh trung học đang sửa bài kiểm tra."},
        {"cat": "3. Spelling Typos Only", "input": "Kết quã đánh giá vẩn đạt mức khá.", "target": "Kết quả đánh giá vẫn đạt mức khá."},
        
        # Scenario 4: Full Diacritic Restoration Only (Chỉ mất dấu)
        {"cat": "4. Diacritic Restoration Only", "input": "mo hinh hoat dong kha on dinh", "target": "mô hình hoạt động khá ổn định"},
        {"cat": "4. Diacritic Restoration Only", "input": "hoc deep learning tai dai hoc nam can tho", "target": "học deep learning tại đại học nam cần thơ"},
        {"cat": "4. Diacritic Restoration Only", "input": "sinh vien khoa cong nghe thong tin", "target": "sinh viên khoa công nghệ thông tin"},
        
        # Scenario 5: Mixed Errors (Lỗi hỗn hợp)
        {"cat": "5. Mixed Errors", "input": "mo hinh hẹ thống sữa lỗi đag phực phục", "target": "mô hình hệ thống sửa lỗi đang phục hồi"},
        {"cat": "5. Mixed Errors", "input": "hoc sinh lấu cơm xong vẩn đag lói chuyện", "target": "học sinh nấu cơm xong vẫn đang nói chuyện"},
        
        # Scenario 6: URLs, Emails & Technical Terms (Có URL/Email/Từ chuyên ngành)
        {"cat": "6. Technical Entities", "input": "Nhóm sử dụng PyTorch và Transformers trên GPU.", "target": "Nhóm sử dụng PyTorch và Transformers trên GPU."},
        {"cat": "6. Technical Entities", "input": "Liên hệ email support@company.vn hoặc web https://huggingface.co nhé.", "target": "Liên hệ email support@company.vn hoặc web https://huggingface.co nhé."},
        {"cat": "6. Technical Entities", "input": "Mô hình ViT5 kết hợp Beam Search đạt hiệu năng tốt.", "target": "Mô hình ViT5 kết hợp Beam Search đạt hiệu năng tốt."},
        
        # Scenario 7: Names, Locations & Numbers (Có Tên người, Địa danh, Số liệu)
        {"cat": "7. Names & Locations", "input": "Nguyễn Văn A sinh ngày 15/08/2002 tại Hà Nội.", "target": "Nguyễn Văn A sinh ngày 15/08/2002 tại Hà Nội."},
        {"cat": "7. Names & Locations", "input": "Trường Đại học Cần Thơ có hơn 40.000 sinh viên.", "target": "Trường Đại học Cần Thơ có hơn 40.000 sinh viên."}
    ]
    
    results = []
    passed = 0
    
    for idx, item in enumerate(scenarios, 1):
        inp = item["input"]
        output = pipeline.process(inp)
        
        # Check pass if output contains main target keywords
        is_pass = True
        target_words = item["target"].split()
        out_words = output.split()
        
        # Compute word overlap accuracy
        match_count = sum([1 for w in target_words if w.lower() in output.lower()])
        acc = match_count / len(target_words)
        
        if acc >= 0.85:
            passed += 1
            status = "PASSED ✅"
        else:
            status = "NEEDS_IMP ⚠️"
            
        print(f"[{item['cat']}] Test #{idx} -> {status} (Accuracy: {acc*100:.1f}%)")
        print(f"  In : {inp}")
        print(f"  Out: {output}")
        
        results.append({
            "ID": idx,
            "Category": item["cat"],
            "Input": inp,
            "Output": output,
            "Target": item["target"],
            "Accuracy": round(acc * 100, 1),
            "Status": status
        })
        
    print(f"\n=========================================================")
    print(f"BENCHMARK SUMMARY: {passed}/{len(scenarios)} PASSED ({passed/len(scenarios)*100:.1f}%)")
    print("=========================================================")
    
    df = pd.DataFrame(results)
    os.makedirs("outputs/benchmarks", exist_ok=True)
    df.to_csv("outputs/benchmarks/multi_scenario_evaluation_30.csv", index=False)

if __name__ == "__main__":
    run_multi_scenario_benchmark()
