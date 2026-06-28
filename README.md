# 🇻🇳 Multi-stage ViT5 Vietnamese Spell Corrector & Diacritic Restorer

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005FE6?style=for-the-badge&logo=onnx&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![W&B](https://img.shields.io/badge/Weights_%26_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)

Hệ thống công nghiệp đa tầng (Multi-Stage Industrial Pipeline) kết hợp mô hình ngôn ngữ Seq2Seq **ViT5-base** (220M tham số), thuật toán từ điển Trie $O(1)$, bộ lọc an toàn ngữ cảnh (Safety Alignment Gate) và tối ưu hóa suy luận **ONNX Runtime INT8**. Đồ án giải quyết trọn vẹn hai bài toán cốt lõi: **Sửa lỗi chính tả & gõ nhầm QWERTY/vùng miền** và **Khôi phục hoàn toàn dấu thanh tiếng Việt**.

---

## 📋 Mục Lục
- [✨ Tính Năng Nổi Bật](#-tính-năng-nổi-bật)
- [📊 Kết Quả Thực Nghiệm & Đánh Giá](#-kết-quả-thực-nghiệm--đánh-giá)
- [🏛️ Kiến Trúc Hệ Thống (5-Stage Pipeline)](#-kiến-trúc-hệ-thống-5-stage-pipeline)
- [📁 Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [🚀 Hướng Dẫn Cài Đặt & Sử Dụng](#-hướng-dẫn-cài-đặt--sử-dụng)
- [🛠️ Quá Trình Huấn Luyện & Đánh Giá](#-quá-trình-huấn-luyện--đánh-giá)
- [📈 Nhật Ký W&B (Weights & Biases)](#-nhật-ký-wb-weights--biases)

---

## ✨ Tính Năng Nổi Bật

- **Khôi Phục Dấu Tiếng Việt Tự Nhiên**: Tự động khôi phục dấu câu ngữ cảnh phức tạp từ văn bản không dấu (ví dụ: `Hoc deep learning` ➔ `Học deep learning`).
- **Sửa Lỗi Chính Tả Đa Dạng**: Xử lý hiệu quả lỗi gõ phím kề QWERTY (`vq̀` ➔ `và`), lỗi phụ âm vùng miền (`ch/tr`, `l/n`), rơi rớt ký tự (`Điển ình` ➔ `Điển hình`) và teencode.
- **Bảo Toàn Thực Thể & Định Dạng (Entity Anchoring)**: Tự động khóa và bảo vệ Email, URL, tên công nghệ tiếng Anh (như `PyTorch`, `ONNX`, `Sony`, `IBM`) không bị mô hình sửa nhầm.
- **Cơ Chế Khóa An Toàn (Alignment & Safety Gate)**: Ngăn chặn triệt để hiện tượng sửa thừa (Over-correction) và ảo giác (Hallucination) của mô hình Seq2Seq bằng thuật toán so khớp Levenshtein mức từ.
- **Tối Ưu Hóa Suy Luận ONNX INT8**: Tốc độ suy luận cực nhanh **10.86 ms/câu** trên CPU, giảm kích thước mô hình xuống 2.01 lần.
- **Giao Diện Web Demo Streamlit Sang Trọng**: Hỗ trợ tương tác trực quan, tô màu Highlight đánh dấu từ được sửa đổi, phân tích thời gian thực.

---

## 📊 Kết Quả Thực Nghiệm & Đánh Giá

Đánh giá định lượng trên tập kiểm thử độc lập 1,000 mẫu (`data/processed/test.parquet`) và bộ 19 kịch bản thực tế:

| Chỉ số Đánh giá (Metric) | Baseline (Edit Distance / N-gram) | ViT5 Multi-Stage Pipeline | Cải Thiện / Trạng Thái |
|---|---|---|---|
| **Multi-Scenario Benchmark** | 42.1% PASSED | **89.5% - 94.7% PASSED** | 🚀 **Xuất Sắc** |
| **Exact Match (Khớp 100% câu)** | 23.58% | **65.70%** | 📈 **+42.12%** |
| **Character Error Rate (CER)** | 5.72% | **3.16% - 4.48%** | 📉 **-2.56%** (Tốt) |
| **Word Error Rate (WER)** | 18.55% | **5.01% - 6.37%** | 📉 **-12.18%** (Rất Tốt < 10%) |
| **Correction Precision** | 34.30% | **81.37%** | 🎯 **+47.07%** |
| **Correction F1-Score** | 44.86% | **86.86%** | 🏆 **+42.00%** |
| **Tốc độ suy luận CPU (ONNX)** | ~59.52 ms/câu | **10.86 ms/câu** | ⚡ **Nhanh gấp 5.5 lần** |

---

## 🏛️ Kiến Trúc Hệ Thống (5-Stage Pipeline)

Hệ thống hoạt động theo mô hình công nghiệp 5 tầng khép kín:

```
[Văn bản Đầu vào Nhiễu]
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 1: Unicode Sanitization & Entity Anchoring          │
│ - Chuẩn hóa NFC & làm sạch ký tự lạ                       │
│ - Anchor bảo vệ Email, URL, Tên công nghệ tiếng Anh       │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 2 & 3: Error Detection Gate & Candidate Generator  │
│ - Tra cứu từ điển Trie O(1) & luật âm tiết tiếng Việt     │
│ - Sinh ứng viên từ phím kề QWERTY & cặp phụ âm vùng miền │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 4: Neural Contextual Seq2Seq Reranking (ViT5)       │
│ - Giải mã Beam Search (num_beams=4, repetition_penalty=1.2)│
│ - Chunking câu dài theo ranh giới ngắt câu tự nhiên       │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│ Stage 5: Alignment & Safety Gate Thresholding             │
│ - So khớp Levenshtein mức từ & kiểm soátEdit Distance (τ) │
│ - Phục hồi từ gốc nếu ViT5 vi phạm ngưỡng an toàn         │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
[Văn bản Đầu ra Hoàn Chỉnh & Chuẩn Xác]
```

---

## 📁 Cấu Trúc Thư Mục

```
.
├── demo_spell_corrector.py          # Script khởi chạy Giao diện Web Streamlit / CLI Demo
├── requirements.txt                 # Danh sách các thư viện phụ thuộc
├── data_card.md                     # Báo cáo chi tiết về Bộ dữ liệu VNSearch-Spell-Correct-2026
├── noise_generation_rules.md        # Quy tắc sinh nhiễu dữ liệu tự động
├── wandb_report_link.txt            # Đường dẫn & tóm tắt 5 runs trên Weights & Biases
├── config/                          # Các tệp cấu hình tham số hệ thống
├── data/                            # Thư mục dữ liệu (Train/Dev/Test dạng Parquet)
├── outputs/                         # Thư mục chứa Models checkpoint, ONNX export & Metrics
│   ├── models/best_model/           # Weights mô hình ViT5 tốt nhất
│   └── onnx/                        # Mô hình ONNX đã lượng tử hóa INT8
├── reports/                         # Thư mục chứa các báo cáo kỹ thuật chuyên sâu
│   ├── FINAL_SUBMISSION_PROJECT_REPORT.md  # Báo cáo tổng kết đồ án
│   ├── ablation_study_and_error_breakdown.md
│   ├── convergence_and_loss_analysis.md
│   ├── deployment_analysis.md
│   ├── error_analysis.md
│   └── over_correction_analysis.md
└── src/                             # Thư mục mã nguồn chính của dự án
    ├── preprocess.py                # Tiền xử lý dữ liệu & sinh nhiễu
    ├── train_baseline_correction.py # Huấn luyện mô hình Baseline N-gram
    ├── train_seq2seq_correction.py  # Fine-tune mô hình ViT5 Seq2Seq
    ├── evaluate_correction.py       # Pipeline đánh giá cốt lõi
    ├── error_detector.py            # Kiểm tra lỗi dựa trên Trie & luật âm tiết
    ├── candidate_generator.py       # Generator ứng viên sửa lỗi
    ├── safety_gate.py               # Alignment & Safety Gate chống ảo giác
    ├── export_onnx.py               # Xuất & lượng tử hóa mô hình sang ONNX INT8
    ├── benchmark_inference.py       # Đo đạc tốc độ suy luận PyTorch vs ONNX
    ├── fast_inference_service.py    # Service suy luận tối ưu cho production
    ├── unified_master_evaluator.py  # Bộ đánh giá tổng hợp đa chỉ số
    ├── system_progression_evaluator.py # Đánh giá tiến trình cải tiến hệ thống
    └── auto_tester.py               # Bộ tự động test kịch bản tự động
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Yêu Cầu Môi Trường
- Python >= 3.10
- GPU NVIDIA (khuyên dùng cho huấn luyện) hoặc CPU x86_64 (cho suy luận ONNX)

### 2. Cài Đặt Thư Viện
Kích hoạt môi trường virtualenv/conda và cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

### 3. Khởi Chạy Giao Diện Web Demo (Streamlit App)
Chạy ứng dụng Web tương tác sang trọng bằng Streamlit:
```bash
streamlit run demo_spell_corrector.py
```
Sau khi chạy, truy cập đường dẫn local: `http://localhost:8501` trên trình duyệt.

### 4. Chạy Demo Giao Diện Dòng Lệnh (CLI Demo)
Chạy thử nghiệm nhanh trên terminal:
```bash
python demo_spell_corrector.py --cli
```

---

## 🛠️ Quá Trình Huấn Luyện & Đánh Giá

### Huấn Luyện Mô Hình Baseline N-gram
```bash
python src/train_baseline_correction.py
```

### Fine-tune Mô Hình ViT5 Seq2Seq
```bash
python src/train_seq2seq_correction.py
```

### Đánh Giá Định Lượng Toàn Diện (Master Evaluator)
```bash
python src/unified_master_evaluator.py
```

### Xuất Mô Hình ONNX & Lượng Tử Hóa INT8
```bash
python src/export_onnx.py
```

### Benchmark Tốc Độ Suy Luận CPU (PyTorch vs ONNX Runtime)
```bash
python src/benchmark_inference.py
```

---

## 📈 Nhật Ký W&B (Weights & Biases)

Dự án theo dõi và lưu trữ offline/cloud toàn bộ tiến trình huấn luyện và đánh giá trên **Weights & Biases**.

- **🔗 Project Dashboard Link**: [W&B Spell Correction Project](https://wandb.ai/csc4005-csc4007-khmt16-01-spell-correction)
- **Tóm tắt 5 Runs bắt buộc**:
  1. `run_01_rule_based_baseline`: Mô hình Baseline Edit Distance & N-gram LM.
  2. `run_02_vit5_small_default`: Fine-tune mô hình baseline `VietAI/vit5-small`.
  3. `run_03_vit5_base_model`: Fine-tune mô hình ngữ cảnh `VietAI/vit5-base`.
  4. `run_04_hyperparameter_tuned`: Tuning thuật toán giải mã Beam Search (`num_beams=4`, `repetition_penalty=1.2`).
  5. `run_05_best_multistage_pipeline`: Mô hình hoàn chỉnh kết hợp ViT5-base và Safety Alignment Gate.

---

## 📜 Giấy Phép & Đồ Án
Báo cáo và mã nguồn được xây dựng phục vụ nghiệm thu Đồ án môn học **Xử Lý Ngôn Ngữ Tự Nhiên (CSC4005 / CSC4007)** - Đề tài 10.
