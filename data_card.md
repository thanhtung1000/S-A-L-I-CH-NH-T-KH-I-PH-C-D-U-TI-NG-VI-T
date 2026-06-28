# Data Card: Vietnamese Spell Correction & Diacritic Restoration Dataset (VSEC & Expanded Synthetic Corpus)

## 1. Dataset Summary
Bộ dữ liệu phục vụ đề tài "Sửa lỗi chính tả và khôi phục dấu tiếng Việt cho văn bản nhiễu" (Đề tài 10 - Bài tập lớn CSC4005/CSC4007). Bộ dữ liệu kết hợp từ tập benchmark chuẩn **VSEC (Vietnamese Spell Correction Dataset)** gồm 9,341 câu và bộ dữ liệu mở rộng 56,000 cặp câu nhiễu - câu sạch được tổng hợp từ nhiều nguồn tiếng Việt (tin tức, Wikipedia, tài liệu bài tập lớn, câu tự biên soạn).

## 2. Motivation and Intended Use
- **Mục đích:** Huấn luyện và đánh giá các mô hình sửa lỗi chính tả, xử lý nhiễu bàn phím, lỗi bỏ dấu và sai lệch phụ âm vùng miền trong tiếng Việt.
- **Tác vụ chính:** Spell Correction, Diacritic Restoration, Noise Robustness Evaluation.
- **Đối tượng sử dụng:** Sinh viên, nghiên cứu sinh và các nhà phát triển hệ thống NLP tiếng Việt.

## 3. Dataset Sources
1. **VSEC Benchmark:** Tải từ Hugging Face (`nguyenthanhasia/vsec-vietnamese-spell-correction`).
2. **Báo chí & Tin tức:** Trích xuất từ các trang báo điện tử công khai tiếng Việt.
3. **Wikipedia tiếng Việt:** Trích xuất các đoạn văn bản chuẩn mực ngữ pháp.
4. **Tài liệu học phần & Câu tự biên soạn:** Các mẫu câu chứa từ ngữ chuyên ngành tin học, NLP và teencode sinh viên.

## 4. Dataset Composition
- **Tổng số mẫu:** 56,000 cặp câu (`noisy_text`, `clean_text`).
- **Các trường thông tin chính:**
  - `id`: Định danh mẫu dữ liệu (Integer).
  - `noisy_text`: Văn bản chứa nhiễu/lỗi chính tả/không dấu (String).
  - `clean_text`: Văn bản chuẩn sau khi sửa (String).
  - `error_type`: Phân loại loại nhiễu (String).
  - `source`: Nguồn gốc văn bản gốc (String).
  - `split`: Tập phân chia (`train`, `val`, `test`).

## 5. Data Collection Process
Văn bản sạch từ các nguồn uy tín được thu thập và làm sạch Unicode (chuẩn NFC). Sau đó, quy trình nhiễu hóa nhân tạo được áp dụng dựa trên các mô hình hành vi gõ bàn phím của người Việt thực tế.

## 6. Annotation Process
- Đối với VSEC: Được gán nhãn thủ công ở mức âm tiết bởi các chuyên gia NLP.
- Đối với bộ dữ liệu mở rộng: Quy tắc sinh lỗi được tự động hóa chính xác dựa trên luật ngữ âm và ma trận khoảng cách bàn phím Telex/QWERTY.

## 7. Preprocessing
- Chuẩn hóa toàn bộ chuỗi ký tự về dạng Unicode NFC (`unicodedata.normalize('NFC', text)`).
- Loại bỏ các khoảng trắng thừa và ký tự điều khiển không hợp lệ.
- Bảo vệ các thực thể cứng (URL, Email, viết tắt tiếng Anh) qua cơ chế Regex Masking.

## 8. Train/Validation/Test Split
Bộ dữ liệu 56,000 mẫu được chia theo tỷ lệ 80% / 10% / 10%:
- **Train set:** 44,317 mẫu (~79.1%)
- **Validation (Dev) set:** 5,852 mẫu (~10.5%)
- **Test set:** 5,831 mẫu (~10.4%)

## 9. Label Distribution
Phân bố các loại nhiễu trong bộ dữ liệu (cân bằng 8,000 mẫu/loại):
1. `diacritic_removal`: Bỏ dấu toàn bộ câu (8,000 mẫu).
2. `keyboard_typo`: Gõ nhầm phím liền kề (8,000 mẫu).
3. `char_deletion`: Thiếu ký tự trong từ (8,000 mẫu).
4. `char_insertion`: Thừa ký tự trong từ (8,000 mẫu).
5. `consonant_substitution`: Sai lệch phụ âm vùng miền `l/n`, `ch/tr`, `x/s`, `d/r/gi` (8,000 mẫu).
6. `mixed`: Hỗn hợp nhiều loại lỗi trong cùng một câu (8,000 mẫu).
7. `none`: Câu hoàn toàn sạch (8,000 mẫu - phục vụ kiểm soát over-correction).

## 10. Data Quality Checks
- 100% các mẫu không chứa giá trị null hoặc chuỗi rỗng.
- Tập `none` chiếm 14.43% tổng dữ liệu giúp mô hình học cách giữ nguyên các câu đã đúng.
- Không có sự rò rỉ dữ liệu (data leakage) giữa các tập train, val và test.

## 11. Ethical Considerations
- Dữ liệu hoàn toàn là văn bản công khai hoặc câu giả lập học thuật.
- Không chứa thông tin định danh cá nhân (PII), thông tin nhạy cảm hoặc thù ghét.

## 12. Biases and Limitations
- Mô hình sinh lỗi tập trung vào bàn phím Telex/QWERTY phổ biến, có thể chưa bao phủ hết bàn phím VNI hoặc các từ địa phương cực đoan.
- Giới hạn độ dài câu khuyến nghị dưới 256 token.

## 13. License and Access
- Bộ dữ liệu VSEC tuân thủ giấy phép nguồn mở của nhóm tác giả.
- Bộ dữ liệu tổng hợp phục vụ mục đích học tập và nghiên cứu nội bộ học phần CSC4005/CSC4007.

## 14. Recommended Uses
- Huấn luyện các mô hình Seq2Seq (ViT5, BARTpho, mT5) sửa lỗi chính tả.
- Huấn luyện các bộ lọc kiểm tra an toàn văn bản tiếng Việt.

## 15. Prohibited or Risky Uses
- Không sử dụng làm công cụ chỉnh sửa tự động tuyệt đối trong các văn bản pháp lý chính thức mà không có sự kiểm duyệt của con người.
