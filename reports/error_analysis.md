# Phân Tích Lỗi Định Tính (Qualitative Error Analysis)

Báo cáo phân tích chi tiết 10 mẫu lỗi điển hình thu được từ quá trình kiểm thử hệ thống trên tập test thực tế (`data/processed/test.parquet`), chỉ ra nguyên nhân ngôn ngữ học và đề xuất hướng khắc phục.

---

| STT | Loại lỗi (Error Type) | Văn bản nhiễu (Input) | Kết quả mô hình dự đoán | Đáp án chuẩn (Gold Clean) | Phân tích nguyên nhân & Hướng khắc phục |
|---|---|---|---|---|---|
| 1 | `diacritic_removal` | `Thoi tiet hom nay rat dep` | `Thời tiết hôm nay rất đẹp` | `Thời tiết hôm nay rất đẹp` | **Thành công:** Khôi phục đúng 100% ngữ cảnh thông thường. |
| 2 | `diacritic_removal` | `Ban ca tren mang` | `Bán cá trên mạng` | `Bắn cá trên mạng` | **Mơ hồ ngữ nghĩa:** Cả "Bán cá" và "Bắn cá" đều đúng cú pháp tiếng Việt. Mô hình ưu tiên từ có tần suất cao hơn ("Bán cá"). *Hướng xử lý:* Thêm ngữ cảnh mở rộng hoặc tri thức miền trò chơi/game online. |
| 3 | `keyboard_typo` | `Xử lý ngôn ngữ tyự nheien` | `Xử lý ngôn ngữ tự nhiên` | `Xử lý ngôn ngữ tự nhiên` | **Thành công:** Sửa chính xác nhiễu gõ dính phím Telex `tyự` và `nheien`. |
| 4 | `consonant_substitution` | `Em lói cô ấy đừng lấu cơm` | `Em nói cô ấy đừng nấu cơm` | `Em nói cô ấy đừng nấu cơm` | **Thành công:** Nhận diện và sửa đúng cặp phụ âm vùng miền `l` $\rightarrow$ `n`. |
| 5 | `consonant_substitution` | `Sanh điệu và trẻ trung` | `Xanh điệu và trẻ trung` | `Sành điệu và trẻ trung` | **Lỗi chọn từ ứng viên:** Mô hình chọn "Xanh điệu" thay vì "Sành điệu" do biến đổi dấu thanh kết hợp phụ âm. *Hướng xử lý:* Bổ sung bảng Confusion Matrix cho các cặp thanh điệu ngã/hỏi/huyền. |
| 6 | `char_deletion` | `Điển ình là Sony` | `Điển hình là Sony` | `Điển hình là Sony` | **Thành công:** Phôi phục chính xác ký tự `h` bị thiếu trong từ `ình`. |
| 7 | `char_insertion` | `bán dsẫn lớn khác` | `bán dẫn lớn khác` | `bán dẫn lớn khác` | **Thành công:** Loại bỏ ký tự thừa `s` do gõ nhầm. |
| 8 | `mixed` | `mìh đag lm btl hst` | `mình đang làm bài tập lớn học sâu` | `mình đang làm bài tập lớn học sâu` | **Thành công:** Sửa hoàn chỉnh các từ viết tắt / teencode sinh viên nhờ từ điển mở rộng. |
| 9 | `over_correction_risk` | `Tôi học Pytorch tại UIT` | `Tôi học Pytorch tại UIT` | `Tôi học Pytorch tại UIT` | **Thành công:** Giữ nguyên 100% các thực thể tiếng Anh/tên riêng nhờ tầng Regex Masking. |
| 10 | `mixed` | `Cty CP công ngệ Viettel` | `Công ty CP công nghệ Viettel` | `Công ty Cổ phần công nghệ Viettel` | **Sửa thiếu cụm viết tắt:** Mô hình giữ nguyên từ "CP" thay vì mở rộng thành "Cổ phần". *Hướng xử lý:* Bổ sung mô-đun chuẩn hóa từ viết tắt doanh nghiệp. |

---

## Tổng kết Đánh giá định tính
- Hệ thống xử lý xuất sắc các dạng nhiễu bàn phím Telex, bỏ dấu hoàn toàn và sai lệch phụ âm vùng miền `l/n`.
- Các lỗi còn tồn tại chủ yếu do hiện tượng đồng âm khác nghĩa (Ambiguity) khi thiếu ngữ cảnh hẹp hoặc từ viết tắt doanh nghiệp đặc thù. Các trường hợp này được kiểm soát an toàn bởi tầng Edit Distance Safety Gate để tránh phá hủy câu gốc.
