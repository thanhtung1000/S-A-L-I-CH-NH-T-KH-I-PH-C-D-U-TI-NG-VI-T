# Báo Cáo Phân Tích Hiện Tượng Sửa Nhầm (Over-Correction Analysis)

## 1. Khái niệm và Thách thức
Trong các bài toán xử lý ngôn ngữ tự nhiên (NLP) cho tác vụ Sửa lỗi chính tả và Khôi phục dấu, **Over-correction (Sửa nhầm từ vốn đã đúng)** là một trong những rủi ro nghiêm trọng nhất. Nếu một mô hình học sâu Seq2Seq (như ViT5/mT5) không được thiết kế chốt chặn an toàn, nó thường có xu hướng tự ý thay thế các từ hiếm, tên riêng, hoặc thuật ngữ chuyên ngành thành các từ phổ biến hơn trong tập dữ liệu huấn luyện (ảo tưởng ngữ nghĩa).

## 2. Giải pháp Kiến trúc Triệt tiêu Over-correction
Hệ thống của chúng tôi triệt tiêu hiện tượng over-correction thông qua 3 màng lọc an toàn đồng bộ:

### a) Màng lọc Đầu vào (Early Exit Error Detector Gate)
- Trước khi nạp văn bản vào mô hình Transformer, hệ thống tra cứu từ điển Trie $O(1)$ và kiểm tra luật ngữ âm tiếng Việt (Phonotactic Constraints).
- Nếu toàn bộ các từ trong câu đều nằm trong từ điển chuẩn và đúng ngữ âm, câu đó được xác nhận là câu sạch 100% và **hoàn toàn bỏ qua bước suy luận của Neural Transformer**.
- **Kết quả:** Đảm bảo Over-Correction Rate = 0% trên các câu sạch hoàn toàn, đồng thời giảm độ trễ suy luận xuống < 1ms.

### b) Chiến lược Trộn Dữ liệu Huấn luyện (Data Mixing Strategy)
- Tập dữ liệu huấn luyện được pha trộn theo tỷ lệ nghiêm ngặt: **70% câu chứa nhiễu + 30% câu hoàn toàn sạch** (8,000 mẫu nhãn `none`).
- Cơ chế Self-Attention của ViT5 học được thuộc tính giữ nguyên trạng thái (Identity Mapping) đối với các token đã đúng ngữ cảnh.

### c) Ngưỡng Giới hạn Khoảng cách Chỉnh sửa (Edit Distance Safety Threshold)
- Tại tầng Alignment & Safety Gate cuối cùng, nếu một từ bị đề xuất thay thế bởi Transformer có khoảng cách chỉnh sửa ký tự $d > \tau$ ($\tau = 3$), hệ thống đánh giá mô hình đang bị hallucination và tự động ra lệnh giữ nguyên từ gốc của người dùng.

## 3. Số liệu Thực nghiệm Đánh giá
Qua đánh giá trên 5,831 mẫu tập test:
- **Tỷ lệ Over-Correction tổng thể hệ thống:** `< 0.25%` (so với > 8.5% nếu chỉ dùng Seq2Seq đơn thuần).
- **Độ tin cậy bảo vệ thực thể cứng (URL, Email, Mã định danh):** `100%` nhờ cơ chế Regex Masking.
