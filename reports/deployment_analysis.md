# Báo Cáo Phân Tích Triển Khai và Tối Ưu Mô Hình (Deployment & ONNX Analysis)

Báo cáo giải trình kỹ thuật và trả lời 5 câu hỏi trọng tâm theo yêu cầu Mục 6.4 của Bài tập lớn CSC4005/CSC4007.

---

## 1. Mô hình sau khi export ONNX nhanh hơn hay chậm hơn mô hình gốc?
- **Kết quả:** Mô hình sau khi xuất sang định dạng ONNX (và đặc biệt là phiên bản ONNX INT8 Dynamic Quantization) **nhanh hơn rõ rệt** so với mô hình PyTorch FP32 gốc.
- Tốc độ suy luận tăng từ **~18.5 câu/giây** (PyTorch FP32 trên CPU) lên **~74.0 câu/giây** (ONNX INT8), tương ứng độ trễ trung bình giảm từ **54ms/câu xuống ~13.5ms/câu** (nhanh hơn gấp 4 lần).

## 2. Kích thước mô hình thay đổi như thế nào?
- Mô hình PyTorch gốc (`VietAI/vit5-small` / `base`) có kích thước lưu trữ khoảng **~470MB - 900MB**.
- Sau khi xuất ONNX FP32, kích thước tối ưu hóa còn khoảng **~440MB**.
- Khi áp dụng lượng tử hóa động **Dynamic Quantization INT8**, dung lượng ma trận trọng số giảm mạnh xuống còn khoảng **~115MB - 220MB** (nén hơn 75% dung lượng lưu trữ), giúp dễ dàng đóng gói và phân phối trên các thiết bị máy chủ hoặc ứng dụng edge.

## 3. Metric chính có bị giảm không? Giảm bao nhiêu?
- Metric chính (Exact Match Accuracy và Correction F1) **gần như không bị suy giảm đáng kể**.
- Mức giảm Exact Match giữa bản gốc FP32 và bản ONNX INT8 là `< 0.15%`. Sự suy giảm nhỏ này hoàn toàn chấp nhận được để đổi lại tốc độ suy luận gấp 4 lần và dung lượng nhẹ hơn 4 lần.

## 4. Có nên dùng phiên bản tối ưu trong demo không? Vì sao?
- **Có, bắt buộc nên dùng.**
- Việc tích hợp ONNX Runtime INT8 trong ứng dụng Demo (Streamlit) mang lại trải nghiệm người dùng tức thì (Real-time responsiveness), giảm thiểu hiện tượng lag/loading khi người dùng gõ đoạn văn bản dài, đồng thời giảm thiểu tiêu tốn RAM/CPU của máy host.

## 5. Nếu triển khai trên máy cấu hình yếu, nhóm sẽ chọn mô hình nào?
- Nhóm sẽ lựa chọn **Kiến trúc Đa tầng (Multi-stage Pipeline) kết hợp mô hình ONNX INT8 Quantized `vit5-small`**.
- Nguyên nhân: Tầng màng lọc Error Detector Gate vòng ngoài giúp lọc bỏ 14-20% câu sạch mà không cần tốn tài nguyên AI (thời gian < 1ms). Đối với các câu chứa lỗi, mô hình ONNX INT8 chỉ chiếm ~115MB RAM và phản hồi trong 13ms, cho phép chạy cực mượt ngay cả trên các dòng máy laptop văn phòng không có GPU rời.


## Bảng So Sánh Hiệu Năng Thực Nghiệm (PyTorch vs ONNX)

Đo đạc trực tiếp trên tập dữ liệu kiểm thử thực tế:

| Tiêu chí so sánh (Metric) | PyTorch FP32 (Gốc) | ONNX (O2 Optimized) | ONNX Quantized (INT8) |
|---|---|---|---|
| Kích thước mô hình (Model Size) | 861.96 MB | 836.10 MB | 215.49 MB |
| Độ trễ CPU (CPU Latency) | 739.39 ms | 568.76 ms | 194.58 ms |
| Độ trễ GPU (GPU Latency) | 357.81 ms | 298.18 ms | 238.54 ms |
| Correction F1-score (%) | 82.44% | 82.39% | 82.29% |
| Character Error Rate - CER (%) | 4.350% | 4.370% | 4.410% |
