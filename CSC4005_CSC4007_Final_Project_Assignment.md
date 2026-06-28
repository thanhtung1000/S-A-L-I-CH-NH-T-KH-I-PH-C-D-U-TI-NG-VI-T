> Hướng dẫn thực hiện bài tập lớn cho sinh viên KHMT 16-01

Bài tập lớn dùng chung cho học phần CSC4005 – Học sâu và CSC4007 – Xử lý
ngôn ngữ tự nhiên

> ThS. Lê Thị Thùy Trang
>
> 2026-04-25

**1** **Giới** **thiệu** **bài** **tập** **lớn**

**1.1** **Bối** **cảnh**

Trong học kỳ này, sinh viên lớp KHMT 16-01 học đồng thời hai học phần
**CSC4005** **–** **Học** **sâu** và **CSC4007** **–** **Xử** **lý**
**ngôn** **ngữ** **tự** **nhiên**. Thay vì giao hai bài tập lớn tách
rời, sinh viên thực hiện một đề tài chung có quy mô lớn hơn. Đề tài phải
thể hiện được cả năng lực thiết kế bài toán NLP lẫn năng lực xây dựng,
huấn luyện và đánh giá mô hình học sâu.

Bài tập lớn được thiết kế theo định hướng **mini-capstone**. Sinh viên
không chỉ chạy mô hình để có một con số kết quả, mà phải xây dựng một
pipeline có thể tái lập, gồm dữ liệu, Data Card, baseline, mô hình học
sâu, thí nghiệm W&B, đánh giá định lượng, phân tích lỗi và demo.

**1.2** **Nguyên** **tắc** **thiết** **kế** **đề** **tài**

> Một đề tài hợp lệ cần thỏa mãn các nguyên tắc sau:
>
> • Có **dataset** **rõ** **ràng** hoặc có quy trình tự xây dựng dữ liệu
> hợp lệ.
>
> • Nguồn dữ liệu phải **gắn** **trực** **tiếp** **với** **bài**
> **toán**; không dùng dataset chỉ để liệt kê cho có.
>
> • Có **Data** **Card** mô tả nguồn gốc, cấu trúc, nhãn, giới hạn,
> license và rủi ro của dữ liệu.
>
> • Có **baseline** để so sánh với mô hình học sâu.
>
> • Có **mô** **hình** **học** **sâu** **hoặc** **Transformer** được
> huấn luyện/fine-tune.
>
> • Có **W&B** **tracking** để log thí nghiệm, metric, hyperparameter và
> artifact.
>
> • Có **triển** **khai** **mô** **hình** ở mức tối thiểu: xuất ONNX,
> benchmark suy luận và phân tích khả năng tối ưu/nén mô hình.
>
> • Có **phân** **tích** **lỗi** dựa trên mẫu dự đoán thật.
>
> • Có **demo** **chạy** **được** và báo cáo kỹ thuật cuối kỳ.

Nhóm không được chỉ xây dựng demo gọi API hoặc prompt đơn thuần. Bắt
buộc phải có dữ liệu, Data Card, baseline, mô hình học sâu hoặc
fine-tuning, thí nghiệm được log bằng W&B, đánh giá định lượng, phân
tích lỗi định tính và bước triển khai mô hình tối thiểu bằng
ONNX/benchmark suy luận.

> 1

CSC4005 & CSC4007 Lê Thị Thùy Trang

**2** **Mục** **tiêu** **học** **tập**

> Sau khi hoàn thành bài tập lớn, sinh viên có thể:
>
> 1\. Xác định một bài toán NLP có đầu vào, đầu ra, nhãn và metric rõ
> ràng.
>
> 2\. Lựa chọn hoặc xây dựng dataset phù hợp với bài toán.
>
> 3\. Viết Data Card để tài liệu hóa dữ liệu theo hướng minh bạch và có
> trách nhiệm.
>
> 4\. Xây dựng baseline và giải thích vai trò của baseline.
>
> 5\. Huấn luyện hoặc fine-tune mô hình học sâu cho bài toán NLP.
>
> 6\. Sử dụng W&B để log hyperparameters, loss, metrics, artifacts và so
> sánh runs.
>
> 7\. Xuất mô hình sang ONNX, benchmark suy luận và đề xuất hướng tối
> ưu/nén mô hình phù hợp.
>
> 8\. Đánh giá mô hình bằng metric phù hợp.
>
> 9\. Phân tích lỗi theo hiện tượng ngôn ngữ và đặc điểm dữ liệu.
>
> 10\. Xây dựng demo đơn giản cho hệ thống.
>
> 11\. Viết báo cáo kỹ thuật có bằng chứng và có khả năng tái lập.

**3** **Quy** **định** **chung**

**3.1** **Tổ** **chức** **nhóm**

> • Mỗi nhóm gồm 2–4 sinh viên.
>
> • Mỗi nhóm chọn một đề tài trong danh sách đề tài bên dưới.
>
> • Một đề tài nên có tối đa 1–2 nhóm chọn. Nếu nhiều nhóm chọn cùng đề
> tài, bắt buộc phải khác dataset mở rộng, hướng mô hình hoặc yêu cầu
> nâng cao.

**3.2** **Mức** **tối** **thiểu** **để** **được** **nghiệm** **thu**

> Một nhóm chỉ được coi là hoàn thành nếu có đủ:
>
> 1\. Dataset hoặc nguồn dữ liệu rõ ràng, đúng với đề tài.
>
> 2\. File data_card.md.
>
> 3\. Baseline model.
>
> 4\. Deep learning/Transformer model hoặc mô hình học sâu tự xây.
>
> 5\. Tối thiểu 5 W&B runs.
>
> 6\. Xuất được mô hình tốt nhất sang ONNX hoặc có giải trình kỹ thuật
> nếu mô hình/giải pháp không thể xuất ONNX trực tiếp.
>
> 7\. Có benchmark suy luận trước và sau khi export/tối ưu mô hình.
>
> 8\. Bảng kết quả định lượng.
>
> 9\. Phân tích lỗi tối thiểu 10 mẫu.
>
> 10\. Demo chạy được.
>
> 11\. Báo cáo cuối kỳ.
>
> 2

CSC4005 & CSC4007 Lê Thị Thùy Trang

> 12\. Link GitHub repository.

**3.3** **Cấu** **trúc** **thư** **mục** **khuyến** **nghị**

project-name/ README.md data_card.md requirements.txt configs/

> baseline.yaml transformer.yaml
>
> data/ raw/
>
> processed/ annotations/ splits/
>
> train.csv dev.csv test.csv
>
> notebooks/ src/
>
> preprocess.py train_baseline.py train_model.py evaluate.py infer.py
> export_onnx.py
>
> benchmark_inference.py app.py
>
> outputs/ metrics/ figures/ predictions/ benchmarks/ models/ onnx/
>
> reports/ error_analysis.md final_report.pdf
>
> wandb_report_link.txt

**3.4** **Cài** **đặt** **môi** **trường** **tham** **khảo**

conda create -n csc4005-4007-project python=3.10 -y conda activate
csc4005-4007-project

pip install -r requirements.txt wandb login

> Một file requirements.txt tối thiểu có thể gồm:
>
> 3

CSC4005 & CSC4007 Lê Thị Thùy Trang

torch transformers datasets scikit-learn pandas

numpy matplotlib wandb evaluate onnx onnxruntime optimum psutil
rouge-score streamlit gradio

sentence-transformers faiss-cpu

**4** **Danh** **sách** **đề** **tài**

> Bảng sau chỉ liệt kê dataset chính gắn với từng đề tài. Chi tiết yêu
> cầu nằm ở các mục tiếp theo.

||
||
||
||
||
||
||
||
||
||
||
||
||
||

> 4

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.1** **Đề** **tài** **1.** **Trợ** **lý** **RAG** **hỗ** **trợ**
**sinh** **viên** **học** **CSC4005** **và** **CSC4007**

**4.1.1** **Mục** **tiêu**

Xây dựng một hệ thống hỏi đáp hỗ trợ sinh viên tra cứu thông tin về hai
học phần CSC4005 và CSC4007. Hệ thống có thể trả lời các câu hỏi về nội
dung học phần, lịch học, bài tập lớn, tiêu chí chấm điểm, tài liệu học
tập, hướng dẫn cài đặt môi trường và câu hỏi thường gặp.

**4.1.2** **Dataset** **và** **nguồn** **dữ** **liệu**

**Nguồn** **dữ** **liệu** **chính**: tài liệu học phần do giảng viên
cung cấp, gồm đề cương học phần, slide bài giảng, rubric, README GitHub
Classroom, hướng dẫn thực hành, FAQ và thông báo lớp học nếu được phép
sử dụng.

**Nguồn** **tham** **khảo** **phù** **hợp**: UIT-ViQuAD2.0 dùng để tham
khảo cấu trúc bài toán hỏi đáp tiếng Việt, không thay thế dữ liệu học
phần. [Link.](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)

**Dữ** **liệu** **tự** **xây** **bắt** **buộc**: nhóm tạo tối thiểu 150
câu hỏi đánh giá. Mỗi câu hỏi cần có câu trả lời chuẩn, đoạn tài liệu
nguồn, mã tài liệu và nhãn answerable/unanswerable nếu có.

**4.1.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

Tài liệu học phần

→ chia đoạn/chunking → tạo embedding

→ lưu index

→ truy hồi top-k đoạn liên quan → sinh câu trả lời

→ hiển thị câu trả lời kèm nguồn

> Nhóm phải có:
>
> • BM25 hoặc TF-IDF retrieval làm baseline;
>
> • dense retrieval bằng Sentence-BERT hoặc multilingual embedding;
>
> • so sánh ít nhất 2 cách chunking;
>
> • reranking hoặc scoring lại top-k;
>
> • câu trả lời có trích dẫn đoạn nguồn;
>
> • cơ chế trả lời *không* *đủ* *thông* *tin* nếu không tìm thấy context
> phù hợp.

**4.1.4** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Query rewriting cho câu hỏi mơ hồ.
>
> 2\. Phát hiện câu hỏi ngoài phạm vi.
>
> 3\. Đánh giá hallucination rate.
>
> 4\. Giao diện chat bằng Streamlit/Gradio.
>
> 5\. Lưu lịch sử hỏi đáp và phản hồi của người dùng.
>
> 5

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.1.5** **W&B** **và** **đánh** **giá**

> Tối thiểu 5 runs:
>
> • BM25 baseline;
>
> • dense retrieval baseline;
>
> • dense retrieval với chunk size nhỏ;
>
> • dense retrieval với chunk size lớn;
>
> • dense retrieval + reranking.
>
> Metric bắt buộc: [Recall@3,](mailto:Recall@3)
> [Recall@5,](mailto:Recall@5) MRR, answer correctness, hallucination
> rate.

**4.1.6** **Sản** **phẩm** **nộp**

\- data_card.md

\- rag_pipeline.py hoặc rag_pipeline.ipynb - evaluation_set.csv

\- retrieval_results.csv - wandb_report_link.txt - demo_app.py

\- final_report.pdf

**4.2** **Đề** **tài** **2.** **Hệ** **thống** **xác** **minh** **phát**
**biểu** **và** **phát** **hiện** **tin** **giả** **tiếng** **Việt**
**dựa** **trên** **bằng** **chứng**

**4.2.1** **Mục** **tiêu**

Xây dựng hệ thống kiểm chứng một phát biểu tiếng Việt dựa trên bằng
chứng. Hệ thống cần phân loại phát biểu thành Supported, Refuted hoặc
Not Enough Information, đồng thời hiển thị đoạn bằng chứng liên quan và
giải thích ngắn gọn.

**4.2.2** **Dataset** **và** **nguồn** **dữ** **liệu**

**Dataset** **chính**: ViFactCheck – benchmark fact-checking tiếng Việt
với các cặp claim–evidence, nhãn Supported, Refuted, NEI
[Link.](https://huggingface.co/datasets/tranthaihoa/vifactcheck)

> Các trường dữ liệu cần khai thác:

Statement, Context, Evidence, Topic, Author, Url, labels

**Không** **khuyến** **nghị**: dùng dataset sentiment hoặc QA thông
thường cho đề tài này, vì chúng không có cấu trúc
claim–evidence–verdict.

**4.2.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

Claim

→ truy hồi evidence từ context → ghép claim + evidence

→ phân loại verdict

→ hiển thị evidence và giải thích

> 6

CSC4005 & CSC4007 Lê Thị Thùy Trang

> Nhóm phải có:
>
> • evidence retrieval bằng BM25/TF-IDF;
>
> • claim verification bằng PhoBERT, XLM-R hoặc Transformer encoder;
>
> • so sánh giữa dùng full context và dùng gold evidence;
>
> • confusion matrix cho 3 nhãn;
>
> • phân tích lỗi theo topic hoặc loại claim.

**4.2.4** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Bi-encoder retrieval + cross-encoder reranking.
>
> 2\. Sinh explanation ngắn cho verdict.
>
> 3\. Đánh giá evidence recall.
>
> 4\. Phân tích lỗi theo chủ đề tin tức.
>
> 5\. Kiểm tra robustness với claim được paraphrase.

**4.2.5** **W&B** **và** **đánh** **giá**

> Tối thiểu 5 runs:
>
> • TF-IDF + Logistic Regression baseline;
>
> • PhoBERT với full context;
>
> • PhoBERT với gold evidence;
>
> • XLM-R hoặc model khác;
>
> • best model với tuned hyperparameters.
>
> Metric: Accuracy, Macro-F1, Per-class F1, Evidence
> [Recall@k,](mailto:Recall@k) Confusion matrix.

**4.2.6** **Sản** **phẩm** **nộp**

\- data_card.md

\- evidence_retrieval.py

\- claim_verification_train.py - evaluate_claims.py

\- error_analysis.md

\- wandb_report_link.txt - demo_app.py

> 7

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.3** **Đề** **tài** **3.** **Sinh** **tự** **động** **Data** **Card**
**cho** **bộ** **dữ** **liệu** **tiếng** **Việt**

**4.3.1** **Mục** **tiêu**

Xây dựng hệ thống hỗ trợ sinh bản nháp Data Card cho một dataset tiếng
Việt. Đầu vào có thể là metadata, vài dòng mẫu dữ liệu, thống kê nhãn và
mô tả nguồn dữ liệu. Đầu ra là một Data Card có cấu trúc.

**4.3.2** **Dataset** **và** **nguồn** **dữ** **liệu**

**Nguồn** **dữ** **liệu** **chính**: chính các dataset cards, README,
metadata và sample của các dataset tiếng Việt. Vì đề tài này học cách
sinh Data Card, việc dùng nhiều dataset là phù hợp, nhưng mỗi dataset
phải được dùng như một mẫu đầu vào hoặc gold reference.

> Nhóm chọn 6–10 dataset từ các nguồn sau, ưu tiên các dataset đã có
> dataset card hoặc README rõ ràng:
>
> • UIT-ViQuAD2.0:
> [Link.](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)
>
> • ViFactCheck:
> [Link.](https://huggingface.co/datasets/tranthaihoa/vifactcheck)
>
> • ViBidLQA:
> [Link.](https://huggingface.co/datasets/ntphuc149/ViBidLQA)
>
> • UIT-ViCTSD: [Link.](https://huggingface.co/datasets/tarudesu/ViCTSD)
>
> • VSEC:
> [Link.](https://huggingface.co/datasets/nguyenthanhasia/vsec-vietnamese-spell-correction)
>
> • 8Opt Vietnamese Summarization:
> [Link.](https://huggingface.co/datasets/8Opt/vietnamese-summarization-dataset-0003)

**Không** **yêu** **cầu**: dùng các dataset này để train task gốc. Chúng
chỉ là dữ liệu đầu vào/nguồn tham chiếu để sinh Data Card.

**4.3.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

Dataset metadata/sample

→ trích xuất thông tin quan trọng → mapping vào template Data Card →
sinh bản nháp Data Card

→ kiểm tra mục còn thiếu

> Nhóm phải có:
>
> • template Data Card cố định;
>
> • module trích xuất metadata;
>
> • module sinh văn bản cho từng mục;
>
> • completeness checker;
>
> • giao diện nhập file CSV hoặc metadata JSON.

**4.3.4** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Chấm điểm chất lượng Data Card theo rubric.
>
> 2\. Gợi ý câu hỏi cần bổ sung khi metadata thiếu.
>
> 3\. Cảnh báo rủi ro dữ liệu cá nhân, bias hoặc license không rõ.
>
> 8
>
> CSC4005 & CSC4007 Lê Thị Thùy Trang
>
> 4\. So sánh generated Data Card với gold Data Card.
>
> 5\. Cho phép xuất kết quả ra Markdown.
>
> **4.3.5** **W&B** **và** **đánh** **giá**
>
> Metric: completeness score, ROUGE/BERTScore nếu đánh giá phần sinh văn
> bản, human evaluation theo rubric, precision/recall cho phát hiện mục
> thiếu nếu có nhãn.
>
> **4.3.6** **Sản** **phẩm** **nộp**
>
> \- data_card_template.md - gold_data_cards/
>
> \- generated_data_cards/ - metadata_samples/
>
> \- data_card_generator.py - evaluation_rubric.md
>
> \- wandb_report_link.txt - demo_app.py
>
> **4.4** **Đề** **tài** **4.** **Hỏi** **đáp** **văn** **bản** **pháp**
> **lý/hành** **chính** **tiếng** **Việt** **có** **trích** **dẫn**
> **nguồn**
>
> **4.4.1** **Mục** **tiêu**
>
> Xây dựng hệ thống hỏi đáp trên văn bản pháp lý hoặc hành chính tiếng
> Việt. Hệ thống phải trả lời câu hỏi và chỉ rõ câu trả lời dựa trên
> điều, khoản hoặc đoạn nào.
>
> **4.4.2** **Dataset** **và** **nguồn** **dữ** **liệu**

**Dataset** **chính**: ViBidLQA – dataset hỏi đáp pháp lý tiếng Việt
trong lĩnh vực Luật Đấu thầu, có cả extractive QA và abstractive QA.
[Link.](https://huggingface.co/datasets/ntphuc149/ViBidLQA)

> **Nguồn** **tự** **xây** **phù** **hợp**: văn bản quy chế đào tạo, quy
> định học vụ, thủ tục hành chính công khai, văn bản pháp luật/hành
> chính công khai. Nếu tự xây, nhóm cần ít nhất 100 câu hỏi, mỗi câu hỏi
> gắn với đoạn nguồn và đáp án chuẩn.
>
> **Không** **khuyến** **nghị**: dùng UIT-ViQuAD làm dataset chính cho
> đề tài này vì UIT-ViQuAD là QA trên Wikipedia, không đại diện cho văn
> bản pháp lý/hành chính.
>
> **4.4.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**
>
> Văn bản pháp lý/hành chính → chia điều/khoản/đoạn
>
> → index tài liệu → nhận câu hỏi
>
> → truy hồi đoạn liên quan → trả lời có căn cứ
>
> Nhóm phải có:
>
> • BM25 baseline;
>
> • dense retrieval;
>
> 9
>
> CSC4005 & CSC4007 Lê Thị Thùy Trang
>
> • extractive QA hoặc generative QA;
>
> • câu trả lời kèm nguồn;
>
> • cơ chế từ chối khi không đủ căn cứ.
>
> **4.4.4** **Yêu** **cầu** **nâng** **cao**
>
> Chọn ít nhất 2:
>
> 1\. So sánh extractive QA và generative QA.
>
> 2\. Kiểm tra answer groundedness.
>
> 3\. Xử lý câu hỏi đa bước.
>
> 4\. Hiển thị điều/khoản liên quan trong giao diện.
>
> 5\. Cảnh báo hệ thống không thay thế tư vấn pháp lý chính thức.
>
> **4.4.5** **W&B** **và** **đánh** **giá**
>
> Metric: Exact Match, Token-level F1 hoặc Span-level F1,
> [Recall@k,](mailto:Recall@k) answer groundedness, human evaluation.
>
> **4.4.6** **Sản** **phẩm** **nộp**
>
> \- legal_corpus/ - qa_dataset.csv - data_card.md
>
> \- retrieval_eval.csv - qa_eval.csv
>
> \- qa_pipeline.py
>
> \- wandb_report_link.txt - demo_app.py
>
> **4.5** **Đề** **tài** **5.** **Phát** **hiện** **bình** **luận**
> **độc** **hại/cyberbullying** **tiếng** **Việt** **có** **giải**
> **thích**
>
> **4.5.1** **Mục** **tiêu**
>
> Xây dựng hệ thống phát hiện bình luận độc hại trong tiếng Việt. Hệ
> thống không chỉ trả nhãn mà còn cần giải thích vì sao bình luận bị coi
> là độc hại.
>
> **4.5.2** **Dataset** **và** **nguồn** **dữ** **liệu**
>
> **Dataset** **chính**: UIT-ViCTSD – dataset phát hiện
> constructive/toxic speech tiếng Việt.
> [Link.](https://huggingface.co/datasets/tarudesu/ViCTSD)

**Nguồn** **thay** **thế**: UIT-ViHSD hoặc dataset hate speech tiếng
Việt tương đương nếu nhóm chứng minh được nguồn và điều kiện sử dụng rõ
ràng.

> **Nguồn** **tự** **xây**: bình luận giả lập hoặc bình luận công khai
> đã ẩn danh, không chứa thông tin cá nhân, chỉ dùng trong phạm vi học
> tập.
>
> 10

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.5.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

Comment

→ chuẩn hóa teencode/emoji/ký tự lặp

→ phân loại toxic/non-toxic hoặc clean/offensive/hate → giải thích
từ/cụm từ gây độc hại

> Nhóm phải có:
>
> • baseline: TF-IDF + SVM/Logistic Regression;
>
> • deep learning: PhoBERT/XLM-R hoặc CNN/BiLSTM;
>
> • explainability: attention, LIME, SHAP hoặc highlight rule-based;
>
> • robustness test với teencode hoặc viết tắt.

**4.5.4** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Phân loại nhiều mức: clean/offensive/hate.
>
> 2\. Phát hiện span độc hại.
>
> 3\. Kiểm tra mô hình với dữ liệu bị biến đổi ký tự.
>
> 4\. Phân tích bias theo nhóm từ nhạy cảm.
>
> 5\. Dashboard thống kê loại độc hại.

**4.5.5** **W&B** **và** **đánh** **giá**

Metric: Accuracy, Macro-F1, Precision/Recall cho lớp toxic/hate,
confusion matrix, span-level F1 nếu có phát hiện span.

**4.5.6** **Sản** **phẩm** **nộp**

\- toxicity_dataset.csv - data_card.md

\- text_normalization.py - train_classifier.py

\- explainability_report.md - robustness_test.csv

\- wandb_report_link.txt - demo_app.py

**4.6** **Đề** **tài** **6.** **Tóm** **tắt** **tin** **tức** **tiếng**
**Việt** **có** **kiểm** **soát** **độ** **dài** **và** **từ** **khóa**

**4.6.1** **Mục** **tiêu**

Xây dựng hệ thống tóm tắt văn bản tiếng Việt. Hệ thống hỗ trợ ít nhất
hai chế độ: tóm tắt ngắn 1–2 câu, tóm tắt chi tiết 3–5 câu, và tóm tắt
có giữ từ khóa bắt buộc nếu nhóm chọn yêu cầu nâng cao.

> 11

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.6.2** **Dataset** **và** **nguồn** **dữ** **liệu**

**Dataset** **chính**: 8Opt Vietnamese Document Summarization Dataset.
Dataset có các trường document, summary, keywords, phù hợp trực tiếp cho
summarization và kiểm soát từ khóa.
[Link.](https://huggingface.co/datasets/8Opt/vietnamese-summarization-dataset-0003)

**Nguồn** **tự** **xây**: tin tức công khai nếu nhóm ghi rõ URL, ngày
truy cập và chỉ dùng trong phạm vi học tập; hoặc bộ dữ liệu do nhóm tự
tạo với document, summary, keywords.

> **Không** **khuyến** **nghị**: dùng dataset classification/sentiment
> cho đề tài này vì không có summary tham chiếu.

**4.6.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

Document

→ tiền xử lý

→ baseline extractive summarization → abstractive summarization

→ kiểm tra độ dài và keyword preservation

> Nhóm phải có:
>
> • baseline: Lead-k hoặc TextRank;
>
> • deep learning: ViT5, mT5, BARTpho hoặc seq2seq model;
>
> • đánh giá ROUGE;
>
> • kiểm tra độ dài summary;
>
> • lưu generated summaries để phân tích lỗi.

**4.6.4** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Tóm tắt theo độ dài người dùng chọn.
>
> 2\. Tóm tắt giữ từ khóa bắt buộc.
>
> 3\. So sánh extractive và abstractive summarization.
>
> 4\. Kiểm tra factual consistency.
>
> 5\. Giao diện nhập bài báo và nhận tóm tắt.

**4.6.5** **W&B** **và** **đánh** **giá**

Log tối thiểu: training loss, validation loss, ROUGE-1, ROUGE-2,
ROUGE-L, keyword preservation rate, average summary length.

**4.6.6** **Sản** **phẩm** **nộp**

\- summarization_dataset_sample.csv - data_card.md

\- train_seq2seq.py

\- evaluate_summary.py

\- generated_summaries.csv

> 12

CSC4005 & CSC4007 Lê Thị Thùy Trang

\- wandb_report_link.txt - demo_app.py

**4.7** **Đề** **tài** **7.** **Trợ** **lý** **đọc** **hiểu** **bài**
**báo** **khoa** **học:** **tóm** **tắt,** **trích** **xuất** **đóng**
**góp** **và** **lập** **bảng** **literature** **review**

**4.7.1** **Mục** **tiêu**

Xây dựng hệ thống hỗ trợ đọc bài báo khoa học. Đầu vào là title,
abstract hoặc đoạn introduction. Đầu ra gồm tóm tắt ngắn, vấn đề nghiên
cứu, phương pháp chính, dataset sử dụng, metric chính, đóng góp chính,
hạn chế và bảng literature review.

**4.7.2** **Dataset** **và** **nguồn** **dữ** **liệu**

> **Dataset** **chính** **nếu** **làm** **tiếng** **Anh**: SciTLDR –
> dataset extreme summarization cho tài liệu khoa học.
> [Link.](https://huggingface.co/datasets/allenai/scitldr)

**Nguồn** **tự** **xây** **nếu** **muốn** **làm** **tiếng** **Việt**:
abstract bài báo tiếng Việt/open-access, kỷ yếu hội nghị, tạp chí trường
hoặc paper do giảng viên cung cấp. Nhóm cần tối thiểu 50 abstract, mỗi
mẫu có nhãn thủ công cho method, dataset, metric, contribution.

**Lưu** **ý**: SciTLDR phù hợp cho tóm tắt bài báo khoa học, nhưng không
phải dataset tiếng Việt. Nếu dùng SciTLDR, nhóm cần nêu rõ giới hạn ngôn
ngữ trong Data Card.

**4.7.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

> Hệ thống phải có ba module:
>
> • Summarization: sinh tóm tắt ngắn;
>
> • Information extraction: trích xuất method, dataset, metric,
> contribution;
>
> • Literature table: xuất bảng so sánh nhiều paper.
>
> Mô hình tối thiểu:
>
> • baseline: keyword/rule-based extraction;
>
> • deep learning: Transformer classifier, sequence labeling hoặc
> seq2seq;
>
> • có đánh giá riêng cho summarization và extraction.

**4.7.4** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Phân loại lĩnh vực bài báo.
>
> 2\. Tự động tạo bullet contribution.
>
> 3\. Tạo bảng so sánh nhiều paper.
>
> 4\. Phát hiện paper thiếu dataset/metric rõ ràng.
>
> 5\. Hỗ trợ cả tiếng Việt và tiếng Anh.
>
> 13

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.7.5** **W&B** **và** **đánh** **giá**

Metric: ROUGE cho summarization, F1 cho extraction nếu có nhãn, Exact
Match cho dataset/metric extraction, human evaluation cho literature
table.

**4.7.6** **Sản** **phẩm** **nộp**

\- paper_abstract_dataset.csv - data_card.md

\- annotation_guideline.md - extractor.py

\- summarizer.py

\- literature_table_generator.py - wandb_report_link.txt

\- final_report.pdf

**4.8** **Đề** **tài** **8.** **Phát** **hiện** **văn** **bản**
**tiếng** **Việt** **do** **AI** **sinh** **ra**

**4.8.1** **Mục** **tiêu**

Xây dựng hệ thống phân loại một đoạn văn/bài báo tiếng Việt là do con
người viết hay do mô hình AI/LLM sinh ra. Hệ thống cần đánh giá được khả
năng phát hiện văn bản AI, giải thích đặc trưng mô hình dựa vào và kiểm
tra độ bền khi văn bản bị chỉnh sửa nhẹ.

**4.8.2** **Dataset** **và** **nguồn** **dữ** **liệu**

**Dataset** **chính**: ICCIES-2025-DetectAI/vietnamese_news_human_ai.
Dataset có 200.000 mẫu tin tức tiếng Việt, gồm văn bản do người viết và
văn bản do AI sinh.
[Link.](https://huggingface.co/datasets/ICCIES-2025-DetectAI/vietnamese_news_human_ai)

**Nguồn** **tự** **xây**: bài báo/tin tức công khai tiếng Việt và văn
bản AI sinh bởi nhiều LLM khác nhau theo cùng chủ đề. Nếu tự xây, nhóm
cần ít nhất 1.000 mẫu, cân bằng tương đối giữa hai lớp human và ai.

Cấu trúc dữ liệu tối thiểu:
**id**,text,label,source,topic,model_name,prompt_id,created_at

**4.8.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

Văn bản đầu vào

→ làm sạch và chuẩn hóa → tách train/dev/test

→ huấn luyện baseline

→ fine-tune mô hình học sâu → đánh giá trên test set

→ kiểm tra robustness

→ giải thích lỗi và demo

> Nhóm phải có:
>
> 1\. Tiền xử lý: loại HTML nếu có, chuẩn hóa khoảng trắng, xử lý dấu
> câu, chuẩn hóa Unicode tiếng Việt.
>
> 2\. EDA: thống kê số mẫu, độ dài văn bản, phân bố nhãn, phân bố chủ
> đề.
>
> 14

CSC4005 & CSC4007 Lê Thị Thùy Trang

> 3\. Baseline: TF-IDF + Logistic Regression hoặc TF-IDF + Linear SVM.
>
> 4\. Mô hình học sâu: fine-tune PhoBERT, XLM-R, mBERT hoặc encoder
> tiếng Việt phù hợp.
>
> 5\. Robustness test: ít nhất 100 mẫu bị biến đổi nhẹ như paraphrase,
> thêm/xóa câu, thêm lỗi chính tả, thay từ đồng nghĩa.
>
> 6\. Explainability: LIME, SHAP, attention visualization hoặc phân tích
> thủ công dựa trên mẫu sai.

**4.8.4** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Phân tích domain shift: train trên tin tức, test trên bài viết
> giáo dục hoặc mạng xã hội.
>
> 2\. Phân tích ảnh hưởng của độ dài văn bản.
>
> 3\. So sánh phát hiện AI theo từng mô hình sinh văn bản nếu nhóm tự
> tạo dữ liệu.
>
> 4\. Thử contrastive learning hoặc sentence embedding + classifier.
>
> 5\. Demo nhập văn bản và hiển thị nhãn, xác suất, cảnh báo độ tin cậy.

**4.8.5** **W&B** **và** **đánh** **giá**

> Tối thiểu 5 runs:

run_01_tfidf_logreg run_02_tfidf_svm run_03_phobert_base
run_04_xlmr_base run_05_best_model_robustness

> Metric: Accuracy, Macro-F1, Precision, Recall, ROC-AUC, robustness
> drop.

**4.8.6** **Sản** **phẩm** **nộp**

\- data_card.md

\- dataset_statistics.ipynb - train_baseline.py

\- train_transformer.py - robustness_test.py

\- explainability_report.md - predictions_test.csv

\- wandb_report_link.txt - demo_app.py

\- final_report.pdf

**4.9** **Đề** **tài** **9.** **Trích** **xuất** **sự** **kiện** **từ**
**tin** **tức** **tiếng** **Việt** **và** **dựng** **timeline**

**4.9.1** **Mục** **tiêu**

Xây dựng hệ thống trích xuất sự kiện từ văn bản tin tức tiếng Việt. Hệ
thống cần nhận diện được sự kiện, loại sự kiện, thành phần tham gia,
thời gian/địa điểm nếu có, sau đó biểu diễn thành timeline có cấu trúc.

> 15

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.9.2** **Dataset** **và** **nguồn** **dữ** **liệu**

**Dataset** **chính**: BKEE – Bách Khoa Event Extraction, dataset trích
xuất sự kiện tiếng Việt được công bố tại LREC-COLING 2024. Link bài báo:
[Link.](https://aclanthology.org/2024.lrec-main.217/)

**Nguồn** **tự** **xây**: tin tức công khai theo một miền cụ thể như
thiên tai, giáo dục, kinh tế, thể thao hoặc pháp luật. Nếu tự gán nhãn,
nhóm nên chọn 3–5 event types và 100–200 câu/tin ngắn để giảm độ khó.

Nếu tự gán nhãn, nhóm phải có annotation_guideline.md mô tả rõ: event
trigger, event types, argument roles, cách xử lý sự kiện chồng lấp, sự
kiện không có thời gian và câu có nhiều sự kiện.

**4.9.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

Document

→ sentence segmentation → tokenization

→ trigger detection

→ event type classification → argument extraction

→ timeline generation

> Tối thiểu bắt buộc:
>
> 1\. Trigger detection: xác định từ/cụm từ kích hoạt sự kiện.
>
> 2\. Event type classification: phân loại loại sự kiện cho trigger.
>
> 3\. Argument extraction: trích xuất ít nhất người/tổ chức tham gia,
> thời gian, địa điểm.
>
> 4\. Timeline generation: xuất bảng có cột time, event_type, trigger,
> arguments, source_sentence.

**4.9.4** **Mô** **hình** **bắt** **buộc**

> • Baseline: rule-based hoặc CRF/BiLSTM-CRF cho trigger detection.
>
> • Mô hình học sâu: PhoBERT/XLM-R token classification cho trigger và
> argument span.
>
> • Phân loại event type: dùng representation của trigger hoặc
> sentence-level embedding.

**4.9.5** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Joint learning: học đồng thời trigger detection và event type
> classification.
>
> 2\. Phân tích lỗi theo loại sự kiện.
>
> 3\. Chuẩn hóa thời gian tương đối như ”hôm qua”, ”sáng nay”, ”tuần
> trước”.
>
> 4\. Dựng timeline trực quan theo ngày/tháng.
>
> 5\. So sánh mô hình khi dùng/không dùng named entity features.
>
> 16

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.9.6** **W&B** **và** **đánh** **giá**

> Tối thiểu 5 runs:

run_01_rule_based_trigger run_02_bilstm_crf_trigger
run_03_phobert_trigger run_04_phobert_trigger_event_type
run_05_best_model_argument_extraction

> Metric: Trigger F1, Event Type Macro-F1, Argument F1, Timeline
> correctness trên ít nhất 30 sự kiện.

**4.9.7** **Sản** **phẩm** **nộp**

\- data_card.md

\- annotation_guideline.md nếu có tự gán nhãn - preprocess_bkee.py

\- train_trigger_model.py

\- train_event_type_model.py - extract_arguments.py

\- timeline_generator.py - timeline_examples.csv -
wandb_report_link.txt - demo_timeline_app.py - final_report.pdf

**4.10** **Đề** **tài** **10.** **Sửa** **lỗi** **chính** **tả** **và**
**khôi** **phục** **dấu** **tiếng** **Việt** **cho** **văn** **bản**
**nhiễu**

**4.10.1** **Mục** **tiêu**

Xây dựng hệ thống sửa lỗi chính tả tiếng Việt và/hoặc khôi phục dấu cho
văn bản không dấu, sai dấu hoặc nhiễu do gõ nhanh. Hệ thống cần xử lý
lỗi gõ nhầm ký tự, thiếu dấu, sai dấu, lỗi phát âm vùng miền, teencode
đơn giản, lặp ký tự, thiếu/thừa khoảng trắng.

**4.10.2** **Dataset** **và** **nguồn** **dữ** **liệu**

**Dataset** **chính**: VSEC – Vietnamese Spell Correction Dataset.
Dataset có câu sai và câu sửa đúng, kèm annotation ở mức âm tiết.
[Link.](https://huggingface.co/datasets/nguyenthanhasia/vsec-vietnamese-spell-correction)

**Nguồn** **tự** **xây**: văn bản sạch từ tin tức, Wikipedia tiếng Việt,
tài liệu học phần hoặc câu do nhóm tự viết; sau đó tạo dữ liệu nhiễu
bằng quy tắc bỏ dấu, thay ký tự gần bàn phím, xóa ký tự, thêm ký tự,
thay phụ âm đầu/cuối.

Cấu trúc dữ liệu tối thiểu:
**id**,noisy_text,clean_text,error_type,source,split

**4.10.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**

Văn bản nhiễu

→ chuẩn hóa Unicode

→ phát hiện vị trí lỗi

> 17

CSC4005 & CSC4007 Lê Thị Thùy Trang

→ sinh phương án sửa

→ chọn câu sửa tốt nhất → đánh giá với câu chuẩn

> Nhóm phải có ít nhất 2 hướng tiếp cận:
>
> 1\. Baseline rule-based: từ điển, edit distance, quy tắc bỏ/khôi phục
> dấu hoặc n-gram language model.
>
> 2\. Mô hình học sâu: seq2seq model, Transformer encoder-decoder,
> ViT5/mT5 hoặc token classification + correction candidate.

Tối thiểu hệ thống phải hỗ trợ một trong hai chế độ: spell correction
hoặc diacritic restoration. Nhóm khá nên làm cả hai chế độ và so sánh độ
khó.

**4.10.4** **Yêu** **cầu** **nâng** **cao**

> Chọn ít nhất 2:
>
> 1\. Phân loại lỗi trước khi sửa: thiếu dấu, sai dấu, gõ nhầm, viết
> tắt, teencode.
>
> 2\. So sánh character-level model và word/syllable-level model.
>
> 3\. Tạo bộ test riêng cho văn bản mạng xã hội hoặc văn bản sinh viên.
>
> 4\. Đánh giá lỗi sửa sai: mô hình sửa cả từ vốn đã đúng.
>
> 5\. Demo tô màu vị trí lỗi và hiển thị phương án sửa.

**4.10.5** **W&B** **và** **đánh** **giá**

> Tối thiểu 5 runs:

run_01_rule_based_edit_distance run_02_ngram_language_model
run_03_bilstm_seq2seq run_04_vit5_correction
run_05_best_model_error_type_analysis

> Metric: Character Error Rate, Word Error Rate, Correction F1,
> over-correction rate.

**4.10.6** **Sản** **phẩm** **nộp**

\- data_card.md

\- noisy_clean_dataset.csv

\- noise_generation_rules.md nếu tự sinh lỗi -
train_baseline_correction.py

\- train_seq2seq_correction.py - evaluate_correction.py

\- over_correction_analysis.md - wandb_report_link.txt

\- demo_spell_corrector.py - final_report.pdf

> 18
>
> CSC4005 & CSC4007 Lê Thị Thùy Trang
>
> **4.11** **Đề** **tài** **11.** **Trích** **xuất** **từ** **khóa**
> **và** **sinh** **tiêu** **đề** **tự** **động** **cho** **bài**
> **báo** **tiếng** **Việt**
>
> **4.11.1** **Mục** **tiêu**

Xây dựng hệ thống đọc một bài viết tiếng Việt và tạo ra danh sách từ
khóa/chủ đề chính, tiêu đề tự động phù hợp với nội dung, và tùy chọn
tiêu đề trung tính hoặc hấp dẫn nhưng không giật tít.

> **4.11.2** **Dataset** **và** **nguồn** **dữ** **liệu**
>
> **Dataset** **chính**: 8Opt Vietnamese Document Summarization Dataset.
> Dataset có trường document, summary, keywords; trường keywords dùng
> cho keyword extraction, còn summary có thể hỗ trợ headline generation.
> [Link.](https://huggingface.co/datasets/8Opt/vietnamese-summarization-dataset-0003)
>
> **Nguồn** **tự** **xây**: tin tức công khai tiếng Việt có tiêu đề và
> nội dung; nhóm cần ghi rõ URL, nguồn, ngày truy cập và chỉ sử dụng
> trong phạm vi học tập. Có thể bổ sung nhãn headline_style: trung tính,
> học thuật, hấp dẫn, ngắn gọn.
>
> Cấu trúc dữ liệu tối thiểu:
> **id**,document,title,keywords,summary,source,topic,split
>
> **4.11.3** **Yêu** **cầu** **kỹ** **thuật** **tối** **thiểu**
>
> Document
>
> → tiền xử lý văn bản → trích xuất từ khóa → sinh tiêu đề
>
> → kiểm tra tiêu đề có bám nội dung không → đánh giá và demo
>
> Nhóm phải có đủ 2 module:
>
> 1\. Keyword extraction: trích xuất 5–10 từ khóa quan trọng từ văn bản.
>
> 2\. Headline generation: sinh tiêu đề ngắn, đúng nội dung, không bịa
> thêm thông tin.
>
> Mô hình bắt buộc:
>
> • baseline keyword: TF-IDF, TextRank hoặc RAKE;
>
> • baseline headline: dùng câu đầu tiên, rút gọn summary hoặc
> template-based;
>
> • mô hình học sâu: fine-tune ViT5/mT5/BARTpho hoặc mô hình seq2seq
> tương đương để sinh tiêu đề.
>
> **4.11.4** **Yêu** **cầu** **nâng** **cao**
>
> Chọn ít nhất 2:
>
> 1\. Sinh tiêu đề theo độ dài: dưới 12 từ hoặc dưới 18 từ.
>
> 2\. Sinh tiêu đề có chứa ít nhất 1–2 từ khóa chính.
>
> 3\. Phân loại và tránh tiêu đề giật tít.
>
> 4\. So sánh extractive keywords và generated keywords.
>
> 5\. Kiểm tra factual consistency giữa tiêu đề sinh ra và nội dung gốc.
>
> 19

CSC4005 & CSC4007 Lê Thị Thùy Trang

**4.11.5** **W&B** **và** **đánh** **giá**

> Tối thiểu 5 runs:

run_01_tfidf_keywords_lead_title run_02_textrank_keywords_template_title
run_03_vit5_headline_generation
run_04_mt5_or_bartpho_headline_generation
run_05_best_model_keyword_controlled_title

Metric: Keyword Precision/Recall/F1, ROUGE-1/2/L, BLEU hoặc BERTScore
nếu có, tỷ lệ tiêu đề chứa từ khóa chính, độ dài trung bình tiêu đề,
human evaluation ít nhất 50 mẫu.

**4.11.6** **Sản** **phẩm** **nộp**

\- data_card.md

\- keyword_headline_dataset.csv - keyword_extraction.py

\- train_headline_generator.py - evaluate_keywords.py

\- evaluate_headlines.py - human_eval_sheet.csv - wandb_report_link.txt

\- demo_headline_generator.py - final_report.pdf

**5** **Yêu** **cầu** **về** **W&B** **cho** **tất** **cả** **đề**
**tài**

Mỗi nhóm phải tạo một project W&B có tên theo cấu trúc:
csc4005-csc4007-khmt16-01-**\<**topic-short-name**\>**

> Mỗi nhóm có tối thiểu 5 runs:
>
> 1\. Baseline run.
>
> 2\. Deep learning model lần 1.
>
> 3\. Deep learning model lần 2 với thay đổi hyperparameter.
>
> 4\. Deep learning model lần 3 với thay đổi model hoặc preprocessing.
>
> 5\. Best model run.
>
> Ngoài 5 runs tối thiểu, nhóm phải log thêm các thông tin triển khai
> sau vào W&B:
>
> • Best model artifact hoặc đường dẫn checkpoint tốt nhất.
>
> • ONNX model artifact hoặc đường dẫn file ONNX trong repository.
>
> • Bảng benchmark suy luận: thời gian suy luận trung bình, kích thước
> mô hình, metric chính trước/sau export.
>
> • Nếu có tối ưu/nén mô hình, log thêm kết quả của phiên bản
> quantized/distilled/pruned.
>
> Trong báo cáo, nhóm phải có bảng tổng hợp:
>
> 20

CSC4005 & CSC4007 Lê Thị Thùy Trang

||
||
||
||
||
||
||
||

**6** **Yêu** **cầu** **triển** **khai,** **ONNX** **và** **tối** **ưu**
**mô** **hình**

Phần này nhằm gắn bài tập lớn với yêu cầu triển khai mô hình thực tế.
Sinh viên không chỉ huấn luyện mô hình để đạt metric tốt, mà còn phải
đánh giá mô hình có thể đưa vào suy luận nhanh, gọn và tái lập hay
không.

**6.1** **Yêu** **cầu** **bắt** **buộc**

> Mỗi nhóm bắt buộc thực hiện các yêu cầu sau:
>
> 1\. Chọn mô hình tốt nhất sau quá trình thí nghiệm.
>
> 2\. Xuất mô hình tốt nhất sang định dạng **ONNX**.
>
> 3\. Viết script suy luận cho mô hình gốc và mô hình ONNX.
>
> 4\. Benchmark suy luận trên cùng một tập mẫu kiểm thử.
>
> 5\. So sánh tối thiểu 4 tiêu chí: kích thước mô hình, thời gian suy
> luận trung bình, metric chính và mức suy giảm chất lượng nếu có.
>
> 6\. Ghi kết quả benchmark vào W&B và đưa bảng so sánh vào báo cáo.

Nếu mô hình hoặc pipeline không thể xuất ONNX trực tiếp, nhóm phải viết
mục **giải** **trình** **kỹ** **thuật** trong báo cáo, nêu rõ nguyên
nhân và thay thế bằng một trong các phương án sau:

> • Xuất riêng phần encoder/classifier sang ONNX.
>
> • Benchmark mô hình PyTorch/Hugging Face gốc và mô hình rút gọn.
>
> • Dùng TorchScript hoặc một định dạng triển khai tương đương.
>
> • Tối ưu pipeline suy luận bằng batching, caching hoặc giảm độ dài đầu
> vào.

**6.2** **Yêu** **cầu** **nâng** **cao**

> Mỗi nhóm nên thực hiện ít nhất một kỹ thuật tối ưu hoặc nén mô hình
> sau:
>
> • **Quantization**: giảm độ chính xác trọng số, ví dụ FP32 sang INT8.
>
> • **Knowledge** **distillation**: dùng mô hình lớn làm teacher, huấn
> luyện mô hình nhỏ làm student.
>
> • **Pruning**: loại bỏ một phần trọng số, neuron hoặc attention heads
> ít quan trọng.
>
> • **Model** **selection**: so sánh mô hình lớn với mô hình nhỏ hơn, ví
> dụ PhoBERT-base với một encoder nhỏ hơn.
>
> • **Input** **optimization**: giảm max sequence length, lọc câu/đoạn
> đầu vào, caching embedding.
>
> 21

CSC4005 & CSC4007 Lê Thị Thùy Trang

**6.3** **Bảng** **benchmark** **bắt** **buộc**

> Trong báo cáo, nhóm phải có bảng theo mẫu sau:

||
||
||
||
||
||

**6.4** **File** **cần** **có** **trong** **repository**

> Repository của nhóm cần có tối thiểu:

src/export_onnx.py src/benchmark_inference.py outputs/onnx/

outputs/benchmarks/inference_benchmark.csv
reports/deployment_analysis.md

> Trong file reports/deployment_analysis.md, nhóm cần trả lời ngắn gọn:
>
> 1\. Mô hình sau khi export ONNX nhanh hơn hay chậm hơn mô hình gốc?
>
> 2\. Kích thước mô hình thay đổi như thế nào?
>
> 3\. Metric chính có bị giảm không? Giảm bao nhiêu?
>
> 4\. Có nên dùng phiên bản tối ưu trong demo không? Vì sao?
>
> 5\. Nếu triển khai trên máy cấu hình yếu, nhóm sẽ chọn mô hình nào?

**7** **Yêu** **cầu** **về** **Data** **Card**

File data_card.md cần có cấu trúc tối thiểu: *\#* *Data* *Card:* *\<Tên*
*dataset\>*

*\##* *1.* *Dataset* *Summary*

*\##* *2.* *Motivation* *and* *Intended* *Use*

*\##* *3.* *Dataset* *Sources*

*\##* *4.* *Dataset* *Composition*

*\##* *5.* *Data* *Collection* *Process*

*\##* *6.* *Annotation* *Process*

*\##* *7.* *Preprocessing*

*\##* *8.* *Train/Validation/Test* *Split*

> 22

CSC4005 & CSC4007 Lê Thị Thùy Trang

*\##* *9.* *Label* *Distribution*

*\##* *10.* *Data* *Quality* *Checks*

*\##* *11.* *Ethical* *Considerations*

*\##* *12.* *Biases* *and* *Limitations*

*\##* *13.* *License* *and* *Access*

*\##* *14.* *Recommended* *Uses*

*\##* *15.* *Prohibited* *or* *Risky* *Uses*

> Với dataset tự xây, nhóm phải nộp thêm annotation_guideline.md mô tả
> quy tắc gán nhãn.

**8** **Rubric** **chấm** **điểm**

||
||
||
||
||
||
||
||
||
||
||

**9** **Lịch** **trình** **gợi** **ý**

||
||
||
||
||
||
||
||
||
||
||

> 23

CSC4005 & CSC4007 Lê Thị Thùy Trang

**10** **Yêu** **cầu** **nộp** **bài**

> Mỗi nhóm nộp tối thiểu:
>
> • Link GitHub repository;
>
> • File README.md;
>
> • File data_card.md;
>
> • File annotation_guideline.md nếu có dữ liệu tự gán nhãn;
>
> • source code đầy đủ;
>
> • file kết quả metrics_summary.csv hoặc metrics_summary.json;
>
> • file ONNX hoặc thư mục outputs/onnx/;
>
> • file benchmark suy luận outputs/benchmarks/inference_benchmark.csv;
>
> • file phân tích triển khai reports/deployment_analysis.md;
>
> • file dự đoán test_predictions.csv;
>
> • file error_analysis.md;
>
> • link W&B project/report;
>
> • demo app hoặc notebook demo;
>
> • báo cáo cuối kỳ dạng PDF.

**11** **Những** **lỗi** **rất** **hay** **gặp**

> • Không có dataset rõ ràng hoặc dataset không phù hợp với đề tài.
>
> • Liệt kê nhiều dataset nhưng không dùng dataset đó trong pipeline.
>
> • Chỉ dùng prompt/API mà không có baseline và không có mô hình huấn
> luyện.
>
> • Không chia train/validation/test đúng cách.
>
> • Dùng test set để tuning nhiều lần.
>
> • Không có Data Card hoặc Data Card viết quá sơ sài.
>
> • Có W&B nhưng không log hyperparameters.
>
> • Có ONNX file nhưng không benchmark hoặc không so sánh với mô hình
> gốc.
>
> • Chỉ báo accuracy, không phân tích macro-F1 hoặc lỗi theo nhãn.
>
> • Không phân tích lỗi thật mà chỉ nhận xét chung chung.
>
> • Demo chạy được nhưng không có đánh giá định lượng.
>
> • Báo cáo không chỉ ra hạn chế của dữ liệu và mô hình.

**12** **Câu** **hỏi** **thảo** **luận** **cuối** **kỳ**

> Mỗi nhóm chuẩn bị trả lời các câu hỏi sau khi bảo vệ:
>
> 1\. Dataset của nhóm có phù hợp với bài toán không? Vì sao?
>
> 24

CSC4005 & CSC4007 Lê Thị Thùy Trang

> 2\. Dataset có bias hoặc giới hạn gì?
>
> 3\. Baseline của nhóm là gì? Mô hình deep learning có vượt baseline
> không?
>
> 4\. Metric chính có phản ánh đúng chất lượng hệ thống không?
>
> 5\. W&B giúp nhóm rút ra điều gì từ các thí nghiệm?
>
> 6\. Sau khi export ONNX hoặc tối ưu mô hình, tốc độ, kích thước và
> chất lượng thay đổi như thế nào?
>
> 7\. Mô hình sai nhiều nhất ở loại mẫu nào?
>
> 8\. Nếu có thêm 2 tuần, nhóm sẽ cải thiện dữ liệu, mô hình hay đánh
> giá trước?
>
> 9\. Demo hiện tại có thể triển khai thực tế chưa? Vì sao?

**13** **Tài** **liệu** **công** **cụ** **chung**

> Các nguồn dataset đã được đặt trực tiếp trong từng đề tài. Phần này
> chỉ giữ tài liệu công cụ dùng chung:
>
> • W&B Experiment Tracking: <https://docs.wandb.ai/models/track>
>
> • ONNX Runtime Documentation: <https://onnxruntime.ai/docs/>
>
> • Hugging Face Optimum ONNX Runtime:
> [https://huggingface.co/docs/optimum/onnxruntime/usage_guides/expo](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/export)
> [rt](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/export)
>
> • Google Data Cards Playbook:
> <https://sites.research.google/datacardsplaybook/>

**14** **Kết** **luận**

> Bài tập lớn này yêu cầu sinh viên thực hành một quy trình học máy có
> trách nhiệm và có thể tái lập:

Dataset → Data Card → Baseline → Deep Learning Model → W&B Tracking →
Evaluation → ONNX/Optimization → Error Analysis → Demo

Nếu CSC4005 giúp sinh viên hiểu cách xây dựng và huấn luyện mô hình học
sâu, thì CSC4007 giúp sinh viên hiểu bài toán ngôn ngữ, dữ liệu, metric
và lỗi của hệ thống NLP. Một đề tài tốt là đề tài kết nối được cả hai
góc nhìn này: không chỉ có điểm số cao, mà còn có phân tích thuyết phục
và sản phẩm có khả năng mở rộng.

> 25
