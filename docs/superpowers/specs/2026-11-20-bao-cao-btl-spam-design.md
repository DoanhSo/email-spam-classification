# Thiết kế: Báo cáo BTL "Hệ thống phát hiện email spam"

Ngày: 20/11/2026. Trạng thái: bản nháp, chờ nhóm duyệt.

Tài liệu này **thay thế phạm vi trong `KE_HOACH_NHOM.md` và `THIET_KE_KHUNG_DU_AN.md`**
(4 mô hình, ensemble, lịch 29/10–25/11). Theo nhóm, những thông tin đó không còn đúng.
Hai file cũ được giữ lại làm lịch sử và không còn là căn cứ triển khai.

## 0. Thông tin đã chốt với nhóm

- **Nhóm 12**, lớp 65CNTT. Đề tài: "Hệ thống phát hiện email spam/rác". Thành viên:
  1. Nguyễn Quốc Doanh – 2351060429 (nhóm trưởng)
  2. Nguyễn Thái Dương – 2351060435
  3. Lê Hữu Hiệu – 2351060446
  4. Thân Đức Minh – 2351060467
- **Hạn nộp: 30/11/2026.** Nhóm đặt mục tiêu xong bản hoàn chỉnh sớm hơn hạn.
- **Định dạng:** báo cáo Word (.docx). Bìa, mục lục tự động, đề mục đánh số và chú thích hình
  theo mẫu.
- **Weka:** người dùng tự cài `weka-3-8-7-bellsoft-x64-windows.exe` (Windows x64) và tự thao
  tác giao diện, chụp màn hình theo checklist có đánh số hình. Phía code chuẩn bị ARFF, script,
  checklist và chèn ảnh vào báo cáo.

## 1. Mục đích và lý do

Nhóm cần nộp BTL môn Khai phá dữ liệu (GV: TS. Trần Mạnh Tuấn, Bộ môn HTTT, ĐH Thủy Lợi).
Báo cáo phải **giống báo cáo mẫu về hình thức và nội dung**. Mẫu là đề tài "Khai phá dữ liệu
mức độ đánh giá chất lượng ô tô bằng phương pháp phân lớp" (29 trang): làm các bước KPDL
trên Weka có ảnh chụp màn hình, chọn một thuật toán phân lớp, tự viết code thuật toán đó,
đối chiếu với Weka và demo dự đoán.

Báo cáo cũng phải đáp ứng slide "Thực hành – Bài 3. Triển khai bài tập lớn". Slide yêu cầu quy
trình nghiên cứu 7 bước: xác định vấn đề → tổng quan tài liệu → giả thuyết → thiết kế nghiên cứu
→ thu thập dữ liệu → phân tích và kiểm định giả thuyết → lý giải và báo cáo. Bài báo cáo gồm
Mở đầu, Nội dung, Kết luận và Tài liệu tham khảo. Báo cáo mẫu chưa có phần **tổng quan tài
liệu, giả thuyết và kiểm định**. Bài của nhóm bổ sung ba phần này.

Phần tổng quan tài liệu ưu tiên các bài báo đăng ở những venue thuộc nhóm *Databases &
Information Systems* của Google Scholar Metrics (danh sách ở phụ lục A).

## 2. Mục tiêu, phi mục tiêu, phản mục tiêu

**Mục tiêu**
- Báo cáo tiếng Việt, có bìa, mục lục và đề mục đánh số theo đúng khung của mẫu (mục 5).
- Mô hình chính là **Multinomial Naive Bayes (NB)**. Mô hình so sánh là **Decision Tree**:
  J48 trên Weka và cây tự viết trong repo.
- Mỗi mô hình có kết quả trên Weka theo đủ 3 cách kiểm thử như mẫu: Use training set,
  Cross-validation 10-fold và Supplied test set (chia 70/30).
- Kiểm định giả thuyết NB so với DT bằng Weka Experimenter.
- Tự viết NB bằng Python (NumPy) trong repo và đối chiếu số liệu với Weka trên cùng tập 70/30.
- Demo dự đoán email nhập vào, dùng lại giao diện Streamlit đã có.

**Phi mục tiêu**
- Không làm ensemble, không làm LR/SVM hay Random Forest, không dùng deep learning.
- Không đặt trước mục tiêu Accuracy/F1.

**Phản mục tiêu** (vẫn tính là thất bại dù code chạy được)
- Báo cáo lệch khung mẫu: thiếu ảnh Weka, thiếu một trong ba cách kiểm thử, hoặc thiếu phần
  giải thích các thống kê và độ đo.
- Số liệu trong báo cáo không tái lập được, hoặc ảnh chụp Weka không khớp với bảng số.
- Rò rỉ dữ liệu: bộ từ vựng hoặc lựa chọn đặc trưng học trên cả tập test rồi báo cáo như kết
  quả test.
- Trích dẫn bài báo không có thật hoặc chưa kiểm tra được.

## 3. Ràng buộc

- Máy làm việc: Windows 11, đã có Java 21. **Weka chưa được cài.** Cần Weka 3.8.x bản stable,
  do người dùng tự tải từ trang chính thức (waikato.github.io/weka-wiki).
- Repo đã có `src/spam_detector` gồm tiền xử lý, đặc trưng, Decision Tree tự viết, inference,
  `app/streamlit_app.py` và các test. Mọi thay đổi phải tái dùng những phần này, không viết lại.
- Dữ liệu đã có trong repo: `data/raw/enron/spam_ham_dataset.csv` (Kaggle Enron, 5171 dòng, các
  cột `<index>, label, text, label_num`) và `data/raw/spamassassin/completeSpamAssassin.csv`.
- Mẫu chỉ dùng **một** thuật toán. Bài của nhóm dùng NB làm trục chính, DT chỉ xuất hiện ở các
  mục so sánh, giả thuyết và kiểm định để báo cáo vẫn giữ một mạch chính.

## 4. Các phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| **NB chính + DT so sánh (đã chọn)** | Khung giống mẫu, xoay quanh NB. Tái dùng DT đã làm. Có 2 mô hình nên đặt được giả thuyết và kiểm định đúng yêu cầu slide Thực hành. | Thêm khoảng nửa ngày cho J48 và Experimenter. |
| Chỉ NB | Giống mẫu nhất, ít việc nhất. | Giả thuyết và kiểm định yếu. Bỏ phí DT. |
| Chỉ DT (ID3) | Giống mẫu từng chữ, code có sẵn. | Không phải mô hình hợp nhất với spam: DT chỉ đạt accuracy 0,41 trên SpamAssassin. ID3 trên Weka cần gói riêng và thuộc tính nominal. |

Vì sao NB là mô hình chính: dữ liệu văn bản thưa và nhiều chiều, NB học theo tần suất từ nên
nhanh và chính xác. NB là baseline kinh điển của lọc spam (Sahami 1998, Androutsopoulos SIGIR
2000, Metsis 2006). Bộ Enron-Spam được tạo ra chính để đánh giá các biến thể NB. Weka có sẵn
`NaiveBayesMultinomial`.

Về công cụ, nhóm chọn **hướng A: Weka kết hợp Python**. Weka dùng cho tiền xử lý, ảnh chụp và
kết quả như mẫu. Python dùng cho chương "Triển khai thuật toán", tương ứng với phần C++ trong mẫu.

## 5. Khung báo cáo (ánh xạ theo mẫu)

Các mục có dấu ★ là phần thêm so với mẫu, để đáp ứng slide Thực hành.

- **Bìa**: Trường ĐH Thủy Lợi, Khoa CNTT, môn KHAI PHÁ DỮ LIỆU, đề tài, giảng viên, nhóm và
  danh sách thành viên kèm MSSV, "HÀ NỘI, 2026".
- **Mục lục**, **Lời cảm ơn**.
- **I. Mô tả bài toán**
  1. Lý do chọn đề tài: spam gây thiệt hại kinh tế, lừa đảo và phishing.
  2. Tổng quan bài toán: phân loại nhị phân ham/spam.
  3. ★ Tổng quan tài liệu nghiên cứu: 6–10 bài, ưu tiên venue ở phụ lục A. Có bảng tóm tắt
     gồm tác giả, năm, venue, dữ liệu, phương pháp và kết quả.
  4. ★ Giả thuyết nghiên cứu: H0 và H1 (mục 7).
  5. Quy trình thực hiện: 6 bước KDD, tóm tắt theo cách mẫu làm.
  6. Phân tích dữ liệu thô: nguồn, số bản ghi, bảng thuộc tính và ý nghĩa, các lớp.
- **II. Quy trình khai phá dữ liệu**
  1. Tiền xử lý
     - 1.1 Làm sạch: loại thuộc tính dư thừa (cột chỉ số; `label` trùng nghĩa với `label_num`),
       loại bản ghi trùng (`RemoveDuplicates`), xử lý thiếu (kiểm tra email rỗng), xử lý không
       nhất quán (cùng văn bản nhưng khác nhãn).
     - 1.2 Tích hợp dữ liệu: chỉ dùng một nguồn nên bỏ qua, giống mẫu.
     - 1.3 Biến đổi: `StringToWordVector` (chữ thường, stopwords, đếm từ, giới hạn số từ giữ lại).
  2. Phân tích dữ liệu: thống kê mô tả, phân bố lớp, top từ theo lớp, độ dài email theo lớp,
     kèm biểu đồ Visualize của Weka.
  3. Biến đổi và lựa chọn thuộc tính: `AttributeSelection` (InfoGain hoặc ChiSquared + Ranker,
     top-k). Tương ứng với bước "tạo thuộc tính Weight + Discretize" của mẫu.
  4. Phân tách dữ liệu 70/30: `Randomize` (seed cố định) rồi `RemovePercentage`, có ảnh chụp.
- **III. Phương pháp phân lớp và thuật toán Naive Bayes**
  1. Bài toán phân lớp.
  2. Multinomial Naive Bayes: lý thuyết (định lý Bayes, giả định độc lập, công thức multinomial,
     làm trơn Laplace, tính bằng log), quy trình theo bước, kết quả 3 cách kiểm thử kèm ảnh
     chụp, giải thích các thống kê (Correctly Classified, Kappa, MAE, RMSE, RAE, RRSE…) và độ
     đo (TP/FP Rate, Precision, Recall, F-Measure, MCC, ROC, PRC) như mẫu.
  3. ★ So sánh với Decision Tree (J48): lý thuyết ngắn (entropy, gain ratio), kết quả 3 cách
     kiểm thử, cây hoặc luật minh họa.
  4. ★ Kiểm định giả thuyết: thiết lập Experimenter, bảng kết quả, kết luận bác bỏ hay không
     bác bỏ H0.
  5. Nhận xét ưu và nhược điểm, phân tích các trường hợp phân loại sai.
- **IV. Triển khai thuật toán (Python)**
  1. Xây dựng mô hình: trích đoạn code NB tự viết, gồm hàm `fit` và `predict_proba`.
  2. Đối chiếu với Weka trên cùng tập 70/30, bảng so sánh số liệu.
  3. Sử dụng mô hình để dự đoán: ảnh chụp Streamlit với email hợp lệ ra Spam/Ham kèm xác suất,
     và email rỗng hoặc không có từ nào trong từ vựng ra "Không dự đoán được". Trường hợp sau
     tương ứng với "Unpredictable" trong mẫu.
- **KẾT LUẬN**: kết quả đạt được, hạn chế, hướng phát triển.
- **TÀI LIỆU THAM KHẢO**: bài giảng môn học, nguồn dữ liệu, các bài báo đã kiểm tra.

## 6. Quy trình Weka đã chốt

1. Python script chuyển CSV sang ARFF với `text` kiểu string và `class {ham,spam}`, escape dấu
   nháy và xuống dòng. Weka đọc CSV chứa văn bản nhiều dòng rất dễ lỗi, nên cần bước này.
   Có thể ghi vào báo cáo như một bước "tạo tập tin dữ liệu đầu vào".
2. Chụp các bước Preprocess trên Explorer theo mục II.
3. **Chống rò rỉ dữ liệu**: số liệu Supplied test set và Cross-validation phải lấy từ
   `FilteredClassifier`, trong đó `StringToWordVector` cùng lựa chọn thuộc tính nằm bên trong
   bộ lọc của classifier. Tập test 30% không được dùng để học từ vựng. Ảnh Preprocess áp bộ lọc
   lên toàn bộ dữ liệu chỉ để minh họa. Báo cáo nói rõ điều này. Mẫu không làm bước này, nên
   đây là điểm cải tiến có thể nêu trong báo cáo.
4. Lưu tập 70/30 ra ARFF và xuất danh sách ID để Python dùng đúng cùng tập khi đối chiếu.

### 6.1 Nguồn dữ liệu chuẩn (quyết định của người viết thiết kế)

- Báo cáo mẫu nằm ở `C:\Users\mrdoanh\Downloads\Bai tap lon.pdf`.
- File ARFF xuất từ CSV Enron **gốc** (5171 dòng) và giữ thêm thuộc tính `id`. Bản ghi trùng
  được loại bằng `RemoveDuplicates` của Weka. Tập 70/30 do Weka tạo bằng `Randomize -S 42` và
  `RemovePercentage`. Hai bước này **định nghĩa** dữ liệu của báo cáo. Weka lưu `train.arff` và
  `test.arff`, Python đọc lại theo `id`. Trong mọi classifier, thuộc tính `id` bị loại bằng bộ
  lọc `Remove`.
- Kết quả 60/20/20 cũ trong repo (`reports/enron-v1`, `artifacts/tree-enron-v1`) **không dùng**
  trong báo cáo. Cây tự viết được huấn luyện lại trên tập 70/30 mới.
- Weka Experimenter chỉ chạy trên `train.arff` (10×10 CV), để tập test 30% không bị dùng lại.

### 6.2 Mức khớp giữa NB Python và Weka (quyết định của người viết thiết kế)

- Đối chiếu chính xác: dùng Weka *batch filtering* (`StringToWordVector` cùng lựa chọn thuộc
  tính, học trên train rồi áp sang test) để xuất ra hai file vector train và test. NB Python học
  trên đúng các vector này và phải cho ra **cùng dự đoán và cùng ma trận nhầm lẫn** với
  `NaiveBayesMultinomial` của Weka, chênh lệch xác suất chỉ ở mức làm tròn. Phần này tương ứng
  với câu "output C++ cho ra kết quả đúng" trong mẫu.
- Demo Streamlit: NB Python dùng pipeline đặc trưng có sẵn trong repo, học trên `train.arff` và
  được đóng gói thành release giống `tree`. Báo cáo ghi rõ demo dùng bộ đặc trưng của repo nên
  số liệu có thể lệch nhẹ so với Weka.

### 6.3 Cây tự viết và SpamAssassin (quyết định của người viết thiết kế)

- Chương IV có một bảng so sánh: NB Python, cây Python, NB Weka và J48 trên cùng tập test 30%.
- SpamAssassin được dùng làm **kiểm thử trên nguồn khác**, đặt thành một mục ngắn ở cuối
  chương III (III.6). Mục này có bảng cho NB và cây, mô hình học trên Enron, kèm nhận xét về lệch
  miền dữ liệu. Ở mục 1.2, báo cáo vẫn ghi "một nguồn huấn luyện nên bỏ qua tích hợp".

## 7. Giả thuyết và kiểm định

- H0: trên Enron, F-Measure lớp spam của NB và J48 không khác nhau.
- H1: F-Measure của NB lớn hơn J48. Đây là giả thuyết một phía, thực hiện bằng t-test hai phía
  của Weka rồi đọc theo dấu của hiệu số.
- Thiết kế: Weka Experimenter, 10 lần × 10-fold CV, *Paired T-Tester (corrected)*, α = 0,05.
  Đo thêm Percent_correct và Area_under_ROC.
- Báo cáo bảng kết quả và ký hiệu v/* của Weka. Trình bày kết luận theo khuôn của slide
  Thực hành, trang 16–17.

## 8. Phần việc trong repo

- Thêm `src/spam_detector/models/naive_bayes.py`: Multinomial NB tự viết, cùng interface
  `fit`/`predict`/`predict_proba` với `models/base.py`, làm trơn Laplace và tính bằng log.
  Viết test theo kiểu `tests/test_decision_tree.py`.
- Thêm script xuất ARFF và nạp lại tập 70/30 theo ID.
- Thêm script tính số liệu Python cho bảng đối chiếu ở mục IV.2.
- Đăng ký NB vào `configs/web.yaml` và cho NB làm mô hình mặc định của Streamlit. DT vẫn chọn được.
- Lưu ảnh chụp Weka vào `reports/weka/` theo thứ tự hình. Lưu bản thảo báo cáo vào `reports/bao-cao/`.

## 9. Phần để người triển khai tự quyết

- Tham số cụ thể của `StringToWordVector` (số từ giữ lại, stemmer, tokenizer) và giá trị k
  khi chọn thuộc tính. Chọn theo kết quả CV trên tập train, ghi lý do vào báo cáo.
- Tham số J48 (`-C`, `-M`).
- Cách định dạng biểu đồ, bảng và đánh số hình.
- Câu chữ của Lời cảm ơn, Lý do chọn đề tài và Kết luận, viết theo giọng của mẫu.

## Phụ lục A. Venue ưu tiên cho tổng quan tài liệu

Theo Google Scholar Metrics, nhóm Databases & Information Systems, h5 giai đoạn 2020–2024,
lấy ngày 20/11/2026: IEEE TKDE; ACM SIGIR; WWW; Information Processing & Management; ACM CIKM;
VLDB; ACM WSDM; Journal of Big Data; ICDE; ACM SIGMOD; ACM TOIS; ICWSM; ACM RecSys; IEEE BigData;
ACM TIST; Information Systems; World Wide Web; IEEE Trans. Big Data; Knowledge and Information
Systems; ACM TOIT.

Một số bài nền tảng có thể dùng làm điểm xuất phát. **Phải tra DOI hoặc trang chính thức
trước khi trích dẫn.** Bài nào không xác minh được thì bỏ.
- Androutsopoulos và cộng sự, "An experimental comparison of naive Bayesian and keyword-based
  anti-spam filtering with personal e-mail messages", SIGIR 2000 (venue nằm trong danh sách).
- Metsis, Androutsopoulos, Paliouras, "Spam filtering with Naive Bayes – which Naive Bayes?",
  CEAS 2006. Đây là bài gốc của bộ Enron-Spam.
- Sahami và cộng sự, "A Bayesian approach to filtering junk e-mail", AAAI Workshop 1998.
- Các bài khảo sát hoặc bài gần đây về lọc spam và phishing trên TKDE, IP&M, CIKM, WWW, KAIS:
  tìm bằng Google Scholar với từ khóa "spam filtering", "phishing email detection" và lọc
  theo tên venue.
