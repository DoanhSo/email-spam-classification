# Đối chiếu báo cáo nhóm 12 với báo cáo mẫu

Báo cáo mẫu: `Bai tap lon.pdf` (đề tài chất lượng ô tô, 29 trang). Báo cáo nhóm: `BaoCao_Nhom12_SpamEmail.docx` / `.pdf` .

| Mục trong báo cáo mẫu | Mục tương ứng của nhóm | Có |
|---|---|---|
| Bìa (trường, khoa, môn, đề tài, giảng viên, nhóm, thành viên + MSSV, Hà Nội năm) | Bìa: Nhóm 12, 65CNTT, 4 thành viên + MSSV, HÀ NỘI 2026 | ✔ |
| Mục lục | MỤC LỤC (trường TOC tự động của Word) | ✔ |
| Lời cảm ơn | LỜI CẢM ƠN | ✔ |
| I.1 Lý do chọn đề tài | I.1 | ✔ |
| I.2 Tổng quan bài toán | I.2 | ✔ |
| I.3 Quy trình thực hiện (6 bước) | I.5 | ✔ |
| I.4 Phân tích dữ liệu thô (nguồn, số bản ghi, bảng thuộc tính, các lớp) + Hình dữ liệu thô | I.6 + Hình 1 | ✔ |
| II.1.1.1 Loại thuộc tính dư thừa (Weka Remove) | II.1.1.1 + Hình 2–4 | ✔ |
| II.1.1.2 Loại nhiễu | II.1.1.2 Loại dữ liệu trùng lặp (RemoveDuplicates) + Hình 5 | ✔ |
| II.1.1.3 Xử lý dữ liệu thiếu | II.1.1.3 + Hình 6 | ✔ |
| II.1.1.4 Dữ liệu không nhất quán | II.1.1.4 | ✔ |
| II.1.2 Tích hợp dữ liệu (một nguồn nên bỏ qua) | II.1.2 | ✔ |
| II.1.3 Biến đổi dữ liệu | II.1.3 StringToWordVector + Hình 7–8 | ✔ |
| II.2 Phân tích dữ liệu (thống kê mô tả, mối tương quan, kết luận) | II.2 + Hình 9 | ✔ |
| II.3 Biến đổi dữ liệu (tạo thuộc tính Weight, Discretize) | II.3 Chọn thuộc tính InfoGain + Hình 10–11 | ✔ |
| II.4 Phân tách dữ liệu 70/30 (Randomize, RemovePercentage) | II.4 + Hình 12–14 | ✔ |
| III.1 Bài toán phân lớp | III.1 | ✔ |
| III.2.1 Lý thuyết + công thức | III.2.1 Multinomial NB (Bayes, Laplace, log) | ✔ |
| III.2.2 Quy trình thực hiện (các bước) | III.2.2 | ✔ |
| III.2.3 Kết quả: Use training set / Cross-validation / Supplied test set | III.2.3 + Hình 15–18 + Bảng 2 | ✔ |
| Cây quyết định (hình cây) | III.3 J48 + Hình 19–22 (Weka) + Hình 23–25 (toàn bộ cây 129 nút) | ✔ |
| Giải thích các thống kê (Correctly…, Kappa, MAE, RMSE, RAE, RRSE, Total) | III.2.3 | ✔ |
| Giải thích các độ đo (TP/FP Rate, Precision, Recall, F, MCC, ROC, PRC) | III.2.3 | ✔ |
| Nhận xét ưu/nhược điểm của thuật toán | III.5 | ✔ |
| IV.1 Triển khai thuật toán: xây dựng mô hình (code) | IV.1 (Python, trích code fit/predict_proba) | ✔ |
| "So sánh với Weka thì output C++ cho kết quả đúng" | IV.2: trùng 1498/1498 dự đoán, Bảng 6 | ✔ |
| IV.2 Sử dụng mô hình để dự đoán + bảng nhãn/output | IV.3 + bảng kết quả + Hình 30–31 | ✔ |
| Trường hợp không dự đoán được ("Unpredictable") | IV.3 + Hình 32 ("Không dự đoán được") | ✔ |
| Kết luận | KẾT LUẬN (kết quả, hạn chế, hướng phát triển) | ✔ |
| Tài liệu tham khảo | TÀI LIỆU THAM KHẢO [1]–[11], đều được trích dẫn trong bài | ✔ |

## Phần bổ sung theo slide "Thực hành – Bài 3" (không có trong mẫu)

| Bước quy trình nghiên cứu | Mục | Có |
|---|---|---|
| Tổng quan tài liệu (venue Databases & Information Systems) | I.3 + Bảng 1 (4 bài thuộc danh sách: SIGIR 2000, SIGIR 2007, ACM TOIS 2007, WWW 2007; DOI đã xác minh) | ✔ |
| Hình thành giả thuyết | I.4 (H0/H1) | ✔ |
| Phân tích và kiểm định giả thuyết | III.4 (Experimenter 10×10 CV, corrected paired t-test, Hình 26–27, Bảng 4) | ✔ |
| Kiểm thử trên nguồn dữ liệu khác | III.6 (SpamAssassin, Bảng 5, Hình 29) | ✔ |

## Rà soát các phản mục tiêu của thiết kế

| Phản mục tiêu | Kết quả |
|---|---|
| Lệch khung mẫu, thiếu ảnh Weka, thiếu một trong ba cách kiểm thử, thiếu giải thích thống kê/độ đo | Không: đủ 24 ảnh Weka và hình toàn bộ cây J48, đủ 3 cách kiểm thử cho cả NB và J48, đủ phần giải thích |
| Số liệu không tái lập được hoặc ảnh không khớp bảng | Số liệu lấy từ `reports/weka/results/*.txt`, `results.json`, `nb_python_vs_weka.json`; đã đối chiếu từng bảng với file nguồn |
| Rò rỉ dữ liệu (từ vựng/chọn thuộc tính học trên test) | Không: FilteredClassifier cho mọi kết quả; Experimenter chỉ chạy trên train.arff; batch filtering học trên train |
| Trích dẫn bài báo không có thật / chưa xác minh | Không: mọi bài có DOI hoặc trang chính thức (`references.csv`, cột verified_on) |

## Ghi chú trung thực

- Kết quả kiểm định (H1 hai phía): không bác bỏ H0 về F-Measure ở α = 0,05 (p = 0,072); Precision khác biệt có ý nghĩa (p = 0,037, chưa hiệu chỉnh đa so sánh). Recall và ROC Area trong Bảng 4 được đánh dấu "(tính lại)" vì không chạy trong tab Analyse.
- Weka Experimenter GUI báo "Access is denied" khi ghi tệp kết quả, nên thí nghiệm được chạy bằng `weka.jar` (dòng lệnh) với cùng cấu hình; phần phân tích (Hình 24) làm trên tab Analyse của GUI.
- Bước batch filtering (xuất vector cho đối chiếu Python) chạy bằng `weka.jar` dòng lệnh, cùng tham số với GUI.
