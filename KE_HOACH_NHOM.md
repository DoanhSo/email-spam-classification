# Kế hoạch bài tập lớn môn Khai phá dữ liệu

**Đề tài:** Hệ thống phát hiện email spam tiếng Anh bằng các mô hình tự cài đặt và kết hợp bỏ phiếu.

**Nhóm trưởng:** Nguyễn Quốc Doanh - phụ trách Decision Tree.

**Thiết kế khung dự án:** Xem [THIET_KE_KHUNG_DU_AN.md](THIET_KE_KHUNG_DU_AN.md) để triển khai cấu trúc thư mục, hợp đồng module, thực nghiệm, lưu mô hình, kiểm thử và mốc xây khung trong 5 ngày đầu.

**Căn cứ:** Tài liệu KhaiPhaDL.pdf do nhóm cung cấp, 13 trang. Đây là bản kế hoạch điều chỉnh từ tài liệu, không phải xác nhận yêu cầu chính thức của giảng viên. Phân công dưới đây ưu tiên thông tin mới của nhóm. Cập nhật theo thông tin mới: còn khoảng 1 tháng. Kế hoạch gồm 4 tuần làm việc và vài ngày dự phòng. Tạm tính bắt đầu ngày 29/10/2026, hoàn thiện bản nộp nội bộ ngày 25/11/2026; ngày 26-28/11 dành cho dự phòng. Đây là lịch đề xuất, không phải hạn nộp chính thức đã được xác nhận.

## 1. Phạm vi và sản phẩm cuối cùng

- Phân loại email tiếng Anh thành ham (0) và spam (1).
- Bốn mô hình chính: Multinomial Naive Bayes, Logistic Regression, Linear SVM và Decision Tree.
- Tự viết thuật toán học, dự đoán và bỏ phiếu ensemble. Theo tài liệu, được dùng thư viện để xử lý dữ liệu, chia tập, tính metrics và làm giao diện; mô hình sklearn chỉ xuất hiện trong phần thực nghiệm đối chiếu riêng.
- Dùng Enron-Spam làm dữ liệu chính; giữ SpamAssassin làm tập kiểm tra trên nguồn khác. Không trộn SpamAssassin vào tập huấn luyện của thí nghiệm kiểm tra chéo nguồn.
- Hoàn thành mã nguồn, hướng dẫn chạy, kết quả thực nghiệm, demo, báo cáo và slide.
- Random Forest chuyển sang hướng phát triển, không nằm trong kế hoạch triển khai 1 tháng. Không đưa SMS, tiếng Việt hoặc deep learning vào phạm vi triển khai chính.
- Không đặt trước Accuracy/F1 phải đạt bao nhiêu khi chưa có kết quả. Tiêu chí hoàn thành là cài đặt đúng, thực nghiệm tái lập được, có phân tích ưu/nhược điểm và lỗi.

## 2. Phân công chính thức đề xuất

### Nguyễn Quốc Doanh - nhóm trưởng, Decision Tree

**Mô hình:** Tự cài đặt cây phân loại nhị phân theo Gini; ưu tiên đặc trưng có/không xuất hiện từ. Hỗ trợ giới hạn độ sâu, số mẫu tối thiểu, xuất luật và tỉ lệ spam tại lá.

**Việc chung:** Chốt phạm vi, cấu trúc dự án, giao diện các mô hình, mốc bàn giao; rà soát tích hợp; kiểm tra tính thống nhất của báo cáo và chịu trách nhiệm đóng gói bản nộp.

**Bàn giao:** `decision_tree.py`, kiểm thử cây, kết quả thử tham số, hình cây hoặc luật quyết định, phần lý thuyết/thực nghiệm Decision Tree và README tổng hợp.

**Giới hạn tải việc:** Không mặc định nhận toàn bộ dữ liệu, demo, ensemble hoặc viết báo cáo thay thành viên. Mỗi người chịu trách nhiệm hoàn thành phần mình trước khi Doanh duyệt.

### Lê Hữu Hiệu - Naive Bayes, dữ liệu và tiền xử lý

**Mô hình:** Multinomial Naive Bayes; làm trơn Laplace; tính toán bằng log để tránh tràn số về 0.

**Việc chung:** Thu thập và ghi nguồn dữ liệu; thống nhất nhãn; xử lý trùng lặp; EDA; làm sạch email; trích đặc trưng thống kê từ văn bản gốc. Chủ trì giao diện demo đơn giản sau khi pipeline hoạt động, các thành viên cung cấp hàm dự đoán.

**Bàn giao:** `naive_bayes.py`, `preprocessing.py`, notebook EDA, dữ liệu theo schema chung, `app/app.py`, kiểm thử và phần báo cáo tương ứng.

### Thân Đức Minh - Logistic Regression, biểu diễn đặc trưng

**Mô hình:** Logistic Regression; sigmoid, cross-entropy, gradient descent, điều chuẩn L2.

**Việc chung:** BoW, TF-IDF, Chi-square; thống nhất danh sách từ và thứ tự cột; cung cấp nhánh đặc trưng nhị phân cho Decision Tree; lưu bộ biến đổi đã học để dùng lại khi demo. Trong lịch 1 tháng, ưu tiên dùng thư viện cho TF-IDF/Chi-square theo phạm vi cho phép của tài liệu; chỉ tự viết thêm khi bốn mô hình và thực nghiệm chính đã hoàn thành.

**Bàn giao:** `logistic_regression.py`, `features.py`, `feature_selection.py`, đường loss, top từ có trọng số, kiểm thử và phần báo cáo tương ứng.

### Nguyễn Thái Dương - Linear SVM, đánh giá và ensemble

**Mô hình:** Linear SVM; hinge loss, điều chuẩn, subgradient descent; ánh xạ nhãn 0/1 sang -1/+1 trong nội bộ.

**Việc chung:** Chia tập và lưu ID mẫu; thiết lập fold chung; metrics, biểu đồ, thời gian chạy; viết ensemble bỏ phiếu theo nhãn dự đoán; tổng hợp bảng kết quả và hỗ trợ Hiệu tích hợp demo.

**Bàn giao:** `svm.py`, `evaluation.py`, `ensemble.py`, cấu hình chia tập, notebook thực nghiệm, bảng kết quả, kiểm thử và phần báo cáo tương ứng.

**Cả nhóm:** Mỗi người tự viết lý thuyết, mô tả cài đặt, phân tích lỗi, kiểm thử và slide cho mô hình của mình. Mỗi người có ít nhất một lần chạy bản tích hợp trên máy của mình.

## 3. Quy ước phải chốt trong 2 ngày đầu

1. **Dữ liệu:** `sample_id`, `text`, `label`, `source`; nhãn 0 là ham, 1 là spam. Loại các cột đánh số dòng hoặc thông tin lộ nhãn khỏi đặc trưng.
2. **Giao diện mô hình:** `fit(X, y)` và `predict(X)` là bắt buộc. Dự đoán trả mảng nhãn 0/1 theo đúng thứ tự mẫu đầu vào.
3. **Điểm dự đoán:** NB/LR/Tree có thể trả `predict_proba(X)` với hai cột `[P(ham), P(spam)]`. SVM dùng `decision_function(X)` cho ROC/PR; không coi sigmoid trực tiếp của margin là xác suất đã hiệu chuẩn. Ensemble dùng nhãn nên không phụ thuộc xác suất SVM.
4. **Đầu vào:** NB/LR/SVM dùng nhánh đặc trưng phù hợp; Tree bản đầu dùng ma trận nhị phân 0/1, chọn 500 hoặc 1.000 từ bằng Chi-square. Lưu tên đặc trưng đúng thứ tự với từng mô hình.
5. **Chia dữ liệu:** Đề xuất Enron train/validation/test = 60/20/20, phân tầng theo nhãn; cố định seed, ví dụ 42. Giữ các bản sao hoặc nhóm email gần trùng ở cùng một tập. Dương quản lý danh sách ID chung.
6. **Tránh rò rỉ dữ liệu:** Vocabulary, IDF, Chi-square, scaler và tham số tiền xử lý có học từ dữ liệu chỉ được fit trên train. Khi cross-validation, fit lại trong từng fold huấn luyện. Validation/test chỉ transform.
7. **Tách mục đích đánh giá:** 5-fold trên phần train để khảo sát độ ổn định; validation để chốt tham số và trọng số ensemble; test giữ kín đến khi cấu hình đã đóng băng. Báo cáo phân biệt kết quả CV, validation và test.
8. **So sánh công bằng:** Dùng cùng tập mẫu, fold và cách đo thời gian. Nếu mỗi mô hình dùng nhánh đặc trưng khác nhau, ghi rõ đang so sánh cả pipeline. Đối chiếu tự viết/sklearn phải dùng cùng biểu diễn đầu vào và cấu hình gần tương đương.
9. **Demo:** Tải mô hình và bộ biến đổi đã lưu, không huấn luyện lại mỗi lần nhập email. Không fit lại từ điển trên nội dung người dùng nhập.
10. **Quản lý phiên bản:** Mỗi phần việc có người phụ trách, hạn nội bộ, đầu ra và người duyệt; lưu cấu hình chạy cùng kết quả. Không đưa dữ liệu lớn hoặc môi trường ảo vào Git một cách mặc định.

## 4. Lộ trình 4 tuần và thời gian dự phòng

Các ngày dưới đây giả định bắt đầu ngay 29/10/2026. Khi có hạn chính thức, dịch lịch tương ứng nhưng giữ ít nhất 2 ngày dự phòng. Mỗi người bắt đầu viết lý thuyết và mô tả cài đặt ngay tuần 1; tuần cuối bổ sung kết quả và biên tập.

### Tuần 1: 29/10-04/11 - Dữ liệu chung và mô hình bản đầu

- **Doanh:** Ngày 1 chốt repo, phân công, giao diện; ngày 2-3 học và tính tay Gini, viết Node/tìm phép chia; ngày 4-7 dựng cây và dự đoán trên dữ liệu nhỏ. Dùng dữ liệu nhị phân tự tạo để không phải chờ pipeline hoàn chỉnh.
- **Hiệu:** Bàn giao mẫu dữ liệu và hàm làm sạch trong 2 ngày đầu; hoàn thiện nhãn, trùng lặp, EDA cơ bản; viết Naive Bayes bản đầu.
- **Minh:** Bàn giao TF-IDF và nhánh nhị phân K=500 chậm nhất ngày 4; ghi rõ bộ biến đổi chỉ fit trên train; viết Logistic Regression bản đầu.
- **Dương:** Chốt ID chia tập trong 2 ngày đầu sau khi có kiểm tra nhãn/trùng lặp; viết metrics và khung chạy; viết SVM bản đầu.
- **Mốc nghiệm thu 04/11:** Có dữ liệu và tập chia cố định, bộ biến đổi chung; bốn mô hình có fit/predict chạy trên ví dụ nhỏ. Khung báo cáo đã có người phụ trách từng phần.

### Tuần 2: 05/11-11/11 - Hoàn thiện bốn mô hình và tích hợp

- **Doanh:** Hoàn thiện điều kiện dừng, max_depth/min_samples_leaf, xác suất tại lá, xuất luật; kiểm tra cây trên dữ liệu thật; nghiệm thu luồng tích hợp.
- **Hiệu:** Hoàn thiện Naive Bayes, kiểm tra log/Laplace/từ ngoài từ điển; tạo giao diện nhập email đơn giản sau khi mô hình ổn định.
- **Minh:** Kiểm tra gradient, loss và L2 của LR; hoàn thiện chọn đặc trưng và lưu/tải bộ biến đổi.
- **Dương:** Kiểm tra cập nhật SVM, thống nhất đầu ra bốn mô hình; viết bỏ phiếu và lưu kết quả validation để tính trọng số.
- **Cả nhóm:** Chạy kiểm thử, đối chiếu sklearn ban đầu trên cùng dữ liệu; ghi nhận lỗi và thời gian chạy. Mỗi người gửi phần lý thuyết/cài đặt bản nháp.
- **Mốc nghiệm thu 11/11:** Bốn mô hình chạy trên cùng tập dữ liệu, có metrics validation; có bản tích hợp đầu tiên nhập email và trả dự đoán. Không thêm Random Forest hoặc giao diện phức tạp.

### Tuần 3: 12/11-18/11 - Thực nghiệm, ensemble và kết quả cuối

- **Doanh:** Thử K=500/1.000, độ sâu và số mẫu tối thiểu theo từng bước; lưu luật từ cây thật, phân tích FP/FN và đóng góp E1/E5/E7.
- **Hiệu:** Chuẩn bị SpamAssassin để kiểm tra nguồn khác, rà soát trùng với dữ liệu học; thử tiền xử lý E3 ở phạm vi nhỏ và phân tích lỗi; hoàn thiện demo.
- **Minh:** Chạy E2 và E5 theo cấu hình giới hạn; bảo đảm bảng so sánh ghi đúng biểu diễn đặc trưng và bộ biến đổi.
- **Dương:** Tổng hợp E1, đối chiếu E7, tính trọng số validation cho E6; phối hợp Hiệu chạy E4; tổng hợp bảng, biểu đồ, thời gian.
- **Mức ưu tiên đề xuất:** Hoàn thành E1/E6/E7 trước; sau đó E4; E2/E3/E5 chạy cấu hình đại diện, không quét toàn bộ tổ hợp. Đây là thứ tự tổ chức công việc, không tự bỏ yêu cầu bắt buộc của giảng viên. Ghi rõ trường hợp nào chưa thực hiện đủ theo tài liệu.
- **Khóa cấu hình trước đánh giá cuối:** Chỉ dùng train/CV/validation để điều chỉnh. Sau khi chốt mô hình, trọng số và ngưỡng, mới mở test và đánh giá SpamAssassin độc lập.
- **Mốc nghiệm thu 18/11:** Có bảng kết quả cuối, phân tích lỗi, ensemble và demo chạy được. Dừng thêm tính năng để chuyển sang hồ sơ nộp.

### Tuần 4: 19/11-25/11 - Báo cáo, kiểm tra tái lập và diễn tập

- **19-21/11:** Mỗi người nộp phần báo cáo, hình, bảng kết quả và slide của mình. Doanh tổng hợp kiến trúc, phần mở đầu và kết luận; Dương thống nhất số liệu thực nghiệm.
- **22-23/11:** Hiệu hoàn thiện hướng dẫn demo/video dự phòng; Minh kiểm tra bộ biến đổi đã lưu; mỗi người chạy lại bản tích hợp trên máy mình theo README. Sửa lỗi làm ảnh hưởng kết quả thì ghi nhận và chạy lại phần đánh giá liên quan.
- **24/11:** Cả nhóm diễn tập thuyết trình, chia thời lượng theo yêu cầu môn học; luyện trả lời lý do chọn mô hình, chống overfitting, rò rỉ dữ liệu và cách đánh giá.
- **25/11:** Doanh nghiệm thu, đóng gói mã nguồn, báo cáo, slide, dữ liệu/cách tải, cấu hình và hướng dẫn chạy. Mỗi thành viên xác nhận phần đóng góp và bản cuối của mình.
- **Mốc nghiệm thu 25/11:** Có bản nộp hoàn chỉnh; không còn hạng mục bắt buộc đang viết dở.

### 26-28/11 - Dự phòng

Chỉ sửa lỗi, xử lý vấn đề môi trường, kiểm tra file và nộp theo hạn chính thức. Nếu hạn thực tế sớm hơn, kéo mốc khóa bản nộp về trước hạn ít nhất 2 ngày; không mặc định 29/11 là hạn do giảng viên đặt.

### Cách xử lý khi chậm mốc

- Hết ngày 4 chưa có đặc trưng: dùng bộ biến đổi thư viện được phép; mô hình vẫn tự viết.
- Hết tuần 1 mô hình chưa chạy trên ví dụ nhỏ: thành viên cùng cặp rà soát hỗ trợ ngay; chưa thử tham số hoặc thêm tính năng.
- Hết tuần 2 chưa tích hợp: ưu tiên ghép fit/predict và demo một màn hình; tạm dừng mở rộng tiền xử lý và thử nghiệm nhiều cấu hình.
- Hết tuần 3 còn lỗi mô hình: giữ cấu hình đơn giản đã kiểm tra, sửa tính đúng; không dùng mô hình sklearn thay vào sản phẩm tự viết. Thông báo rõ phần thực nghiệm chưa xong và đối chiếu yêu cầu môn học trước khi giảm phạm vi.

## 5. Kế hoạch riêng cho Nguyễn Quốc Doanh: Decision Tree

### A. Kiến thức cần nắm

- Nút gốc, nút trong, nút lá; cách đi từ đặc trưng đến dự đoán.
- Gini: `1 - p_ham² - p_spam²`.
- Độ giảm Gini phải có trọng số theo số mẫu của hai nhánh.
- Với đặc trưng nhị phân, ngưỡng 0,5 tách từ không xuất hiện và có xuất hiện; không cần quét mọi ngưỡng như đặc trưng liên tục.
- Cây sâu có thể học thuộc dữ liệu; giới hạn độ sâu và số mẫu ở lá giúp kiểm soát độ phức tạp.
- Tỉ lệ spam tại lá là một ước lượng xác suất theo mẫu ở lá, không tự động bảo đảm được hiệu chuẩn tốt.

### B. Các phần tự cài đặt theo thứ tự

1. Cấu trúc `Node`: chỉ số đặc trưng, ngưỡng, con trái/phải, số mẫu, số mẫu mỗi lớp, nhãn lá.
2. Hàm Gini và hàm tính độ giảm Gini có trọng số.
3. Hàm tìm đặc trưng chia tốt nhất; bỏ qua cột hằng và phép chia sinh nhánh rỗng.
4. Dựng cây đệ quy; chỉ chấp nhận phép chia có cải thiện dương và thỏa kích thước lá.
5. Điều kiện dừng: một lớp, đạt `max_depth`, không đủ `min_samples_split`, không tìm được phép chia hợp lệ.
6. Dự đoán một mẫu rồi mở rộng sang nhiều mẫu, giữ đúng thứ tự đầu vào.
7. Xác suất theo tỉ lệ lớp tại lá; quy tắc hòa nhãn cố định, đề xuất ưu tiên ham khi đúng hòa.
8. Xuất luật dạng “có từ A và không có từ B → spam”, kèm số mẫu hỗ trợ và tỉ lệ lớp của lá.
9. Lưu/tải mô hình cùng danh sách từ và cấu hình tiền xử lý.
10. Đối chiếu kết quả và phân tích chênh lệch với cây sklearn trong thí nghiệm riêng.

### C. Phạm vi thử tham số đề xuất

- Bắt đầu bằng K=500, `max_depth=5`, `min_samples_leaf=5` để kiểm tra luồng chạy.
- Sau khi đúng, khảo sát K=500/1.000 và `max_depth=5/10/15` trước; sau đó thử `min_samples_leaf=1/5/10` trên cấu hình tốt.
- Ghi Precision spam, Recall spam, F1 spam, FP, FN, độ sâu, số nút và thời gian huấn luyện.
- E5 bổ sung K=3.000 và toàn bộ từ nếu tài nguyên cho phép; ghi rõ giới hạn nếu không hoàn thành. Không cần nhân toàn bộ tổ hợp thành một lưới thử quá lớn.
- Không chọn cấu hình chỉ vì điểm train cao; dùng validation, xem cả Precision và Recall.

### D. Tiêu chí nghiệm thu phần Decision Tree

- Kết quả Gini/phép chia đúng trên ví dụ tính tay.
- Không lỗi ở nút chỉ có một lớp, cột hằng, nhánh không thể tách hoặc email không có từ trong từ điển.
- Dự đoán chỉ trả nhãn 0/1; xác suất hợp lệ và mỗi hàng có tổng bằng 1.
- Không vượt giới hạn độ sâu; không tạo lá ít mẫu hơn cấu hình khi thực hiện chia.
- Có kết quả trên tập chia chung; có luật minh họa từ cây thực sự đã học, không tự đặt tỉ lệ dự đoán.
- Có phân tích một trường hợp chặn nhầm thư thật và một trường hợp bỏ sót spam nếu chúng xuất hiện; nếu không có thì ghi đúng kết quả.
- Báo cáo giải thích vì sao dùng đặc trưng nhị phân, vì sao chọn Gini, cách hạn chế overfitting và giới hạn của cây trên văn bản nhiều chiều.

### E. Nhịp làm việc cá nhân đề xuất

Với lịch 1 tháng, dự trù ban đầu khoảng 12-15 giờ/người/tuần nếu đã biết Python cơ bản; đây là ước lượng lập kế hoạch, cần điều chỉnh sau tuần đầu. Riêng Doanh dành khoảng 7-9 giờ cho cây/kiểm thử, 3 giờ điều phối/tích hợp và 2-3 giờ viết báo cáo/ôn phản biện. Tuần 4 chuyển phần lớn thời gian sang hoàn thiện hồ sơ và diễn tập. Nếu cả nhóm chỉ có dưới 8 giờ/người/tuần, ưu tiên giảm mở rộng và số cấu hình thử, đồng thời chốt sớm phạm vi thực nghiệm với giảng viên.

## 6. Thực nghiệm và cách kết luận

- **E1:** So sánh bốn mô hình với fold chung; báo cáo trung bình và độ lệch chuẩn của Precision/Recall/F1, thời gian học/dự đoán. Số liệu test cuối trình bày riêng.
- **E2:** So sánh BoW, TF-IDF, TF-IDF cộng đặc trưng thống kê ở các mô hình hỗ trợ. Tree nhị phân là một cấu hình riêng; không gọi dữ liệu đã nhị phân hóa là thử nghiệm TF-IDF liên tục. Chỉ bổ sung nhánh cây liên tục khi có thời gian cài đặt và kiểm tra.
- **E3:** Thử tác động của stemming/lemmatization và stopword; thay đổi từng yếu tố để diễn giải được kết quả.
- **E4:** Huấn luyện bằng Enron, dùng nguyên bộ biến đổi để đánh giá SpamAssassin; báo cáo thay đổi phân phối và từ ngoài từ điển. Không dùng nhãn SpamAssassin để chọn cấu hình rồi gọi đó là kiểm tra độc lập.
- **E5:** Khảo sát K=500/1.000/3.000/toàn bộ từ; ghi cấu hình và giới hạn tài nguyên.
- **E6:** So sánh ensemble với từng mô hình. Trọng số bằng F1 validation của mỗi mô hình chia tổng F1; nếu tổng bằng 0 thì dùng trọng số đều. Quy định điểm đúng 0,5 theo công thức gốc là spam. Bỏ phiếu có trọng số vẫn có thể hòa; cần quy tắc này để kết quả xác định.
- **E7:** So sánh tự viết và sklearn với cùng đầu vào; giải thích khác biệt về tối ưu, dừng hoặc cách xử lý hòa. Điểm gần nhau hỗ trợ đánh giá nhưng không thay thế kiểm thử tính đúng.

Ưu tiên theo dõi Precision của lớp spam và số thư thật bị chặn nhầm, đồng thời báo cáo Recall/F1 để tránh mô hình bỏ sót quá nhiều spam. Không chỉ báo cáo Accuracy. ROC/PR phải dùng điểm số liên tục thích hợp của mô hình, không dùng nhãn cứng thay cho điểm.

## 7. Cơ chế quản lý nhóm

- Đầu tuần: họp 20-30 phút, mỗi người chốt đầu ra và hạn nội bộ.
- Mỗi ngày: cập nhật ngắn ba dòng “đã xong / đang vướng / cần hỗ trợ”; giữa tuần kiểm tra tích hợp 15 phút. Vướng quá một ngày thì báo để điều chỉnh sớm.
- Cuối tuần: chạy sản phẩm để nghiệm thu, không chỉ báo phần trăm hoàn thành. Kiểm tra cả mã, dữ liệu vào/ra, kết quả và hướng dẫn.
- Ghép cặp rà soát: Doanh ↔ Minh; Hiệu ↔ Dương. Người viết vẫn chịu trách nhiệm sửa lỗi của mình.
- Một đầu việc chỉ coi là xong khi chạy được, có hướng dẫn và đầu ra kiểm chứng; phần mô hình phải có kiểm thử phù hợp.
- Ghi quyết định thay đổi phạm vi để tránh một người tự đổi schema, tập chia hoặc tên nhãn làm hỏng tích hợp.

## 8. Rủi ro và phương án xử lý

- **Cây chạy chậm:** Bắt đầu bằng đặc trưng nhị phân và K=500; giới hạn độ sâu; đo thời gian trên tập nhỏ trước khi mở rộng.
- **Nhóm trưởng quá tải:** Giữ Doanh ở Decision Tree, kiến trúc và nghiệm thu. Hiệu chủ trì demo, Dương chủ trì ensemble, các thành viên tự viết báo cáo.
- **Điểm số cao do rò rỉ:** Rà soát ID chia tập, trùng lặp, từ điển/IDF/Chi-square và cách chọn tham số trước khi tin kết quả.
- **Mô hình dùng sai bộ từ điển khi demo:** Lưu một gói pipeline và cấu hình cho mỗi mô hình; thử lại trên cùng email trước/sau khi tải.
- **Không đủ thời gian:** Hoãn Random Forest, SMS, lemmatization nâng cao và giao diện phức tạp trước; giữ bốn mô hình chính, đánh giá chung, ensemble cơ bản và báo cáo. Nếu giảm E1-E7 phải ghi rõ phạm vi chưa làm và đối chiếu yêu cầu giảng viên.

## 9. Việc nên giao ngay trong buổi họp đầu tiên

- [ ] Doanh chốt hạn nộp thực tế, lịch họp và mục tiêu Decision Tree.
- [ ] Hiệu bàn giao mẫu dữ liệu có nhãn và nguồn, kèm số liệu kiểm tra nhãn/trùng lặp sơ bộ.
- [ ] Minh bàn giao thiết kế hai nhánh đặc trưng: TF-IDF và nhị phân.
- [ ] Dương bàn giao kế hoạch chia tập, bộ metrics và mẫu lưu kết quả.
- [ ] Cả nhóm thống nhất `fit/predict`, quy ước nhãn, thứ tự mẫu và danh sách file phải bàn giao.
- [ ] Mỗi người chọn một ví dụ nhỏ để giải thích và kiểm tra mô hình bằng tay.

## 10. Hồ sơ nộp

Mã nguồn bốn mô hình tự viết; mã ensemble; pipeline tiền xử lý/đặc trưng; kiểm thử; hướng dẫn tải dữ liệu; cấu hình chia tập và tham số; bảng kết quả thật; demo và video dự phòng; báo cáo; slide; phân công/đóng góp. Báo cáo theo mạch: bài toán → lý thuyết → dữ liệu/EDA → thiết kế/cài đặt → thực nghiệm → phân tích lỗi → kết luận và hướng phát triển.

**Nguồn đối chiếu:** KhaiPhaDL.pdf, trang 1-3 (phạm vi, dữ liệu), 7-9 (cây, ensemble, phân công, lộ trình), 10 (E1-E7), 11-13 (cấu trúc, báo cáo, phản biện). Tên thành viên và vai trò nhóm trưởng/Decision Tree lấy từ yêu cầu của Nguyễn Quốc Doanh. Việc chia lại demo/ensemble, các mốc nghiệm thu và quy trình chống rò rỉ là đề xuất bổ sung của bản kế hoạch này.
