# KẾT LUẬN

Khai phá dữ liệu với phương pháp phân lớp là một trong những phương pháp phổ biến, mang tính ứng dụng và hiệu quả thực tiễn cao, đặc biệt trong các bài toán xử lý văn bản như lọc thư rác, phân loại tin tức hay phát hiện lừa đảo.

Sau khi hoàn thành đề tài "Hệ thống phát hiện email spam/rác", nhóm em đã đạt được các kết quả sau:

- Tìm hiểu tổng quan về khám phá tri thức, khai phá dữ liệu, bài toán phân lớp; khảo sát các công trình về lọc thư rác trên các hội nghị, tạp chí thuộc nhóm Databases & Information Systems; nắm vững thuật toán Multinomial Naive Bayes và cây quyết định C4.5 (J48).
- Thực hiện đầy đủ quy trình khai phá dữ liệu trên bộ Enron-Spam bằng Weka: loại thuộc tính dư thừa, loại 178 bản ghi trùng, vector hóa văn bản, chọn 500 thuộc tính theo Information Gain, phân tách dữ liệu 70/30 và đặt các bộ lọc trong FilteredClassifier để tránh rò rỉ dữ liệu.
- Xây dựng mô hình Naive Bayes đạt độ chính xác 93,06% và F-Measure lớp spam 0,881 trên tập kiểm thử; so sánh với cây quyết định J48 (91,86%; 0,867) theo ba phương pháp kiểm thử.
- Đặt và kiểm định giả thuyết bằng Weka Experimenter (10 lần × 10 fold, kiểm định t ghép cặp có hiệu chỉnh): Naive Bayes có kết quả trung bình cao hơn J48 ở mọi độ đo; khác biệt về Precision có ý nghĩa thống kê, còn khác biệt về F-Measure chưa đủ ý nghĩa ở mức 0,05 nên chưa bác bỏ được H0.
- Tự cài đặt thuật toán Naive Bayes bằng Python; kết quả trùng khớp với Weka trên toàn bộ 1498 email kiểm thử. Xây dựng chương trình demo cho phép nhập email và giải thích kết quả bằng các từ đóng góp.

Hạn chế của đề tài:

- Mô hình chỉ học trên thư tiếng Anh của một công ty (Enron) nên giảm mạnh độ chính xác khi áp dụng cho nguồn thư khác (khoảng 44% trên SpamAssassin).
- Biểu diễn túi từ bỏ qua thứ tự từ, ngữ cảnh và các thông tin khác của email như người gửi, tiêu đề kỹ thuật, tệp đính kèm, đường dẫn.

Hướng phát triển:

- Huấn luyện trên dữ liệu đa dạng từ nhiều nguồn, bổ sung thư tiếng Việt.
- Thử nghiệm thêm các mô hình như Logistic Regression, SVM, Random Forest và kết hợp nhiều mô hình; bổ sung đặc trưng từ tiêu đề kỹ thuật và đường dẫn trong thư.

Trong quá trình hoàn thành đề tài, dù nhóm em đã cố gắng tìm hiểu và thực hiện bài toán nhưng vẫn không thể tránh những thiếu sót. Do vậy, rất mong nhận được sự đóng góp ý kiến của thầy để chúng em rút kinh nghiệm, cải thiện kỹ năng và tích lũy kiến thức trong môn học.
