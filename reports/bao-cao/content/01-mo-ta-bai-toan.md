# I. Mô tả bài toán

## 1. Lý do chọn đề tài

Thư điện tử vẫn là kênh trao đổi chính trong công việc và học tập. Tuy nhiên, một phần lớn lưu lượng thư là thư rác: quảng cáo không mong muốn, mời chào đầu tư, bán thuốc, và nguy hiểm hơn là thư lừa đảo nhằm đánh cắp tài khoản hay tiền bạc. Thư rác khiến người dùng mất thời gian lọc thư, làm đầy hộp thư và có thể khiến thư quan trọng bị bỏ sót. Lọc thủ công hoặc lọc theo danh sách từ khóa cố định không theo kịp tốc độ thay đổi nội dung của thư rác. Vì vậy cần một hệ thống tự học từ dữ liệu thư đã gán nhãn để phân biệt thư rác với thư hợp lệ. Đây là một bài toán phân lớp điển hình của khai phá dữ liệu. Nhóm em chọn đề tài "Hệ thống phát hiện email spam/rác" để vận dụng quy trình khai phá dữ liệu đã học vào một vấn đề thực tế và gần gũi.

## 2. Tổng quan bài toán

Tập dữ liệu gồm các email tiếng Anh, mỗi email có nội dung văn bản (tiêu đề và thân thư) và nhãn cho biết đó là thư hợp lệ hay thư rác. Bài toán là xây dựng mô hình phân lớp nhị phân xác định một email mới thuộc lớp nào:

- **ham**: thư hợp lệ (không phải thư rác);
- **spam**: thư rác.

Lớp spam được coi là lớp dương tính (positive) khi tính các độ đo Precision, Recall và F-Measure.

## 3. Tổng quan tài liệu nghiên cứu

Để xác định hướng tiếp cận, nhóm khảo sát các công trình về lọc thư rác. Ưu tiên các bài đăng ở những hội nghị và tạp chí thuộc nhóm *Databases & Information Systems* của Google Scholar Metrics [8], gồm ACM SIGIR, International World Wide Web Conference (WWW) và ACM Transactions on Information Systems (TOIS).

Lọc thư rác bằng học máy bắt đầu từ cuối thập niên 1990. Sahami và cộng sự [2] là những người đầu tiên đưa bộ phân lớp Bayes vào bài toán lọc thư rác. Họ chỉ ra rằng các luật viết tay ít hiệu quả. Các tác giả còn bổ sung những đặc trưng riêng của thư rác, ngoài nội dung văn bản, vào mô hình xác suất. Androutsopoulos và cộng sự [3] (SIGIR 2000) so sánh Naive Bayes với bộ lọc dựa trên từ khóa trên thư cá nhân. Kết quả cho thấy Naive Bayes vượt trội so với cách lọc theo từ khóa. Nhóm tác giả cũng nhấn mạnh rằng nếu tính chi phí của việc chặn nhầm thư hợp lệ, bộ lọc cần có thêm cơ chế an toàn.

Metsis và cộng sự [4] nhận thấy có nhiều biến thể Naive Bayes nhưng tài liệu lọc thư rác thường không phân biệt. Họ đánh giá năm biến thể trên sáu tập dữ liệu mới, gọi là **Enron-Spam**, ghép thư hợp lệ của nhân viên Enron với thư rác. Đây là nguồn gốc của bộ dữ liệu Enron mà nhóm sử dụng. Về phương pháp đánh giá, Cormack và Lynam [5] (ACM TOIS 2007) thử nghiệm 11 biến thể của 6 bộ lọc mã nguồn mở trên chuỗi thư theo thời gian, và báo cáo khoảng tin cậy cho mọi độ đo. Sculley và Wachman [6] (SIGIR 2007) đề xuất *Relaxed Online SVM*, giảm đáng kể chi phí tính toán của SVM trong lọc thư rác trực tuyến. Ở bài toán gần với thư rác là thư lừa đảo (phishing), Fette và cộng sự [7] (WWW 2007) xây dựng bộ lọc PILFER từ các đặc trưng của email, cho kết quả tốt hơn SpamAssassin.

Bảng 1: Tóm tắt các công trình liên quan

| TT | Tác giả (năm) | Nơi công bố | Dữ liệu | Phương pháp | Kết quả chính |
|---|------|-----|------|--------|--------|
| 1 | Sahami và cs. (1998) [2] | AAAI Workshop | Thư người dùng | Bộ phân lớp Bayes, đặc trưng văn bản và đặc trưng riêng của thư rác | Bayes hiệu quả hơn luật viết tay |
| 2 | Androutsopoulos và cs. (2000) [3] | ACM SIGIR ★ | Thư cá nhân (Ling-Spam) | Naive Bayes so với lọc theo từ khóa | Naive Bayes tốt hơn; cần tính chi phí chặn nhầm |
| 3 | Metsis và cs. (2006) [4] | CEAS | Enron-Spam (6 tập) | 5 biến thể Naive Bayes | Công bố bộ Enron-Spam; so sánh các biến thể NB |
| 4 | Cormack, Lynam (2007) [5] | ACM TOIS ★ | Chuỗi thư theo thời gian | 11 biến thể của 6 bộ lọc mã nguồn mở | Phương pháp đánh giá trực tuyến có khoảng tin cậy |
| 5 | Sculley, Wachman (2007) [6] | ACM SIGIR ★ | Dữ liệu TREC Spam | Relaxed Online SVM | Giảm chi phí tính toán, giữ hiệu quả lọc |
| 6 | Fette và cs. (2007) [7] | WWW ★ | Thư lừa đảo và thư hợp lệ | Đặc trưng email và học máy (PILFER) | Tốt hơn SpamAssassin trên thư lừa đảo |

*★: hội nghị hoặc tạp chí thuộc danh sách Databases & Information Systems của Google Scholar Metrics [8].*

**Nhận xét và hướng tiếp cận của nhóm.** Các công trình cho thấy Naive Bayes là phương pháp nền tảng và phổ biến nhất trong lọc thư rác. Phương pháp này đơn giản, huấn luyện nhanh, phù hợp với dữ liệu văn bản nhiều chiều và thưa. Ngoài ra, bộ dữ liệu Enron-Spam được xây dựng chính để đánh giá Naive Bayes. Vì vậy nhóm chọn **Multinomial Naive Bayes** làm mô hình chính. Cây quyết định (J48) được dùng làm mô hình đối chứng, vì cây cho ra luật dễ giải thích, nhưng thường kém hơn trên dữ liệu văn bản thưa. Giả thuyết ở mục 4 kiểm tra nhận định này bằng thực nghiệm.

## 4. Giả thuyết nghiên cứu

Từ tổng quan tài liệu, nhóm đặt câu hỏi nghiên cứu: *với dữ liệu email tiếng Anh, Multinomial Naive Bayes có phát hiện thư rác tốt hơn cây quyết định hay không?* Câu hỏi này được chuyển thành cặp giả thuyết có thể kiểm định:

- **H0** (giả thuyết không): trên dữ liệu Enron, F-Measure của lớp spam do Multinomial Naive Bayes và cây quyết định J48 đạt được **không khác nhau**.
- **H1** (giả thuyết đối): F-Measure của lớp spam do Multinomial Naive Bayes đạt được **lớn hơn** cây quyết định J48.

Thiết kế kiểm định: dùng Weka Experimenter chạy kiểm chứng chéo 10 lần × 10 fold trên tập huấn luyện, so sánh hai mô hình bằng kiểm định t ghép cặp có hiệu chỉnh (*Paired T-Tester, corrected*) với mức ý nghĩa α = 0,05. Ngoài F-Measure, nhóm theo dõi thêm độ chính xác (Percent correct) và diện tích dưới đường cong ROC (AUC). Kết quả được trình bày ở mục III.4.

## 5. Quy trình thực hiện

Theo bài giảng của môn học [1], quy trình khai phá dữ liệu gồm 6 bước:

- Bước 1: Tạo tập tin dữ liệu đầu vào
- Bước 2: Tiền xử lý, làm sạch tập dữ liệu
- Bước 3: Chọn tác vụ khai phá dữ liệu (phân lớp)
- Bước 4: Khai phá dữ liệu: tìm kiếm tri thức
- Bước 5: Đánh giá mẫu tìm được
- Bước 6: Biểu diễn tri thức

Ở bài này, nhóm em tóm tắt các bước thành những mục sau:

i. Thu thập dữ liệu và chuyển sang định dạng ARFF của Weka
ii. Tiền xử lý dữ liệu (làm sạch, loại trùng lặp)
iii. Biến đổi dữ liệu (vector hóa văn bản, lựa chọn thuộc tính)
iv. Phân tách dữ liệu huấn luyện/kiểm thử (70/30)
v. Xây dựng và đánh giá mô hình phân lớp trên công cụ Weka [11], kiểm định giả thuyết
vi. Tự cài đặt thuật toán bằng Python, đối chiếu với Weka và xây dựng chương trình dự đoán

## 6. Phân tích dữ liệu thô

- Nguồn dữ liệu thô: bộ **Enron-Spam** [4], bản phân phối "Spam Mails Dataset" trên Kaggle [9], tệp `spam_ham_dataset.csv`.
- Hiểu dữ liệu: mỗi bản ghi là một email tiếng Anh (bắt đầu bằng dòng "Subject:") cùng nhãn ham/spam.
- Dữ liệu gồm **5171 bản ghi** với 4 cột, trong đó **3672 thư ham (71,0%)** và **1499 thư spam (29,0%)**. Dữ liệu lệch lớp ở mức vừa phải.
- Để Weka đọc được văn bản nhiều dòng, nhóm viết một chương trình nhỏ chuyển tệp CSV sang định dạng ARFF (`enron_raw.arff`). Nội dung email được khai báo kiểu `STRING`, nhãn được khai báo kiểu `{ham, spam}`.

![Hình 1: Dữ liệu thô](../weka/figures/hinh-01.png)

➢ Hiểu các thuộc tính:

| STT | Thuộc tính | Kiểu | Ý nghĩa thuộc tính |
|---|-----|----|-----------|
| 1 | row_index | Numeric | Số thứ tự của bản ghi trong tệp gốc |
| 2 | label_num | Numeric | Nhãn dạng số (0 = ham, 1 = spam) |
| 3 | text | String | Nội dung email (tiêu đề và thân thư) |
| 4 | class | Nominal | Phân loại email: ham, spam |

Các lớp phân loại:

- ham: thư hợp lệ
- spam: thư rác

