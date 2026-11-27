# III. Phương pháp phân lớp và thuật toán Naive Bayes

## 1. Bài toán phân lớp

- Phân lớp dữ liệu là kĩ thuật dựa trên tập huấn luyện đã biết nhãn lớp để xây dựng mô hình, sau đó dùng mô hình đó gán nhãn cho dữ liệu mới.
- Phân lớp là một hình thức học có giám sát: dữ liệu mới được phân lớp dựa trên tập huấn luyện.
- Quá trình phân lớp dữ liệu gồm hai bước:
  - Xây dựng mô hình: học từ tập huấn luyện (70% dữ liệu).
  - Sử dụng mô hình: đánh giá trên tập kiểm thử (30% dữ liệu), sau đó phân lớp email mới.

## 2. Thuật toán Multinomial Naive Bayes

### 2.1. Lý thuyết

- Naive Bayes là bộ phân lớp xác suất dựa trên định lý Bayes, với giả định "ngây thơ" rằng các thuộc tính (ở đây là các từ) độc lập có điều kiện với nhau khi đã biết lớp. Giả định này hiếm khi đúng hoàn toàn với văn bản, nhưng trên thực tế Naive Bayes vẫn phân lớp văn bản rất hiệu quả [2], [3], [4].
- Định lý Bayes, với email $d$ và lớp $c \in \{ham, spam\}$:

$$P(c \mid d) = \frac{P(c)\,P(d \mid c)}{P(d)}$$

- Mô hình **Multinomial** biểu diễn email bằng số lần xuất hiện $f_w$ của mỗi từ $w$ trong từ vựng $V$. Bỏ mẫu số $P(d)$ (giống nhau với mọi lớp), lớp được chọn là:

$$\hat{c} = \arg\max_{c} \Big[ \log P(c) + \sum_{w \in V} f_w \log P(w \mid c) \Big]$$

- Xác suất tiên nghiệm của lớp (có làm trơn Laplace), với $N_c$ là số email thuộc lớp $c$ và $N$ là tổng số email:

$$P(c) = \frac{N_c + 1}{N + 2}$$

- Xác suất của từ trong lớp, làm trơn Laplace ($\alpha = 1$) để tránh xác suất bằng 0 khi một từ chưa từng xuất hiện trong lớp, với $n_{wc}$ là tổng số lần từ $w$ xuất hiện trong các email thuộc lớp $c$:

$$P(w \mid c) = \frac{n_{wc} + \alpha}{\sum_{w' \in V} n_{w'c} + \alpha\,|V|}$$

- Tính toán trong không gian logarit giúp tránh tràn số dưới (underflow) khi nhân rất nhiều xác suất nhỏ. Xác suất hậu nghiệm được chuẩn hóa lại bằng kỹ thuật log-sum-exp.
- Ưu điểm: đơn giản, huấn luyện rất nhanh (chỉ cần đếm), phù hợp dữ liệu văn bản nhiều chiều và thưa, cho ra xác suất và các từ đóng góp nhiều nhất nên dễ giải thích.
- Hạn chế: giả định độc lập giữa các từ không đúng thực tế; xác suất ước lượng thường bị đẩy về gần 0 hoặc 1 (chưa hiệu chuẩn); kết quả nhạy với cách chọn từ vựng.

### 2.2. Quy trình thực hiện

Bước 1: Biểu diễn mỗi email thành vector đếm từ (StringToWordVector) và chọn ra các từ có Information Gain cao nhất (chỉ học trên tập huấn luyện).

Bước 2: Đếm số email của mỗi lớp, tính xác suất tiên nghiệm $P(c)$.

Bước 3: Đếm tổng số lần xuất hiện của từng từ trong mỗi lớp, tính $P(w \mid c)$ có làm trơn Laplace.

Bước 4: Với email mới, tính điểm $\log P(c) + \sum f_w \log P(w \mid c)$ cho từng lớp.

Bước 5: Chuẩn hóa điểm thành xác suất hậu nghiệm, gán email vào lớp có xác suất lớn nhất.

Bước 6: Đánh giá mô hình bằng ba phương pháp kiểm thử: Use training set, Cross-validation 10 fold và Supplied test set.

### 2.3. Kết quả thu được

➢ Cấu hình trên Weka: tab **Classify → Choose → meta → FilteredClassifier**, trong đó:

- `filter` = **MultiFilter** gồm hai bộ lọc theo thứ tự: StringToWordVector (tham số như Hình 7) rồi AttributeSelection (InfoGain, Ranker, 500 thuộc tính, như Hình 10);
- `classifier` = **bayes → NaiveBayesMultinomial**.

![Hình 15: Cấu hình FilteredClassifier với NaiveBayesMultinomial](../weka/figures/hinh-15.png)

➢ Test options với **Use training set** (đánh giá trên chính 3495 email huấn luyện):

![Hình 16: Kết quả Naive Bayes khi sử dụng Use training set](../weka/figures/hinh-16.png)

➢ Test options với **Cross-validation** (10 fold trên tập huấn luyện):

![Hình 17: Kết quả Naive Bayes khi sử dụng Cross-validation](../weka/figures/hinh-17.png)

➢ Test options với **Supplied test set** (tập kiểm thử 1498 email chưa từng dùng để huấn luyện):

![Hình 18: Kết quả Naive Bayes khi sử dụng Supplied test set](../weka/figures/hinh-18.png)

Bảng 2: Tổng hợp kết quả Naive Bayes theo ba cách kiểm thử

| Thống kê | Use training set | Cross-validation (10) | Supplied test set |
|---|---|---|---|
| Số mẫu | 3495 | 3495 | 1498 |
| Correctly Classified | 3310 (94,71%) | 3294 (94,25%) | 1394 (93,06%) |
| Incorrectly Classified | 185 (5,29%) | 201 (5,75%) | 104 (6,94%) |
| Kappa statistic | 0,8745 | 0,8636 | 0,8320 |
| Mean absolute error | 0,0537 | 0,0578 | 0,0674 |
| Root mean squared error | 0,2087 | 0,2176 | 0,2326 |
| Relative absolute error | 12,90% | 13,88% | 16,32% |
| Root relative squared error | 45,73% | 47,69% | 51,44% |
| Precision (spam) | 0,893 | 0,886 | 0,865 |
| Recall (spam) | 0,933 | 0,924 | 0,897 |
| F-Measure (spam) | 0,912 | 0,905 | 0,881 |
| MCC | 0,875 | 0,864 | 0,832 |
| ROC Area | 0,979 | 0,977 | 0,974 |
| Ma trận nhầm lẫn [TN FP; FN TP] | [2346 116; 69 964] | [2339 123; 78 955] | [1009 60; 44 385] |

➢ Nhận xét:

- Kết quả giảm dần từ Use training set → Cross-validation → Supplied test set, đúng như kỳ vọng: đánh giá trên chính dữ liệu huấn luyện luôn lạc quan nhất, còn tập kiểm thử chưa từng thấy phản ánh sát nhất hiệu quả thực tế. Mức chênh lệch nhỏ (94,71% so với 93,06%) cho thấy mô hình ít bị quá khớp.
- Trên tập kiểm thử, mô hình phân loại đúng **93,06%** email. Trong 429 thư spam, mô hình phát hiện được 385 thư (Recall 0,897); trong 1069 thư hợp lệ, có 60 thư bị chặn nhầm (FP Rate 0,056).

➢ Giải thích các thống kê:

- **Correctly Classified Instances**: số lượng mẫu được mô hình phân loại chính xác.
- **Incorrectly Classified Instances**: số lượng mẫu bị mô hình phân loại sai.
- **Kappa statistic** (thống kê Kappa): chỉ số đo mức độ đồng thuận giữa nhãn dự đoán và nhãn thực tế, đã hiệu chỉnh để loại trừ sự trùng hợp ngẫu nhiên.
- **Mean absolute error** (sai số tuyệt đối trung bình): trung bình độ lệch tuyệt đối giữa xác suất dự đoán của mô hình và giá trị thực tế của mỗi mẫu.
- **Root mean squared error** (căn bậc hai sai số bình phương trung bình): căn bậc hai của trung bình bình phương độ lệch giữa xác suất dự đoán và giá trị thực tế.
- **Relative absolute error** (sai số tuyệt đối tương đối): sai số tuyệt đối của mô hình so với sai số của bộ dự đoán đơn giản luôn dự đoán theo tỉ lệ lớp, tính theo phần trăm.
- **Root relative squared error** (căn bậc hai sai số bình phương tương đối): tương tự như trên nhưng tính trên sai số bình phương.
- **Total Number of Instances**: tổng số mẫu được đánh giá.

➢ Giải thích các độ đo (tính cho từng lớp, lớp dương tính là spam):

- **TP Rate**: tỉ lệ mẫu dương tính được phân loại đúng trên tổng số mẫu dương tính (bằng Recall).
- **FP Rate**: tỉ lệ mẫu âm tính bị phân loại nhầm thành dương tính trên tổng số mẫu âm tính. Với bài toán spam, FP Rate của lớp spam cho biết tỉ lệ thư hợp lệ bị chặn nhầm, là lỗi người dùng khó chấp nhận nhất.
- **Precision**: trong các mẫu được dự đoán là dương tính, tỉ lệ mẫu thực sự dương tính.

$$Precision = \frac{TP}{TP + FP}$$

- **Recall**: trong các mẫu thực sự dương tính, tỉ lệ mẫu được dự đoán đúng.

$$Recall = \frac{TP}{TP + FN}$$

- **F-Measure**: trung bình điều hòa của Precision và Recall. F-Measure càng lớn thì phân loại càng tốt.

$$F = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

- **MCC** (Matthews Correlation Coefficient): hệ số tương quan giữa nhãn dự đoán và nhãn thực tế. MCC = 1 tương ứng phân loại hoàn hảo, MCC = 0 tương đương đoán ngẫu nhiên, MCC = −1 là hoàn toàn ngược.
- **ROC Area**: diện tích dưới đường cong ROC, thể hiện khả năng xếp hạng mẫu dương tính cao hơn mẫu âm tính. Giá trị càng gần 1 càng tốt, 0,5 tương đương đoán ngẫu nhiên.
- **PRC Area**: diện tích dưới đường cong Precision–Recall, phù hợp khi dữ liệu lệch lớp.
- **Confusion Matrix** (ma trận nhầm lẫn): bảng đếm số mẫu theo cặp (lớp thực tế, lớp dự đoán), gồm TN, FP, FN, TP.

## 3. So sánh với cây quyết định J48

### 3.1. Lý thuyết

- J48 là cài đặt trong Weka của thuật toán C4.5 (Quinlan), phát triển từ ID3. Cây được xây dựng từ trên xuống. Tại mỗi nút, thuật toán chọn thuộc tính và ngưỡng chia tốt nhất, chia dữ liệu thành các tập con, rồi lặp lại cho đến khi nút thuần nhất hoặc quá ít mẫu.
- Độ đo dùng để chọn thuộc tính là tỉ lệ độ lợi thông tin (Gain Ratio), giúp giảm thiên lệch của Information Gain về các thuộc tính nhiều giá trị:

$$Entropy(D) = -\sum_{i=1}^{m} P_i \log_2 P_i$$

$$Gain(A) = Entropy(D) - \sum_{j=1}^{v} \frac{|D_j|}{|D|} Entropy(D_j)$$

$$SplitInfo_A(D) = -\sum_{j=1}^{v} \frac{|D_j|}{|D|} \log_2 \frac{|D_j|}{|D|}, \qquad GainRatio(A) = \frac{Gain(A)}{SplitInfo_A(D)}$$

- Với thuộc tính số (số lần xuất hiện của từ), C4.5 tìm ngưỡng $t$ để chia thành hai nhánh $\le t$ và $> t$.
- Sau khi xây cây, J48 cắt tỉa (pruning) với hệ số tin cậy `confidenceFactor = 0.25` và yêu cầu tối thiểu `minNumObj = 2` mẫu ở mỗi lá để giảm quá khớp.
- Ưu điểm: cho ra luật dạng "nếu... thì..." dễ hiểu. Hạn chế: với dữ liệu văn bản hàng trăm chiều và thưa, mỗi nút chỉ xét một từ nên cây dễ sâu, dễ quá khớp và thường kém hơn Naive Bayes.

### 3.2. Kết quả thu được

Giữ nguyên bộ lọc như mục 2.3, chỉ đổi `classifier` sang **trees → J48** (`confidenceFactor = 0.25`, `minNumObj = 2`) và chạy lại ba cách kiểm thử.

![Hình 19: Kết quả J48 khi sử dụng Use training set](../weka/figures/hinh-19.png)

![Hình 20: Kết quả J48 khi sử dụng Cross-validation](../weka/figures/hinh-20.png)

![Hình 21: Kết quả J48 khi sử dụng Supplied test set](../weka/figures/hinh-21.png)

➢ Cây quyết định thu được có **65 lá** và **129 nút**. Phần gốc của cây cho thấy các luật dễ hiểu: nếu email chứa từ `2004` thì là spam (86 mẫu), nếu không thì xét tiếp `prescription`, `meds`, `sex`, `dealer`…, mỗi nhánh "> 0" đều dẫn đến lá spam.

![Hình 22: Phần gốc của cây quyết định J48](../weka/figures/hinh-22.png)

Bảng 3: So sánh Naive Bayes và J48 trên Weka

| Thống kê | NB – Training | J48 – Training | NB – CV 10 | J48 – CV 10 | NB – Test | J48 – Test |
|---|---|---|---|---|---|---|
| Accuracy | 94,71% | 95,54% | 94,25% | 93,05% | **93,06%** | 91,86% |
| Kappa | 0,875 | 0,895 | 0,864 | 0,835 | **0,832** | 0,809 |
| Precision (spam) | 0,893 | 0,898 | 0,886 | 0,864 | **0,865** | 0,815 |
| Recall (spam) | 0,933 | 0,957 | 0,924 | 0,907 | 0,897 | **0,925** |
| F-Measure (spam) | 0,912 | 0,927 | 0,905 | 0,885 | **0,881** | 0,867 |
| ROC Area | 0,979 | 0,991 | 0,977 | 0,970 | **0,974** | 0,968 |
| Thư hợp lệ bị chặn nhầm (FP) | 116 | 112 | 123 | 147 | **60** | 90 |

➢ Nhận xét:

- Trên chính tập huấn luyện, J48 cao hơn Naive Bayes (95,54% so với 94,71%), nhưng khi chuyển sang Cross-validation và tập kiểm thử thì J48 giảm mạnh hơn và thấp hơn Naive Bayes. Đây là dấu hiệu J48 quá khớp với dữ liệu huấn luyện.
- Trên tập kiểm thử, J48 bắt được nhiều spam hơn một chút (Recall 0,925 so với 0,897) nhưng chặn nhầm nhiều thư hợp lệ hơn hẳn (90 so với 60 thư). Với bộ lọc thư, chặn nhầm thư hợp lệ là lỗi nghiêm trọng hơn, nên Naive Bayes phù hợp hơn.

## 4. Kiểm định giả thuyết

➢ Thiết kế thí nghiệm trên **Weka Experimenter**:

- Dữ liệu: chỉ dùng tập huấn luyện `train.arff` (3495 email), không dùng tập kiểm thử, để tập kiểm thử không bị "dùng lại" cho việc so sánh mô hình.
- Phương pháp: kiểm chứng chéo 10 fold, lặp lại 10 lần với các cách chia khác nhau, tổng cộng 100 lần đánh giá cho mỗi mô hình.
- Hai mô hình: FilteredClassifier + NaiveBayesMultinomial và FilteredClassifier + J48, cùng một bộ lọc.
- Các độ đo Precision, Recall, F-Measure được tính cho lớp **spam** (`classForIRStatistics` của ClassifierSplitEvaluator đặt về lớp spam).
- Kiểm định: **Paired T-Tester (corrected)**, mức ý nghĩa **α = 0,05** (hai phía). Đây là kiểm định t ghép cặp có hiệu chỉnh phương sai (Nadeau và Bengio) cho kết quả kiểm chứng chéo lặp lại, vì các fold dùng chung dữ liệu huấn luyện nên không hoàn toàn độc lập:

$$t = \frac{\bar{d}}{\sqrt{\left(\frac{1}{k} + \frac{n_2}{n_1}\right)\hat{\sigma}^2}}$$

trong đó $\bar{d}$ và $\hat{\sigma}^2$ là trung bình và phương sai của hiệu số giữa hai mô hình trên $k = 100$ lần đánh giá, $n_2/n_1 = 1/9$ là tỉ lệ kích thước tập kiểm thử trên tập huấn luyện của mỗi fold.

![Hình 23: Thiết lập thí nghiệm trên Weka Experimenter](../weka/figures/hinh-23.png)

![Hình 24: Kết quả kiểm định F-Measure của lớp spam (base: Naive Bayes)](../weka/figures/hinh-24.png)

Bảng 4: Kết quả kiểm định (trung bình ± độ lệch chuẩn trên 100 lần đánh giá)

| Độ đo (lớp spam) | Naive Bayes | J48 | Kết quả Weka | Giá trị p (hai phía) |
|---|---|---|---|---|
| F-Measure | 0,905 ± 0,017 | 0,888 ± 0,024 | Khác biệt không có ý nghĩa | 0,072 |
| Accuracy | 94,27% ± 1,09 | 93,16% ± 1,49 | Khác biệt không có ý nghĩa | 0,058 |
| Precision | 0,887 ± 0,026 | 0,863 ± 0,030 | J48 kém hơn có ý nghĩa (*) | 0,037 |
| Recall | 0,925 ± 0,021 | 0,914 ± 0,032 | Khác biệt không có ý nghĩa | 0,426 |
| ROC Area | 0,976 ± 0,007 | 0,968 ± 0,012 | Khác biệt không có ý nghĩa | 0,079 |

*Ghi chú: cột "Kết quả Weka" lấy từ tab Analyse (ký hiệu `*` nghĩa là mô hình so sánh kém hơn mô hình gốc có ý nghĩa thống kê); giá trị p được nhóm tính lại theo đúng công thức trên từ tệp kết quả của Experimenter.*

➢ Kết luận kiểm định:

- Naive Bayes đạt kết quả trung bình **cao hơn** J48 ở tất cả các độ đo.
- Với độ đo chính là F-Measure của lớp spam, hiệu số là +0,018 với p = 0,072 > 0,05. Do đó, ở mức ý nghĩa 0,05 **chưa đủ bằng chứng để bác bỏ H0**: dữ liệu cho thấy xu hướng ủng hộ H1 nhưng chưa khẳng định được Naive Bayes có F-Measure lớn hơn J48 một cách có ý nghĩa thống kê. (Nếu kiểm định một phía theo đúng hướng của H1 thì p ≈ 0,036; nhóm giữ kết luận theo kiểm định hai phía của Weka như thiết kế ban đầu.)
- Riêng **Precision** của lớp spam, Naive Bayes tốt hơn J48 có ý nghĩa thống kê (p = 0,037). Nghĩa là trong các thư bị mô hình gắn nhãn spam, tỉ lệ thư thực sự là spam của Naive Bayes cao hơn, hay nói cách khác Naive Bayes **chặn nhầm thư hợp lệ ít hơn** một cách đáng tin cậy. Điều này phù hợp với kết quả trên tập kiểm thử (60 so với 90 thư bị chặn nhầm).

## 5. Nhận xét và phân tích lỗi

➢ Các từ mà mô hình Naive Bayes coi là đặc trưng nhất cho mỗi lớp (tỉ số $\log P(w \mid spam) - \log P(w \mid ham)$ lớn nhất và nhỏ nhất):

![Hình 25: Các từ đặc trưng nhất của mỗi lớp theo Naive Bayes](results/figures/top_words_nb.png)

➢ Phân tích các trường hợp phân loại sai trên tập kiểm thử (60 thư ham bị nhận nhầm là spam và 44 thư spam bị bỏ sót):

| Loại lỗi | Ví dụ nội dung email (rút gọn) | Nguyên nhân |
|---|---|---|
| Thư spam bị bỏ sót | "Subject: " (thư chỉ có tiêu đề rỗng) | Không có từ nào để mô hình dựa vào, chỉ còn xác suất tiên nghiệm nghiêng về ham |
| Thư spam bị bỏ sót | "Subject: holiday e - cards" | Quá ngắn, các từ không đặc trưng cho spam |
| Thư ham bị chặn nhầm | "Subject: important video announcement … please go to to access the video" | Thư nội bộ nhưng dùng lối viết giống thư quảng cáo |
| Thư ham bị chặn nhầm | "Subject: please note my new email address … yahoo . com" | Chứa địa chỉ email cá nhân, ít từ đặc trưng công việc |

➢ Nhận xét chung:

- Ưu điểm của Naive Bayes: huấn luyện và dự đoán rất nhanh, độ chính xác cao trên dữ liệu cùng nguồn (93,06% trên tập kiểm thử), kết quả dễ giải thích qua các từ đóng góp.
- Hạn chế: mô hình phụ thuộc mạnh vào từ vựng của dữ liệu huấn luyện; thư quá ngắn hoặc không chứa từ đặc trưng dễ bị phân loại sai; xác suất dự đoán thường bị đẩy về gần 0 hoặc 1 nên không nên hiểu là "độ tin cậy" thực sự.

## 6. Kiểm thử trên nguồn dữ liệu khác (SpamAssassin)

Để đánh giá khả năng tổng quát hóa, nhóm dùng mô hình Python tự cài đặt (mục IV) đã học trên tập huấn luyện Enron để dự đoán bộ dữ liệu **SpamAssassin** [10], một nguồn email hoàn toàn khác (sau khi làm sạch còn 5267 email: 3912 ham, 1355 spam). Bộ dữ liệu này không được dùng trong huấn luyện.

Bảng 5: Kết quả trên tập kiểm thử Enron và trên SpamAssassin

| Mô hình | Tập dữ liệu | Accuracy | Precision (spam) | Recall (spam) | F-Measure (spam) |
|---|---|---|---|---|---|
| Naive Bayes | Enron – test (1498) | 93,79% | 0,859 | 0,937 | 0,896 |
| Naive Bayes | SpamAssassin (5267) | 44,33% | 0,314 | 0,985 | 0,476 |
| Cây quyết định | Enron – test (1498) | 92,66% | 0,828 | 0,939 | 0,880 |
| Cây quyết định | SpamAssassin (5267) | 44,88% | 0,294 | 0,816 | 0,432 |

![Hình 26: Ma trận nhầm lẫn của Naive Bayes trên SpamAssassin](results/figures/confusion_nb_external.png)

➢ Nhận xét:

- Khi chuyển sang nguồn dữ liệu khác, độ chính xác của cả hai mô hình giảm mạnh từ khoảng 93% xuống khoảng 44%. Nguyên nhân chủ yếu là **lệch miền dữ liệu**: mô hình học được rất nhiều từ đặc thù của thư nội bộ Enron (như `enron`, `hpl`, `gas`, `meter`) làm dấu hiệu của thư hợp lệ. Thư hợp lệ của SpamAssassin không chứa các từ này nên phần lớn bị gắn nhầm là spam (2911 thư ham bị chặn nhầm với Naive Bayes).
- Recall của lớp spam vẫn rất cao (Naive Bayes bắt được 98,5% thư spam), chứng tỏ mô hình vẫn nhận ra đặc điểm của thư rác, nhưng mất khả năng nhận ra thư hợp lệ của tổ chức khác.
- Kết luận: mô hình chỉ nên dùng cho dữ liệu cùng nguồn với dữ liệu huấn luyện. Muốn áp dụng rộng hơn cần huấn luyện trên dữ liệu đa dạng từ nhiều tổ chức (hướng phát triển).

