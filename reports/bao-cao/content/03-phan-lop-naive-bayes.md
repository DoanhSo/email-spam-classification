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

TODO

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

TODO

## 4. Kiểm định giả thuyết

TODO

## 5. Nhận xét và phân tích lỗi

TODO

## 6. Kiểm thử trên nguồn dữ liệu khác (SpamAssassin)

TODO
