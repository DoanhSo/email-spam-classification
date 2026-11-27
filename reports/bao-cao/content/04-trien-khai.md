# IV. Triển khai thuật toán (Python)

## 1. Xây dựng mô hình

Nhóm tự cài đặt Multinomial Naive Bayes bằng Python, chỉ dùng thư viện NumPy để tính toán ma trận. Mã nguồn nằm trong tệp `src/spam_detector/models/naive_bayes.py`. Công thức làm trơn giống hệt `NaiveBayesMultinomial` của Weka, để có thể đối chiếu kết quả từng email.

Hàm huấn luyện `fit` đếm số email của mỗi lớp và tổng số lần xuất hiện của mỗi từ trong từng lớp, sau đó tính $\log P(c)$ và $\log P(w \mid c)$:

```python
def fit(self, X, y):
    counts = count_matrix(X)
    labels = binary_labels(y, counts.shape[0], require_both=True)
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    self.class_log_prior_ = np.log((class_counts + 1) / (class_counts.sum() + 2))
    word_counts = np.vstack([counts[labels == c].sum(axis=0) for c in (0, 1)])
    denominators = word_counts.sum(axis=1, keepdims=True) + self.alpha * counts.shape[1]
    self.feature_log_prob_ = np.log((word_counts + self.alpha) / denominators)
    return self
```

Hàm `predict_proba` tính điểm log của từng lớp bằng một phép nhân ma trận, rồi chuẩn hóa bằng log-sum-exp để thu xác suất hậu nghiệm:

```python
def predict_proba(self, X):
    counts = count_matrix(X, self.feature_log_prob_.shape[1])
    scores = counts @ self.feature_log_prob_.T + self.class_log_prior_
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)
```

Ngoài ra, hàm `explain` liệt kê các từ có mặt trong email, xếp theo mức đóng góp $f_w \,(\log P(w \mid spam) - \log P(w \mid ham))$ vào kết luận của mô hình: với email dự đoán là spam, các từ có đóng góp dương lớn nhất đứng đầu; với email dự đoán là ham, các từ có đóng góp âm nhất đứng đầu. Chương trình demo dùng hàm này để giải thích kết quả cho người dùng. Mô hình được kiểm thử tự động bằng các ví dụ tính tay, ví dụ: với tập huấn luyện 3 email và 3 từ, xác suất dự đoán phải đúng bằng $[0{,}6;\ 0{,}4]$ như khi tính tay.

## 2. Đối chiếu kết quả với Weka

Để kiểm chứng chương trình tự viết, nhóm cho mô hình Python và mô hình `NaiveBayesMultinomial` của Weka chạy trên **cùng một dữ liệu đầu vào**: dùng Weka (chế độ batch filtering) áp dụng StringToWordVector và AttributeSelection đã học trên tập huấn luyện để xuất ra các vector 500 từ cho tập huấn luyện và tập kiểm thử. Mô hình Python học trên vector huấn luyện rồi dự đoán 1498 email của tập kiểm thử; kết quả được so sánh với tệp dự đoán mà Weka xuất ra ở Hình 18.

Bảng 6: Đối chiếu Naive Bayes tự viết với Weka trên tập kiểm thử

| Tiêu chí | Python (tự viết) | Weka |
|---|---|---|
| Số email dự đoán | 1498 | 1498 |
| Số email có dự đoán trùng nhau | 1498 / 1498 (100%) | |
| Chênh lệch xác suất lớn nhất | 0,0005 (do Weka làm tròn 3 chữ số) | |
| Ma trận nhầm lẫn [TN FP; FN TP] | [1009 60; 44 385] | [1009 60; 44 385] |
| Accuracy | 93,06% | 93,06% |
| Precision / Recall / F-Measure (spam) | 0,865 / 0,897 / 0,881 | 0,865 / 0,897 / 0,881 |

➢ So sánh với kết quả chạy bằng Weka thì chương trình Python cho kết quả **trùng khớp hoàn toàn trên từng email**, chứng tỏ thuật toán được cài đặt đúng.

Ngoài ra, nhóm huấn luyện thêm mô hình Python trên bộ đặc trưng riêng của chương trình (từ có/không xuất hiện, 1000 từ chọn theo Chi-square, học trên cùng tập huấn luyện 70%) để dùng cho chương trình demo, cùng với cây quyết định tự viết đã có:

Bảng 7: Kết quả các mô hình trên cùng tập kiểm thử 1498 email

| Mô hình | Accuracy | Precision (spam) | Recall (spam) | F-Measure (spam) |
|---|---|---|---|---|
| Naive Bayes – Weka | 93,06% | 0,865 | 0,897 | 0,881 |
| J48 – Weka | 91,86% | 0,815 | 0,925 | 0,867 |
| Naive Bayes – Python (dùng cho demo) | 93,79% | 0,859 | 0,937 | 0,896 |
| Cây quyết định – Python | 92,66% | 0,828 | 0,939 | 0,880 |

Ở cả Weka và Python, Naive Bayes đều cho Accuracy và F-Measure cao hơn cây quyết định.

## 3. Sử dụng mô hình để dự đoán

Nhóm xây dựng chương trình demo bằng thư viện Streamlit (tệp `app/streamlit_app.py`), dùng mô hình Naive Bayes tự viết làm mô hình mặc định. Người dùng:

- Nhập tiêu đề (không bắt buộc) và nội dung email tiếng Anh;
- Bấm **Kiểm tra email**;
- Chương trình hiển thị kết quả **Spam** hoặc **Thư thường (ham)**, xác suất spam và bảng các từ đóng góp nhiều nhất vào kết luận. Nếu nội dung email để trống, chương trình yêu cầu nhập nội dung trước khi kiểm tra.

| Kết quả | Ý nghĩa |
|---|---|
| Spam | Email được dự đoán là thư rác |
| Thư thường (ham) | Email được dự đoán là thư hợp lệ |
| Không dự đoán được | Email không chứa từ nào trong bộ từ vựng của mô hình Naive Bayes |

➢ Thực hành dự đoán với một email quảng cáo thuốc: mô hình dự đoán **Spam** với các từ đóng góp mạnh nhất là `prescription`, `viagra`, `meds`, `cialis`.

![Hình 27: Kết quả dự đoán một email spam](figures/demo-spam.png)

➢ Với một email công việc nội bộ: mô hình dự đoán **Thư thường (ham)**; các từ nghiêng mạnh nhất về thư hợp lệ được liệt kê với đóng góp âm.

![Hình 28: Kết quả dự đoán một email hợp lệ](figures/demo-ham.png)

➢ Trường hợp dữ liệu không nằm trong phạm vi dự đoán (email chỉ gồm các từ không có trong bộ từ vựng): kết quả trả về **"Không dự đoán được"**, tương tự kết quả "Unpredictable" trong chương trình mẫu.

![Hình 29: Kết quả khi nhập email không có từ nào trong bộ từ vựng](figures/demo-khong-du-doan.png)

