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

Ngoài ra, hàm `explain` liệt kê các từ có mặt trong email, xếp theo mức đóng góp vào quyết định spam $f_w \,(\log P(w \mid spam) - \log P(w \mid ham))$. Chương trình demo dùng hàm này để giải thích kết quả cho người dùng. Mô hình được kiểm thử tự động bằng các ví dụ tính tay, ví dụ: với tập huấn luyện 3 email và 3 từ, xác suất dự đoán phải đúng bằng $[0{,}6;\ 0{,}4]$ như khi tính tay.

## 2. Đối chiếu kết quả với Weka

TODO

## 3. Sử dụng mô hình để dự đoán

TODO
