# Hợp đồng tích hợp cho phần Decision Tree

## Phạm vi đã triển khai

Package `spam_detector`, cấu hình baseline, interface mô hình, Decision Tree tự viết, bundle JSON và CLI cho nhánh cây đã chạy được. Đã bổ sung data.py/preprocessing.py/features.py, pipeline train-enron và inference.py để train/dự đoán email thật. NB/LR/SVM, ensemble đủ bốn mô hình và Streamlit toàn nhóm vẫn theo phân công; không có model giả thay thế.

`tree_workflow.py` là đường chạy độc lập phục vụ nghiệm thu phần Doanh. Khi Dương hoàn thiện evaluator/training chung, dùng lại `DecisionTreeClassifier`, `TreeBundle` và giao thức đầu vào; hợp nhất hàm metrics vào evaluator chung thay vì duy trì hai cách tính.

## Interface mô hình

```python
from spam_detector import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=10, min_samples_split=10, min_samples_leaf=5,
)
model.fit(X_train, y_train)
labels = model.predict(X_validation)
probabilities = model.predict_proba(X_validation)
rules = model.export_rules(feature_names)
```

- X: dense array `(n_samples, n_features)`, số hữu hạn 0/1; khuyến nghị `uint8`.
- y: vector `(n_samples,)`, 0=ham, 1=spam. Fit yêu cầu hai lớp; các lá có thể một lớp.
- predict: vector int64, giữ nguyên thứ tự hàng; hòa chọn ham.
- predict_proba: float64 `(n_samples, 2)`, cột `[ham, spam]`; tỷ lệ lớp ở lá, chưa hiệu chuẩn.
- Batch dự đoán rỗng `(0, n_features)` hợp lệ; tập train rỗng không hợp lệ.
- `max_depth=0` tạo stump; `None` không giới hạn. Không chấp nhận độ sâu âm, tham số bool hoặc số lẻ.
- `min_samples_split` tối thiểu 2; `min_samples_leaf` tối thiểu 1. Nếu không đủ mẫu để tách, giữ lá.
- Ngưỡng nhị phân cố định 0,5; bên trái không có từ, bên phải có từ.
- Hai cột có cùng impurity chính xác: ưu tiên cột có index nhỏ hơn. Chỉ nhận gain > 1e-12.
- `NotFittedError` khi predict/export trước fit; `ValueError` khi dữ liệu/tham số sai. Fit thất bại trước khi công bố state không làm mất mô hình đã fit hợp lệ.

Module `models/base.py` chứa Protocol `Classifier`, `ProbabilityClassifier`, `ScoredClassifier`. SVM chỉ cần fit/predict/decision_function; không cần xác suất giả.

## Bàn giao từ Minh và Hiệu/Dương

Minh cung cấp đặc trưng đã chọn trên train, cùng thứ tự cột cho mọi tập. Hiệu/Dương cung cấp ID/nhóm đã khử trùng lặp, giữ nguyên thứ tự giữa X/y/ID. Chỉ nhị phân hóa ma trận sau khi chọn tập cột cần thiết; không densify toàn bộ từ điển.

CLI nhận file `.npz`, không chứa object array hoặc dữ liệu pickle:

```python
import numpy as np

# X/y/ID/nhóm dưới đây lấy từ pipeline thật của nhóm.
np.savez_compressed(
    "tree_features.npz",
    feature_names=np.asarray(feature_names, dtype=str),
    X_train=X_train.astype(np.uint8), y_train=y_train,
    ids_train=np.asarray(ids_train, dtype=str),
    groups_train=np.asarray(groups_train, dtype=str),
    X_validation=X_validation.astype(np.uint8), y_validation=y_validation,
    ids_validation=np.asarray(ids_validation, dtype=str),
    groups_validation=np.asarray(groups_validation, dtype=str),
)
```

Đoạn ví dụ giả định X đã là 0/1; không dùng ép uint8 để âm thầm biến TF-IDF thành nhị phân. Phải thực hiện `(X_selected > 0)` ở pipeline trước đó.

`feature_names`: chuỗi 1 chiều, duy nhất, không rỗng. X phải đúng số cột và không vượt `select_k` trong cấu hình. Nhóm trùng lặp có thể xuất hiện nhiều hàng trong cùng tập; không được giao nhau giữa train và validation.

Đánh giá test/external dùng các khóa tương ứng `X_test`, `y_test`, `ids_test`, `groups_test` hoặc hậu tố `external_test`, kèm feature_names y hệt. Có thể đặt trong archive riêng; test phải được transform bằng bộ biến đổi của train, không chọn lại từ bằng nhãn test.

CLI kiểm tra ID/nhóm trùng, shape và tên cột; không thể tự chứng minh ma trận do thành viên cung cấp đã được vector hóa đúng. Người tạo archive phải lưu bộ biến đổi, cấu hình và nguồn dữ liệu theo thiết kế chung. File JSON cây chỉ chứa state cây + tên từ + metadata; chưa phải pipeline văn bản hoàn chỉnh.

Luồng `train-enron` hiện đã tự tạo archive từ CSV bằng pipeline học trên train. Release văn bản ghép `tree.json`, `features.json` và `release.json` có checksum; loader `TextPredictor` kiểm tra phiên bản tiền xử lý và thứ tự cột. Bộ biến đổi nhị phân lưu bằng JSON chứa từ đã chọn và cấu hình thay cho joblib: transform từ vựng cố định vẫn cho kết quả tương đương, dễ kiểm tra và không cần pickle. Khi các thành viên ghép module dùng chung, giữ hợp đồng này hoặc chuyển phiên bản format rõ ràng.

## Lưu và đọc mô hình

```python
from spam_detector.artifacts import load_tree_bundle, save_tree_bundle

save_tree_bundle("runs/my-run/tree.json", model, feature_names)
bundle = load_tree_bundle("runs/my-run/tree.json")
predictions = bundle.model.predict(X_new)
```

Bundle lưu format version, tham số, nút, thứ tự đặc trưng và metadata. Loader kiểm tra cấu trúc cây, chỉ số con, số mẫu, depth và cột trước khi dự đoán. File lỗi trả ValueError. Luồng train CLI tạo thư mục run mới, không ghi đè một lần chạy có sẵn.

## File đầu ra của train-tree/tune-tree

- `tree.json`: bundle cây đã chọn, kèm ID/nhóm train/validation.
- `rules.json`: từng đường đi đến lá, số mẫu và tỷ lệ spam.
- `summary.json`: metrics validation, tham số, depth, số nút/lá, thời gian và checksum archive.
- `predictions.validation.csv`: ID, nhãn thật/dự đoán, tỷ lệ spam ở lá.
- `candidates.validation.csv`: các cấu hình đã thử và kết quả validation.
- `config.resolved.json`, `environment.json`: cấu hình và phiên bản chạy.
- `benchmark.json`: chỉ tạo khi bật `--benchmark`, chứa so sánh sklearn.

Thời gian fit/predict không gồm tạo đặc trưng. Khi tune, xếp theo F1 validation, rồi Precision, rồi ưu tiên cây thực tế nông/ít nút; nếu vẫn bằng nhau giữ ứng viên đầu tiên. Không đọc mảng test trong train/tune.

Lệnh evaluate-tree in JSON để lưu hoặc ghép vào evaluator nhóm. Khi đánh giá test/external, đối chiếu ID/nhóm với metadata train/validation và tên cột với bundle. Chỉ chạy sau khi khóa cấu hình; CLI không thay thế việc quản lý test kín của trưởng nhóm.
