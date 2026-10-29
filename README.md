# Email Spam Classification

Bài tập lớn môn Khai phá dữ liệu: phát hiện email spam tiếng Anh bằng bốn mô hình tự cài đặt và kết hợp bỏ phiếu có trọng số.

## Thành viên và phân công

- **Nguyễn Quốc Doanh:** Nhóm trưởng, Decision Tree, kiến trúc và nghiệm thu tích hợp.
- **Lê Hữu Hiệu:** Naive Bayes, dữ liệu, tiền xử lý, EDA và giao diện demo.
- **Thân Đức Minh:** Logistic Regression, biểu diễn và lựa chọn đặc trưng.
- **Nguyễn Thái Dương:** Linear SVM, đánh giá thực nghiệm và ensemble.

## Tài liệu dự án

- [Kế hoạch nhóm trong 4 tuần](KE_HOACH_NHOM.md): phạm vi, phân công, tiến độ và đầu ra từng giai đoạn.
- [Thiết kế khung toàn bộ dự án](THIET_KE_KHUNG_DU_AN.md): kiến trúc, cấu trúc thư mục dự kiến, hợp đồng dữ liệu/mô hình, huấn luyện, đánh giá, lưu pipeline, demo và kiểm thử.

## Trạng thái

Đã triển khai khung package Python, cấu hình, interface mô hình, Decision Tree tự viết bằng NumPy, lưu/tải bundle JSON, xuất luật, thử tham số và benchmark sklearn tùy chọn. Có kiểm thử và CI. NB/LR/SVM, pipeline email thật, ensemble và giao diện vẫn chờ triển khai theo phân công.

Các lệnh toàn dự án trong bản thiết kế là đích tích hợp. Những lệnh đã chạy được cho nhánh Decision Tree được liệt kê dưới đây.

Lộ trình dự kiến gồm 4 tuần, tạm tính từ 29/10 đến 25/11/2026, dự phòng 26-28/11. Hạn nộp chính thức sẽ được cập nhật theo thông báo môn học.

## Phạm vi triển khai

- Dữ liệu chính: Enron-Spam; đánh giá trên nguồn khác bằng SpamAssassin.
- Mô hình tự viết: Multinomial Naive Bayes, Logistic Regression, Linear SVM và Decision Tree.
- Ensemble: bỏ phiếu theo nhãn, trọng số từ F1 validation.
- Dự kiến dùng Python, NumPy/SciPy, pandas và Streamlit; thư viện mô hình dựng sẵn chỉ dùng trong thực nghiệm đối chiếu.
- Random Forest và các mở rộng ngoài email tiếng Anh để ở hướng phát triển.

## Bắt đầu phối hợp

1. Đọc bản thiết kế và chốt interface, quy ước nhãn, schema dữ liệu.
2. Dựng package và môi trường dùng chung.
3. Hoàn thành một luồng học thật từ email đến metrics và lưu mô hình.
4. Tích hợp lần lượt bốn mô hình, sau đó ensemble và demo.

Dữ liệu đầy đủ, môi trường ảo, mô hình đã học và đầu ra thực nghiệm lớn được loại khỏi Git. Khi triển khai, bổ sung hướng dẫn tải dữ liệu và phiên bản dependencies để tái lập kết quả.

## Cài đặt và chạy phần Decision Tree

Đã kiểm tra cục bộ với Python 3.12.14. Package yêu cầu Python >=3.11; CI có cấu hình kiểm tra 3.11/Linux và 3.12/Windows.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,benchmark]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m spam_detector.cli check-config
.venv\Scripts\python -m spam_detector.cli demo-tree --output runs/tree-demo --benchmark
```

Linux/macOS thay `.venv\Scripts\python` bằng `.venv/bin/python`. Nếu chỉ cần mô hình, cài `python -m pip install -e .`; sklearn chỉ cần cho extra `benchmark`. Thư mục `--output` phải chưa tồn tại; dùng tên mới cho mỗi lần chạy.

Demo tạo dữ liệu số học giả lập 4 đặc trưng, không tải hoặc mô phỏng kết quả Enron. Dùng demo để kiểm tra mô hình, xuất luật, round-trip và luồng đánh giá; không dùng điểm số này làm chất lượng email thực tế.

```powershell
.venv\Scripts\python -m spam_detector.cli tune-tree --data runs/tree-demo/synthetic.npz --output runs/tree-tuning --depths 3 5 10 --leaf-sizes 1 5
.venv\Scripts\python -m spam_detector.cli evaluate-tree --bundle runs/tree-demo/run/tree.json --data runs/tree-demo/synthetic.npz --split test
```

Khi có đặc trưng thật từ Minh và ID/nhóm từ Hiệu/Dương:

```powershell
.venv\Scripts\python -m spam_detector.cli train-tree --data data/processed/tree_features.npz --output runs/tree-enron-01 --benchmark
```

Archive phải chứa ma trận 0/1, nhãn, tên đặc trưng, ID và nhóm theo [hợp đồng bàn giao](docs/contracts.md). Train/tune chỉ đọc train và validation; test/external chỉ chạy sau khi chốt cấu hình.

## Tài liệu triển khai

- [Interface và định dạng NPZ/bundle](docs/contracts.md).
- [Tiến độ và phần bàn giao của Doanh](docs/DOANH_WORK.md).
- [Lý thuyết, cài đặt và báo cáo Decision Tree](reports/decision_tree.md).
- [Biên bản kiểm tra và phiên bản đã dùng](docs/verification.md).
- [Kế hoạch triển khai và train trên Enron/SpamAssassin](docs/KE_HOACH_TRAIN_ENRON.md).

Mã thuật toán chính: `src/spam_detector/models/decision_tree.py`. Các module mô hình không gọi sklearn. Phần import cây sklearn chỉ nằm trong nhánh benchmark tùy chọn của `tree_workflow.py` và các kiểm thử đối chiếu.
