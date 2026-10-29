# Biên bản kiểm tra phần Decision Tree

Ngày chạy: 29/10/2026, Windows, Python 3.12.14. Phiên bản dependencies đã dùng được ghi trong `requirements-tested-py312.txt`. Đây là môi trường đã kiểm tra cục bộ; CI trên GitHub có trạng thái riêng.

## Kết quả đã chạy

- Cài package editable thành công.
- `python -m pytest -q`: **48 passed**.
- `python -m spam_detector.cli check-config`: valid, implemented_models=[tree].
- `python -m compileall -q src`: thành công.
- `git diff --check`: thành công trước commit.
- Demo có benchmark: tạo run, bundle, rules, predictions, config và môi trường.
- Tune trên 8 tổ hợp depth/leaf: tạo bảng ứng viên và chọn bằng validation.
- Evaluate test từ bundle đã lưu: chạy thành công, không huấn luyện lại.
- Thử tải 5.000 mẫu số nhị phân × 500 đặc trưng trên quy tắc tổng hợp: fit khoảng 0,027 giây ở máy kiểm tra, cây sâu 3 và 7 nút. Đây là phép kiểm tra kỹ thuật trên dữ liệu dễ; không đại diện tốc độ với email thật hoặc cây sâu hơn.

## Phạm vi kiểm thử

Tính tay Gini/phép chia, trọng số số mẫu, quy tắc hòa, nút thuần/cột hằng/XOR, độ sâu và kích thước lá; shape/nhãn/tham số sai; xác suất; predict trước fit; round-trip JSON; bundle hỏng; cấu hình YAML; ID/nhóm giao nhau; sai thứ tự tên cột; train không đọc test; CLI; chọn cấu hình validation; đối chiếu sklearn trên trường hợp phép chia tốt nhất xác định.

## Tái lập

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-tested-py312.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m spam_detector.cli demo-tree --output runs/verification-demo --benchmark
.venv\Scripts\python -m spam_detector.cli tune-tree --data runs/verification-demo/synthetic.npz --output runs/verification-tuning --depths 0 3 5 10 --leaf-sizes 1 5
.venv\Scripts\python -m spam_detector.cli evaluate-tree --bundle runs/verification-demo/run/tree.json --data runs/verification-demo/synthetic.npz --split test
```

Dùng tên thư mục output mới khi chạy lại. File khóa phiên bản dành cho Python 3.12; môi trường khác dùng dependencies trong pyproject và kiểm tra lại. Dữ liệu demo được sinh cố định bằng seed 42, độc lập với seed của benchmark cấu hình để tái lập fixtures kỹ thuật.

## Chưa được chứng minh bằng lần chạy này

Chưa có dữ liệu thật nên chưa có kết quả Enron/SpamAssassin, độ tổng quát, CV trên email, quality của bộ biến đổi hay tích hợp ensemble/demo toàn nhóm. Không diễn giải F1=1 trên dữ liệu giả lập thành độ chính xác lọc spam thật.
