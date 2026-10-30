# Biên bản kiểm tra phần Decision Tree

## Cập nhật web ngày 30/10/2026

- Tổng bộ kiểm thử: **74 passed**; thêm 17 service/AppTest tests cho web. Môi trường web kiểm tra cục bộ dùng Streamlit 1.66.1 và Python 3.12.14.
- Có đối chiếu label, tỷ lệ lá và path của service web với TextPredictor/CLI trên cùng input/release; có kiểm tra model thiếu, lỗi runtime, score hỏng, input rỗng/dài, session, kết quả cũ, ví dụ và ensemble fixture.
- Đã chạy HTTP server local, nhận HTTP 200; kiểm tra trình duyệt với viewport desktop 1440x1100 và mobile 390x844, không tràn ngang (documentWidth=viewport=390), không có console error/warning trong lần kiểm tra cuối.
- Đã submit bằng form, xem nhãn ham từ email minh họa, mở path cây, kiểm tra điều hướng focus và đọc lại screenshot. Không dùng ví dụ này để đánh giá độ chính xác chung.
- Browser QA phát hiện Streamlit tự render return object của biểu thức điều kiện; đã đổi sang if/else và bổ sung regression assertion để không xuất nội dung nội bộ lên UI.
- Ảnh QA nằm ở output/playwright (ignore trong Git). Không thay đổi/training lại model hoặc checksum release cho thay đổi web.
- Hướng dẫn và giới hạn thực tế: docs/WEB_USAGE.md. NB/LR/SVM thật và ensemble đủ bốn mô hình còn phụ thuộc bàn giao, fixture kiểm thử không được dùng làm output của ứng dụng.

## Cập nhật kiểm tra pipeline email thật

- Bộ kiểm thử hiện tại: **57 passed**, gồm reader/khử trùng, chia tập không phụ thuộc thứ tự, vocabulary không học từ validation, CV fit mới từng fold, release checksum, CSV email dài, chạy toàn quy trình trên fixture không cần mạng và tải release đã công bố để dự đoán văn bản.
- Đã chạy train-enron trên Kaggle thật hai lần, cùng seed 42 và raw cache có checksum. Xác nhận sạch dữ liệu, ID split, K/tham số được chọn, state cây, tất cả dự đoán/metrics và CV mean/std giống nhau; thời gian có thể khác. Xem reports/enron-v1/reproducibility.json.
- Đã đọc lại release, đối chiếu dự đoán từ văn bản và từ ma trận test/external, kiểm tra hình xuất ra. Nguồn kết quả thật là reports/enron-v1/summary.json; báo cáo giả lập bên dưới chỉ nghiệm thu thuật toán.
- K=1000, max_depth=15, min_samples_leaf=1: Enron test F1=88,64%; external F1=44,00%. Benchmark sklearn trên cùng validation F1=88,995%, so với tự viết 88,462%; tỷ lệ dự đoán khớp 99,489%.
- Source checksum/seed/Python nằm trong reports/enron-v1/run_manifest.json. Các phiên bản đã dùng được cập nhật trong requirements-tested-py312.txt.
- `.gitattributes` giữ text ở LF trên các máy để Git không đổi byte JSON làm sai checksum release; ảnh và ZIP giữ dạng binary.
- Sau khi train, bước đóng gói chuẩn hóa JSON của release sang LF và tính lại checksum để khớp byte được lưu trong Git. State cây, tên từ và dự đoán không thay đổi. JSON writer cũng được sửa dùng LF cho các lần train sau; source checksum trong run_manifest phản ánh đúng mã tại thời điểm train, trước sửa đổi định dạng ghi file này.

Các mục bên dưới lưu lại biên bản đợt đầu với dữ liệu giả lập; số 48 tests là kết quả lịch sử, không phải số kiểm thử mới nhất.

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
