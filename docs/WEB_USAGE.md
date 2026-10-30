# Sử dụng web kiểm tra email spam

## Cài đặt và chạy

Trong project root, dùng Python >=3.11:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[web,dev]"
.venv\Scripts\python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Mở http://127.0.0.1:8501. Nếu môi trường .venv đã có, bỏ bước tạo môi trường. Linux/macOS dùng `.venv/bin/python`. Dùng Ctrl+C trong terminal để dừng. Bản này chạy cục bộ, chưa triển khai hosting công khai.

Ứng dụng dùng release `artifacts/tree-enron-v1` trong repo; không tải dataset hoặc train lại khi mở web. Nếu chạy lệnh từ thư mục khác, chỉ ra đường dẫn tuyệt đối tới file app; đường dẫn release mặc định vẫn được tính từ project root của app.

## Kiểm tra một email

1. Giữ chế độ “Một mô hình” và Decision Tree ở sidebar.
2. Nhập body bằng tiếng Anh; Subject không bắt buộc. Có thể dán toàn bộ thư vào body và để Subject trống.
3. Bấm “Kiểm tra email”. Nhãn, release và thời gian inference hiển thị ở khu vực kết quả.
4. Mở “Vì sao Decision Tree có dự đoán này?” để xem đường đi, số mẫu ở lá và tỷ lệ spam của lá.
5. Sửa nội dung rồi submit lại để cập nhật. Kết quả luôn gắn với lần gửi gần nhất; sửa form chưa submit không tự chạy mô hình.

Xóa nội dung, chọn ví dụ hoặc đổi chế độ/mô hình sẽ xóa kết quả cũ. Submit email rỗng/quá giới hạn cũng bỏ kết quả cũ và đưa thông báo lỗi. Giới hạn mặc định là 100.000 ký tự cho Subject ghép với body, không cắt ngầm nội dung.

Các ví dụ công việc, quảng cáo và Enron chỉ là nội dung minh họa. Mô hình có thể đoán sai ví dụ công việc do khác nguồn huấn luyện. Không coi nhãn dự đoán là nhãn kiểm chứng.

## So sánh và ensemble

“So sánh mô hình sẵn sàng” chạy các adapter thực sự tải được và trả bảng kết quả. Hiện chỉ Decision Tree có release. NB/LR/SVM được ghi “Chưa bàn giao”; không có kết quả giả.

Ensemble chỉ xuất hiện khi đủ bốn adapter và cấu hình trọng số hợp lệ, trỏ đúng fingerprint manifest của từng release. Một mô hình lỗi runtime thì giữ kết quả các mô hình chạy được nhưng không tạo kết luận ensemble hoặc tự đổi trọng số.

## Cấu hình và cache

`configs/web.yaml` khai báo max_characters, default_model, model registry và ensemble. Quản trị viên có thể dùng biến môi trường `SPAM_WEB_CONFIG` để chỉ định file cấu hình khác; không có trường cho người dùng web tự nhập đường dẫn model.

Ứng dụng cache PredictionService theo session bằng st.cache_resource(scope="session"). Cache key chứa config và fingerprint các file release hiện được hỗ trợ. Khi file tree/features/manifest thay đổi, service và kết quả cũ bị thay thế. Nội dung email và kết quả không nằm trong cache toàn cục; không có body email trong log ứng dụng hoặc lịch sử chung. Kết quả vẫn ở bộ nhớ session trong lúc người dùng sử dụng trang.

Thay release bằng bản đã nghiệm thu, giữ checksum chính xác. Không chỉnh JSON model/feature bằng tay rồi bỏ qua bước xác thực.

## Bàn giao adapter mới

1. Thành viên cung cấp release văn bản hoàn chỉnh, preprocessing/feature pipeline đúng lúc train và manifest/checksum.
2. Bổ sung nhánh loader tin cậy trong `web_service.default_loader`; không import class tùy ý từ cấu hình người dùng.
3. Adapter cung cấp `predict_texts([text])` trả đúng một dictionary với label 0/1. NB/LR trả score_kind=`probability_uncalibrated` và score_value hợp lệ; SVM trả score_kind=`margin`, score_value có thể âm. Nếu có khả năng hiệu chuẩn khác, cập nhật tên loại điểm và UI tương ứng.
4. Decision Tree tiếp tục dùng spam_fraction, path và leaf_training_samples như TextPredictor hiện có.
5. Mở rộng registry_fingerprint để tính checksum các file trọng số/đặc trưng của adapter mới nếu không dùng tree.json/features.json. Khai báo đường dẫn release trong configs/web.yaml và kiểm thử loader lỗi, output lỗi, web/CLI tương đương.
6. Sau khi đủ bốn model, Dương cung cấp ensemble config theo mẫu dưới đây, thay mọi placeholder bằng dữ liệu thật. Không nhập trọng số mẫu để bật chức năng giả.

```yaml
ensemble:
  weight_source: validation_f1
  weights: {tree: <w_tree>, nb: <w_nb>, lr: <w_lr>, svm: <w_svm>}
  release_fingerprints:
    tree: <SHA256_of_tree_release_json>
    nb: <SHA256_of_nb_release_json>
    lr: <SHA256_of_lr_release_json>
    svm: <SHA256_of_svm_release_json>
```

Trọng số không âm, tổng bằng 1; được tính từ validation của các release được ghép. Ensemble tính weighted hard vote trên nhãn, ngưỡng >=0,5 trả spam. Điểm bỏ phiếu không phải xác suất đã hiệu chuẩn. Không so trực tiếp margin SVM với tỷ lệ lá hoặc xác suất NB/LR để kết luận mô hình nào chắc chắn hơn.

## Giới hạn của release cây

Test Enron F1=88,64%; SpamAssassin F1=44,00%, với nhiều false positive. Web hiển thị cả hai kết quả trong phần chất lượng mô hình. Release này phục vụ demo và phân tích, chưa thích hợp tự động chặn thư thật ngoài miền Enron. Tỷ lệ spam tại lá chưa hiệu chuẩn cho email mới.

## Kiểm thử

```powershell
.venv\Scripts\python -m pip install -e ".[dev,training,benchmark,web]"
.venv\Scripts\python -m pytest -q
```

AppTest kiểm tra form/session, submit, lỗi, ví dụ, xóa và chuyển chế độ. Service tests kiểm tra tính tương đương pipeline, input, score, lỗi từng mô hình, checksum và điều kiện ensemble bằng adapter fixture. Fixture ensemble không phải mô hình đã train thật và không được ứng dụng production sử dụng.

Đã kiểm tra trình duyệt ở desktop và viewport 390px: submit và kết quả, mở luật cây, bố cục co lại và không có tràn ngang tài liệu. Ảnh kiểm tra được giữ cục bộ ở output/playwright, không đưa email người dùng vào screenshot mặc định. Các test và demo hiện dùng dữ liệu giả lập hoặc mẫu minh họa do nhóm đặt.
