# Kế hoạch giao diện web kiểm tra email spam

Ngày lập: 30/10/2026. Phạm vi: giao diện phục vụ demo môn Khai phá dữ liệu và tích hợp bốn mô hình của nhóm.

**Cập nhật triển khai:** Đã có app/streamlit_app.py, configs/web.yaml, PredictionService, kiểm thử service/AppTest và bản web Decision Tree dùng release thật. Xem docs/WEB_USAGE.md để cài/chạy và ghép adapter mới. Các đoạn “chưa được lập trình” và lệnh mục tiêu phía dưới giữ lại bối cảnh bản kế hoạch; trạng thái mới nhất là README và WEB_USAGE. NB/LR/SVM/ensemble đầy đủ vẫn cần các release của thành viên.

## 1. Mục tiêu và phần có thể sử dụng ngay

Người dùng nhập email tiếng Anh, chọn cách kiểm tra, bấm phân loại và xem dự đoán ham/spam. Kết quả giải thích mô hình đã dùng tín hiệu nào và cho phép so sánh các mô hình khi chúng đã được bàn giao.

Repo hiện đã có `TextPredictor.predict_texts()` và release Decision Tree thật ở `artifacts/tree-enron-v1/`. Có thể làm bản web dùng cây ngay. NB/LR/SVM và ensemble đủ bốn mô hình chưa có release triển khai trong repo; giao diện không hiển thị dự đoán giả cho các phần này.

Đây là kế hoạch xây dựng, chưa phải giao diện đã được lập trình. Các tên file/lệnh web bên dưới là đích triển khai.

## 2. Lựa chọn công nghệ

Dùng **Streamlit** làm giao diện và gọi package Python hiện có trong cùng tiến trình. Bản đầu chạy cục bộ để cả nhóm kiểm tra và thuyết trình. Không cần tách backend HTTP hoặc bổ sung database cho chức năng kiểm tra một email.

- Dùng `st.form` để gửi nội dung/chế độ kiểm tra bằng một lần submit, tránh dự đoán lại khi người dùng chỉ chỉnh widget.
- Dùng `st.cache_resource` để tải predictor theo đường dẫn release và fingerprint manifest. Cache chỉ giữ tài nguyên mô hình; không cache nội dung email hoặc kết quả người dùng ở phạm vi global.
- Predictor đã cache được dùng chỉ để đọc/predict, không fit hoặc thay đổi vocabulary. Nếu adapter mới có state thay đổi khi predict, dùng cache theo session hoặc sửa adapter trước khi dùng chung nhiều phiên.
- `st.session_state` chỉ giữ dữ liệu giao diện trong phiên: nội dung đang nhập, lần gửi cuối và kết quả tương ứng. Không coi đây là cơ chế lưu lịch sử lâu dài.
- Dùng `streamlit.testing.v1.AppTest` kiểm tra luồng nhập/submit/thông báo; kiểm tra trình duyệt thật riêng cho bố cục và bàn phím.
- Thêm extra `web` vào pyproject; cài và chốt phiên bản thực sự đã kiểm tra khi triển khai.

Tài liệu chính thức: [forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form), [cache_resource](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource), [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

## 3. Phạm vi phiên bản đầu

### Chức năng cần có

1. Ô Subject tùy chọn và vùng nhập body email bắt buộc, nhiều dòng.
2. Chế độ kiểm tra một mô hình hoặc so sánh các mô hình đang sẵn sàng.
3. Nút “Kiểm tra email”, “Xóa nội dung” và một số ví dụ giả lập để demo.
4. Hiển thị nhãn dự đoán, mô hình/release, thời gian xử lý và thông báo liên quan.
5. Xem giải thích Decision Tree theo chuỗi từ có/không xuất hiện dẫn đến lá.
6. Hiển thị rõ trạng thái các mô hình chưa tích hợp; chỉ cho chọn mô hình tải được release hợp lệ.
7. Thông tin phạm vi tiếng Anh và giới hạn của mô hình, cùng đường dẫn báo cáo kết quả.
8. Khi đủ bốn mô hình và cấu hình trọng số hợp lệ, bật chế độ ensemble.

### Mở rộng sau khi chức năng chính ổn định

Upload `.txt`/`.eml`, kiểm tra nhiều email, xuất kết quả JSON/CSV, thống kê lịch sử theo phiên và triển khai hosting. Các chức năng này là đầu việc riêng, không làm chậm demo nhập một email.

## 4. Bố cục màn hình

### Khu vực đầu trang

- Tiêu đề: “Kiểm tra email spam”.
- Mô tả ngắn: “Nhập email tiếng Anh để xem dự đoán của mô hình”.
- Dòng thông tin: “Bản demo học thuật; kết quả có thể sai, đặc biệt với dữ liệu khác nguồn huấn luyện”.
- Trạng thái ngắn: số mô hình sẵn sàng; không đưa checksum, tham số code hoặc log kỹ thuật lên vùng chính.

### Khu vực nhập

- Subject: một dòng, không bắt buộc.
- Nội dung email: textarea đủ lớn, giữ xuống dòng để người dùng kiểm tra thư đã nhập.
- Chọn chế độ và mô hình hợp lệ; bản đầu mặc định Decision Tree.
- Nút kiểm tra là hành động chính. Xóa và chọn ví dụ nằm ở nhóm phụ.
- Ví dụ chỉ là nội dung minh họa, không gắn “chắc chắn đúng” hoặc chọn ví dụ để che giấu lỗi mô hình.

### Khu vực kết quả

Kết quả chỉ xuất hiện sau submit thành công. Tiêu đề dùng “Mô hình dự đoán: Spam” hoặc “Mô hình dự đoán: Thư thường (ham)”. Nhãn luôn có chữ, không chỉ dựa vào màu.

Trong chế độ so sánh, trình bày một hàng cho mỗi mô hình thực sự chạy được: tên mô hình, nhãn, loại điểm số và thời gian. Nếu các mô hình bất đồng, ghi “Các mô hình cho kết quả khác nhau”; không tự chọn một kết luận chung nếu chưa có ensemble hợp lệ.

Với Decision Tree, phần “Vì sao có dự đoán này?” mở rộng để xem:

- Từng câu hỏi như “Có từ enron?” → “Không”.
- Số mẫu train đi đến lá tương ứng.
- “Tỷ lệ spam tại lá” với giải thích chưa hiệu chuẩn; không gọi là “độ tin cậy chính xác”.
- Thông báo khi không có từ nào khớp bộ đặc trưng đã chọn.

Phần thông tin mô hình mở rộng chứa tên release, nguồn dữ liệu, metrics test/external và liên kết báo cáo. Thông tin mặc định cần đủ ngắn để người dùng tập trung vào kiểm tra email.

### Màn hình nhỏ và thao tác bàn phím

Dùng bố cục xếp dọc làm gốc. Trên desktop có thể chia phần nhập và kết quả thành hai cột; trên màn hình nhỏ phải xếp lại, không ép hàng dài. Label gắn rõ với trường nhập, trạng thái xử lý có chữ, lỗi nằm gần trường tương ứng. Giữ thứ tự tab tự nhiên và kiểm tra toàn bộ luồng bằng bàn phím.

## 5. Luồng xử lý một email

1. Đọc cấu hình web và kiểm tra danh sách release khi khởi động.
2. Chỉ tải predictor cho mô hình đang dùng; cache theo release path và fingerprint, không theo nội dung email.
3. Người dùng nhập Subject/body và chọn chế độ.
4. Khi submit, xác thực body không trắng và tổng nội dung không vượt giới hạn mặc định đề xuất 100.000 ký tự. Không cắt ngầm nội dung quá dài; yêu cầu rút gọn hoặc xử lý bằng luồng batch sau này.
5. Nếu có Subject, ghép theo một quy tắc duy nhất `Subject: <subject>\n<body>`. Nếu người dùng dán toàn thư vào body và để Subject trống, giữ nguyên body; không tự nhân đôi Subject. Ghi rõ quy tắc cho các adapter.
6. Lưu snapshot lần gửi gồm nội dung chuẩn đầu vào, chế độ, danh sách release và mã request trong session.
7. Hiển thị trạng thái “Đang phân tích”, gọi service dự đoán; không gọi fit/train, tải lại dataset hoặc thay đổi selector.
8. Xác thực output và tạo view model chung cho giao diện.
9. Hiển thị kết quả gắn với lần gửi đó, cùng giải thích và cảnh báo thực tế nếu có.
10. Khi xóa/chọn ví dụ/chuyển release, xóa kết quả cũ. Khi form chưa gửi lại, ghi “Kết quả của lần kiểm tra gần nhất” để không khiến người dùng hiểu nhầm kết quả tự cập nhật theo bản đang chỉnh.

Nếu lỗi ở lần gửi mới, bỏ kết quả cũ khỏi vùng kết luận và hiển thị thông báo; không giữ nhãn cũ dưới nội dung mới.

## 6. Kiến trúc và hợp đồng tích hợp

Luồng phụ thuộc:

```text
app/streamlit_app.py
  → PredictionService
    → adapter của mô hình được chọn
      → bộ biến đổi và mô hình đúng release
    → chuẩn hóa kết quả cho giao diện
```

Giao diện không tự viết tokenizer, Gini, TF-IDF hoặc bỏ phiếu. Logic này phải dùng lại package đã kiểm tra.

### Các file dự kiến

```text
app/
  streamlit_app.py           # widget, form, trình bày kết quả
configs/
  web.yaml                  # model/release registry, mặc định, giới hạn input
src/spam_detector/
  web_service.py            # chọn adapter, xác thực input/output, đo thời gian
  inference.py              # TextPredictor cây đang có
  ensemble.py               # do Dương cung cấp khi đủ bốn mô hình
tests/
  test_web_service.py       # giao thức, lỗi, nhiều mô hình và ensemble
  test_web_app.py           # AppTest cho form/session/stale result
docs/
  WEB_USAGE.md              # cách chạy và kịch bản demo sau triển khai
```

Không cần tạo trước class rỗng cho ba mô hình chưa có. Chốt interface bàn giao, thêm adapter khi có implementation và release thật.

### Adapter mô hình

Mỗi adapter nhận danh sách text và trả một kết quả cho mỗi text, giữ thứ tự. DecisionTreeAdapter bọc `TextPredictor` hiện có, không sửa artifact đã train chỉ để phục vụ UI.

Kết quả chuẩn hóa cho một email gồm:

```text
model_id, model_name, release_id
label: 0 hoặc 1
score_kind: leaf_fraction / probability_uncalibrated / margin / vote_score
score_value: số hữu hạn nếu mô hình hỗ trợ
elapsed_ms
explanation: dữ liệu giải thích tùy mô hình
warnings: thông báo ngắn có căn cứ
```

- Decision Tree ánh xạ `spam_fraction` hiện tại thành leaf_fraction, giữ path và số mẫu lá.
- NB/LR chỉ dùng probability nếu adapter thực sự có output đó; mô tả trạng thái hiệu chuẩn trung thực.
- SVM dùng margin có thể âm hoặc dương, không đổi thành phần trăm xác suất bằng sigmoid tùy ý.
- Không so sánh độ lớn margin SVM với tỷ lệ lá hoặc xác suất NB để chọn mô hình “chắc chắn nhất”.

Trạng thái mô hình: `ready`, `not_configured`, `load_error`. Trạng thái lỗi runtime chỉ gắn với lần request cụ thể. Config chỉ cho phép đường dẫn release do nhóm đặt, không có ô người dùng tải model hoặc nhập đường dẫn tùy ý trong bản đầu.

## 7. Điều kiện bật ensemble

Ensemble chỉ bật khi đủ bốn adapter NB/LR/SVM/Tree và file cấu hình bỏ phiếu hợp lệ. Cấu hình phải ghi danh sách model_id theo thứ tự, release_id/fingerprint, trọng số tính từ validation và ngưỡng.

- Trọng số lấy từ F1 validation theo kế hoạch nhóm, không dùng test/external.
- Mỗi email gửi cho đúng pipeline của từng mô hình; không dùng ma trận của cây để dự đoán LR/SVM nếu khác bộ đặc trưng.
- Tổng phiếu có trọng số trên nhãn 0/1; score >=0,5 trả spam theo quy tắc đã chốt.
- “Điểm bỏ phiếu” là tên hiển thị, không gọi là xác suất spam đã hiệu chuẩn.
- Một mô hình lỗi: hiển thị kết quả mô hình đơn lẻ đã có, ensemble báo không đủ dữ liệu. Không tự phân phối lại trọng số rồi hiển thị như ensemble bốn mô hình.
- Khi một release thay đổi, phải xem lại tính phù hợp của trọng số validation; không tự dùng cấu hình trọng số cũ cho release mới.

Nút chế độ ensemble có thể được ẩn hoặc vô hiệu hóa kèm lý do “Chưa đủ 4 mô hình”; giao diện vẫn kiểm tra email bằng Decision Tree bình thường.

## 8. Trình bày đúng giới hạn mô hình hiện tại

Snapshot đang có: test Enron F1=88,64%, SpamAssassin F1=44,00%, external có nhiều thư thật bị chặn nhầm. Chất lượng trên corpus Enron không đồng nghĩa email bất kỳ đều được phân loại tốt.

- Trong phần thông tin mô hình, hiển thị cả kết quả test và external, không chỉ chọn chỉ số cao nhất.
- Kết luận trên màn hình là dự đoán để tham khảo, không xác nhận email nguy hiểm hoặc an toàn tuyệt đối.
- Nêu phạm vi email tiếng Anh; không tự tuyên bố hỗ trợ tiếng Việt vì tokenizer hiện tại không được thiết kế cho mục tiêu đó.
- Không mở link/attachment trong email; hiển thị nội dung dưới dạng văn bản, không render HTML email như trang web.
- Mặc định không ghi body email vào log, không lưu lịch sử toàn cục. Log lỗi dùng request_id, model_id và lỗi kỹ thuật; session không dùng chung nội dung giữa người dùng.

Các điểm trên ảnh hưởng trực tiếp cách người dùng diễn giải kết quả và cách bảo toàn nội dung họ nhập, nên cần nằm trong thiết kế bản đầu.

## 9. Kiểm thử và nghiệm thu

### Service và adapter

- Output web có cùng label/score với CLI/TextPredictor trên cùng nội dung và release.
- Không có lời gọi fit/train trong luồng dự đoán; vocabulary không đổi sau request.
- Tên mô hình/release, loại điểm số và số output đúng; giá trị không hữu hạn hoặc nhãn ngoài 0/1 bị phát hiện.
- Checksum hoặc thứ tự đặc trưng sai: không cho dùng release, thông báo dễ hiểu.
- Văn bản chỉ có từ ngoài bộ từ vẫn giữ một kết quả và cảnh báo đúng.
- Nhiều session không chia sẻ nội dung; cached predictor không bị mutate khi predict.
- Nếu có ensemble: kiểm thử hòa, sai thứ tự model_id, thiếu adapter, trọng số sai và model lỗi.

### Giao diện

- Mở trang không tự phân loại và không hiển thị kết quả mẫu như kết quả thật.
- Chỉ bấm nút kiểm tra mới chạy inference; body trắng và quá giới hạn có thông báo rõ.
- Chọn ví dụ/xóa nội dung/chuyển release không để lại kết quả cũ gây nhầm.
- Form hiển thị trạng thái đang xử lý rồi kết quả, không có exception chưa xử lý trên trang.
- Phân biệt spam/ham bằng chữ; dùng bàn phím được, không cắt path giải thích ở màn hình nhỏ.
- Một mô hình chưa bàn giao không làm hỏng chế độ Decision Tree; ensemble không bật thiếu điều kiện.
- Giới hạn ngoài nguồn nằm trong thông tin mô hình; tỷ lệ lá không bị gắn nhãn độ tin cậy đã hiệu chuẩn.

### Tiêu chí hiệu năng cho demo

Đo trên máy nghiệm thu với bộ email mẫu của nhóm; báo riêng lần tải mô hình đầu và request đã cache. Mục tiêu đề xuất: email dưới 10.000 ký tự phản hồi trong khoảng 2 giây sau khi cache đã ấm. Đây là mục tiêu cần đo, không phải hiệu năng đã đạt. Email lớn cần phản hồi trạng thái xử lý, không treo UI im lặng.

## 10. Phân công và lịch 5 ngày

### Ngày 1: Dựng trang và nối Decision Tree

Hiệu chủ trì Streamlit, Doanh cung cấp release và service bọc TextPredictor. Thêm extra web, config release, form nhập và kết quả nhãn. Nghiệm thu: kiểm tra được một email thật qua web và kết quả bằng CLI.

### Ngày 2: Giải thích, trạng thái và xử lý lỗi

Doanh hoàn thiện view model/path cây, Hiệu hoàn thiện form/session và thông báo. Minh rà việc dùng đúng bộ biến đổi. Nghiệm thu: email trắng/OOV/HTML, thiếu release và kết quả cũ đều xử lý được; không fit trong inference.

### Ngày 3: Registry và ghép các mô hình đã sẵn sàng

Mỗi người bàn giao adapter/release của mình; Minh cung cấp LR, Dương SVM, Hiệu NB. Doanh rà interface; Hiệu bổ sung chế độ so sánh. Nghiệm thu theo số mô hình thực sự có, ghi rõ phần còn thiếu, không tạo dự đoán thay thế.

### Ngày 4: Ensemble và kiểm thử luồng

Dương bàn giao ensemble + trọng số validation; Doanh kiểm tra khớp release, Hiệu tích hợp. Cả nhóm chạy AppTest và kịch bản bàn phím/màn hình nhỏ. Nếu chưa đủ mô hình, dùng ngày này hoàn thiện bản web cây/so sánh có thật, chưa công bố ensemble đầy đủ.

### Ngày 5: Nghiệm thu demo và tài liệu

Doanh chủ trì nghiệm thu, Hiệu ghi hướng dẫn/video dự phòng, mỗi người thử trên máy mình. Khóa version dependencies/release, chạy CI và ghi số liệu hiệu năng. Đầu ra: source web, cấu hình, kiểm thử, WEB_USAGE và kịch bản thuyết trình.

Mốc giao diện cây cơ bản: 1-2 ngày. Mốc đủ bốn mô hình/ensemble: phụ thuộc các release còn lại; không hứa hoàn tất trong 5 ngày nếu các phần đó chưa sẵn sàng. Không cần huấn luyện lại cây chỉ để làm giao diện.

## 11. Chạy và bàn giao dự kiến

Sau khi thêm extra web và app, lệnh mục tiêu là:

```powershell
.venv\Scripts\python -m pip install -e ".[training,web,dev]"
.venv\Scripts\python -m streamlit run app/streamlit_app.py
```

Hai lệnh này chưa áp dụng cho repo hiện tại vì extra web và app chưa được tạo. Cấu hình mặc định sẽ trỏ tới `artifacts/tree-enron-v1` bằng đường dẫn tính từ project root, không hardcode thư mục trên máy Doanh.

Tài liệu bàn giao phải nêu: cách cài/chạy, release đang dùng, trạng thái tích hợp từng mô hình, cách đổi release, cách diễn giải score và giới hạn đã đánh giá. Lịch sử hoặc file tải xuống nếu bổ sung phải gắn đúng snapshot request và release.

## 12. Điều kiện hoàn thành

- [ ] Web nhập một email và dự đoán đúng theo pipeline hiện có.
- [ ] Có đường đi cây, loại điểm số và phạm vi mô hình dễ hiểu.
- [ ] Có config/registry; phần chưa có release không tạo kết quả giả.
- [ ] So sánh các mô hình thực tế hoạt động, ensemble chỉ bật khi đủ điều kiện.
- [ ] Có xử lý lỗi/input/session và không lưu body trong log mặc định.
- [ ] Có kiểm thử service + AppTest, kiểm tra bố cục và bàn phím.
- [ ] Có hướng dẫn, version/release chốt, video dự phòng và kết quả nghiệm thu.

Ưu tiên triển khai: web dùng được cây đã train → giải thích và kiểm thử → ghép mô hình khác → ensemble → hoàn thiện demo. Các mở rộng upload/batch/hosting chỉ thực hiện sau khi luồng này ổn định.
