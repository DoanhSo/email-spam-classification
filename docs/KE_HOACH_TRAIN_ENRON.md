# Kế hoạch triển khai và huấn luyện Decision Tree trên dữ liệu Kaggle

**Cập nhật thực hiện:** Đã chạy toàn quy trình trên dữ liệu thật, gồm làm sạch, split, K=500/1.000, 18 ứng viên, CV, test/external, predictor văn bản và bàn giao validation. Xem `reports/enron-v1/REPORT.md`, `summary.json` và `artifacts/tree-enron-v1/`. Nội dung dưới đây giữ lại kế hoạch gốc; trạng thái triển khai mới nhất nằm trong README và báo cáo kết quả. Các CLI chuẩn bị/đặc trưng/dự đoán đã được bổ sung.

## 1. Mục tiêu và trạng thái hiện tại

Triển khai luồng chạy được từ CSV email thật → kiểm tra/chia dữ liệu → tiền xử lý → đặc trưng nhị phân → Decision Tree tự viết → chọn tham số → đánh giá độc lập → dự đoán email mới.

Repo đã có cây tự viết, cấu hình, JSON bundle, export luật, train-tree/tune-tree/evaluate-tree và kiểm thử. Phần còn thiếu để train email thật là reader CSV, chuẩn hóa/khử trùng lặp, split manifest, bộ biến đổi binary/Chi-square đã lưu và luồng dự đoán từ văn bản. Chạy demo hiện có chưa thay thế việc huấn luyện trên Enron.

Đây là kế hoạch triển khai tiếp theo; chưa có số liệu huấn luyện Enron trong tài liệu này. Những lệnh được đánh dấu “cần bổ sung” chưa có trong CLI hiện tại.

## 2. Dữ liệu sử dụng và vai trò

### Dữ liệu chính

- Nguồn: [Spam Mails Dataset / venky73](https://www.kaggle.com/datasets/venky73/spam-mails-dataset).
- File: `spam_ham_dataset.csv`.
- Bản CSV đã kiểm tra ngày 29/10/2026: 5.171 hàng, 3.672 ham và 1.499 spam; các cột là cột index không tên, `label`, `text`, `label_num`.
- Giữ `text` làm nội dung, `label_num` làm nhãn 0/1. Dùng `label` để kiểm tra tính nhất quán rồi loại khỏi X; không dùng index hoặc nhãn dạng chữ làm đặc trưng.
- Phát hiện ban đầu: 178 dòng lặp sau khi trim; số sau canonicalization có thể khác. Không mặc định số mẫu cuối cùng trước khi xử lý nhãn mâu thuẫn và trùng lặp.

### Đánh giá trên nguồn khác

- Nguồn: [Email Spam Dataset / nitishabharathi](https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset).
- Chỉ lấy `completeSpamAssassin.csv` cho thí nghiệm ngoài nguồn.
- Bản đã kiểm tra: 6.046 hàng, gồm 4.150 nhãn 0 và 1.896 nhãn 1; nội dung `Body`, nhãn `Label`; có 4 nội dung rỗng và 768 dòng lặp sau trim.
- Không dùng `enronSpamSubset.csv` của bộ này như một nguồn kiểm tra độc lập với Enron chính.
- Các số lượng trên là kiểm tra bản tải tại thời điểm đó; bước triển khai phải tải lại phiên bản xác định, lưu checksum và đếm lại, không chép các số này làm báo cáo làm sạch cuối cùng.

## 3. Phân công cho đợt triển khai

- **Doanh:** chủ trì tích hợp nhánh cây, train/tune, kiểm tra luật, benchmark và nghiệm thu bundle/predictor. Viết phần báo cáo kết quả cây.
- **Hiệu:** reader Enron/SpamAssassin, kiểm tra nhãn/nội dung, chuẩn hóa email và thống kê làm sạch; bàn giao dữ liệu có nguồn.
- **Minh:** binary vectorizer, Chi-square, K=500/1.000, lưu/tải bộ biến đổi; phối hợp Doanh kiểm tra thứ tự đặc trưng và predictor văn bản.
- **Dương:** split manifest dùng chung, kiểm tra ID/nhóm, fold, metrics/biểu đồ và ghép kết quả vào evaluator/ensemble nhóm.

Đợt này giữ nguyên phân công đã chốt. Khi một module chưa có, Doanh có thể chủ trì bản tích hợp tối thiểu với hợp đồng rõ ràng để unblock phần cây; ghi lại việc bàn giao để tránh hai thành viên viết hai phiên bản tiền xử lý khác nhau. Không yêu cầu phải xong ba mô hình khác mới được train cây.

## 4. Giai đoạn A: Tải, xác thực và chuẩn hóa dữ liệu

### A1. Tải và ghi nguồn

Lưu file nguồn nguyên vẹn dưới `data/raw/enron/` và `data/raw/spamassassin/`. Tạo manifest có URL dataset, owner/slug, version nếu lấy được, tên file, thời điểm tải, SHA-256, số hàng/cột và thông tin license hiển thị ở nguồn. Bản raw không đưa vào Git theo .gitignore hiện tại.

Giải nén chỉ file CSV cần dùng; không thay dữ liệu bằng bản GitHub không xác định nguồn. Nếu tải qua API không được, người phụ trách tải CSV từ trang Kaggle rồi đưa vào cùng đường dẫn; quy trình sau đó vẫn giống nhau.

### A2. Kiểm tra nội dung và nhãn

- Enron: xác nhận `label_num` chỉ có 0/1 và phù hợp `ham/spam` ở `label`.
- SpamAssassin: xác nhận schema và ánh xạ `Label`, không suy đoán nhãn từ nội dung.
- Không âm thầm đổi lỗi parse/encoding thành chuỗi rỗng hoặc bỏ hàng mà không ghi log.
- Nội dung null, trắng hoặc sau xử lý không còn chữ cần được thống kê riêng. Trường hợp dữ liệu không dùng được thì loại với lý do rõ ràng trước chia tập.
- Xuất báo cáo phân bố lớp, độ dài nội dung, số lỗi và số mẫu bị loại.

### A3. Xác định mẫu trùng và ID

Giữ text gốc để phân tích. Tạo bản canonical riêng để tìm trùng: chuẩn hóa Unicode, xuống dòng/khoảng trắng, trim và lowercase theo quy tắc cố định. Không xóa dấu câu/URL/quá nhiều token trong bước này vì dễ gộp những email khác nhau.

- `sample_id`: source + định danh ổn định của bản ghi gốc.
- `group_id`: hash nội dung canonical, độc lập với nhãn và index dòng sau khi trộn.
- Các bản cùng canonical và cùng nhãn: baseline giữ một bản theo thứ tự xác định, ghi số lượng bị loại.
- Nhóm có nhãn mâu thuẫn: xuất danh sách để kiểm tra; nếu không giải quyết được, loại cả nhóm khỏi baseline và ghi rõ.
- Gần trùng do reply/forward/subject khác nhau có thể còn lại; ghi giới hạn này, không tuyên bố đã khử toàn bộ near-duplicate nếu chỉ dùng exact canonical hash.

Đầu ra: `records.csv`, `data_quality.json`, `dataset_manifest.json`. Kiểm tra cân đối lại số lượng đầu vào = giữ lại + loại theo các nhóm lý do không đếm chồng.

**Nghiệm thu A:** Nội dung/nhãn/nguồn/ID hợp lệ; có báo cáo làm sạch; chưa có fit vectorizer hoặc chọn đặc trưng.

## 5. Giai đoạn B: Khóa các tập dữ liệu

Enron chia train/validation/test khoảng 60/20/20, seed=42, phân tầng theo nhãn. Nếu giữ nhiều bản cùng nhóm thay vì loại, bắt buộc nhóm không đi qua hai tập; ghi tỷ lệ thực tế vì có thể không đúng tuyệt đối 60/20/20.

Không cam kết số mẫu từng tập dựa trên 5.171 hàng raw. Chỉ tính sau khi làm sạch. Mỗi tập phải có cả ham và spam; nếu không đủ cho 5-fold thì giảm fold theo chính sách được ghi nhận, không chạy CV không hợp lệ.

Lưu `splits.csv` với sample_id, group_id, split và fingerprint records. Cùng dữ liệu và seed phải cho cùng thành viên mỗi tập, kể cả khi thứ tự dòng CSV thay đổi; sắp ID ổn định trước khi chia.

Đối với SpamAssassin: làm sạch/khử trùng lặp bằng cùng quy tắc và loại nhóm trùng với bất kỳ tập Enron nào khỏi external test; báo số lượng và phân bố còn lại. Không lấy nhãn external để sửa từ điển hoặc chọn tham số.

**Nghiệm thu B:** Không giao ID/nhóm giữa các tập; có checksum manifest; Dương dùng chính split này cho bốn mô hình. Các bước chọn cấu hình chỉ được cấp train và validation.

## 6. Giai đoạn C: Tiền xử lý và tạo đặc trưng cho cây

### C1. Baseline văn bản đơn giản, tái lập được

Giữ Subject và body trong trường text theo dữ liệu CSV; không đọc chuỗi CSV như file MIME thô nếu không đúng định dạng. Bỏ HTML bằng quy tắc chung, chuẩn hóa lowercase, thay URL/email/tiền/số bằng token quy ước, tách từ. Kiểm tra token đặc biệt còn tồn tại sau tokenizer.

Đợt đầu không mở đồng thời stemming, lemmatization và nhiều n-gram. Chốt một baseline đơn giản, sau đó thay một yếu tố mỗi thí nghiệm. Stopword/stemming nếu bật phải được lưu trong cấu hình và dùng giống hệt khi predict.

### C2. Biểu diễn nhị phân

- Vectorizer học từ train; dùng word presence 0/1 (ví dụ CountVectorizer với binary=True).
- Cấu hình khởi đầu: unigram, min_df=2, bộ quy tắc tokenizer cố định. Các giá trị này là ứng viên baseline, không phải kết quả tối ưu đã biết.
- Chi-square fit trên ma trận train và y_train; chọn K=500, thử thêm K=1.000 sau khi baseline đúng.
- Với vocabulary nhỏ hơn K, dùng số từ thực tế và ghi `effective_k`; không bịa cột hoặc lỗi không rõ lý do.
- Giữ dạng sparse đến sau chọn cột; sau đó đổi `(X_selected > 0)` thành dense uint8 cho Tree.
- Ghi tên từ theo chính thứ tự cột output, không tự sort sau transform.

Minh lưu một bộ biến đổi cho mỗi cấu hình K cùng tokenizer, vocabulary, selector và feature_names. Validation/test/external chỉ transform. Pipeline K=500 và K=1.000 dùng cùng ID split nhưng khác bộ cột, nên không dùng chung bundle cây.

### C3. Archive bàn giao

Tuân theo `docs/contracts.md`. Archive phát triển chứa `feature_names`, `X_train`, `y_train`, `ids_train`, `groups_train` và các khóa hậu tố validation. Archive test/external riêng chứa các khóa tương ứng và cùng tên/thứ tự cột.

File gợi ý:

```text
data/processed/tree/k500.development.npz
data/processed/tree/k500.test.npz
data/processed/tree/k500.external.npz
data/processed/tree/k1000.development.npz
data/processed/tree/k1000.test.npz
data/processed/tree/k1000.external.npz
```

Có thể trì hoãn tạo archive test/external đến khi chốt bộ biến đổi để hạn chế mở test sớm. Mỗi archive phải trỏ đến fingerprint split và bộ biến đổi của mình.

**Nghiệm thu C:** Một từ chỉ có ở validation không lọt vào vocabulary; tên cột/shape đúng; load/tải bộ biến đổi cho cùng output; kiểm tra ID/X/y đồng bộ và không mất hàng OOV.

## 7. Giai đoạn D: Huấn luyện baseline

Cấu hình cây khởi đầu dùng baseline hiện có:

```yaml
seed: 42
data:
  split_ratios: [0.60, 0.20, 0.20]
features:
  tree:
    representation: binary
    select_k: 500
models:
  tree:
    max_depth: 10
    min_samples_split: 10
    min_samples_leaf: 5
```

Lệnh **đã có**, chỉ chạy được sau khi có archive đúng hợp đồng:

```powershell
.venv\Scripts\python -m spam_detector.cli train-tree --config configs/baseline.yaml --data data/processed/tree/k500.development.npz --output runs/enron-tree-k500-baseline --benchmark
```

Nghiệm thu đầu tiên bằng dữ liệu thật: fit không lỗi; dự đoán đủ số hàng validation; tỷ lệ lớp không bất thường; có luật và tên từ hợp lệ. Xem số nút, độ sâu, thời gian, Precision/Recall/F1, FP/FN. Cần bổ sung metrics trên train để so sánh train-vs-validation, vì workflow hiện tại mới xuất metrics validation.

Đối chiếu sklearn cùng X, split và tham số; ghi sai khác về tie-breaking/zero-gain. Không dùng sklearn thay mô hình tự viết. Đừng vội tối ưu khi phép chia, quy tắc lá hoặc ID đang sai.

**Nghiệm thu D:** Có run baseline, bundle, luật, dự đoán theo ID, metrics và benchmark; toàn bộ số liệu gắn nhãn validation, chưa mở test.

## 8. Giai đoạn E: Chọn tham số và kiểm tra độ ổn định

### E1. Thử ít cấu hình trước

Giữ min_samples_split=10; thử max_depth=5/10/15 và min_samples_leaf=1/5/10, tổng 9 cấu hình cho mỗi K. Với K=500/1.000 là 18 ứng viên trong lần khảo sát đầu, chưa phải toàn bộ E5.

```powershell
.venv\Scripts\python -m spam_detector.cli tune-tree --config configs/baseline.yaml --data data/processed/tree/k500.development.npz --output runs/enron-tree-k500-tune --depths 5 10 15 --leaf-sizes 1 5 10 --benchmark
```

Với K=1.000: tạo `configs/tree-k1000.yaml` chỉ khai báo các section hiện được config loader hỗ trợ và đổi select_k=1000, dùng đúng archive K=1.000, output khác. Không thêm tùy ý preprocessing/evaluation vào YAML hiện tại vì loader sẽ từ chối các section chưa hỗ trợ.

Workflow hiện tại xếp ứng viên theo F1 validation, rồi Precision, rồi ưu tiên cây nông/ít nút. Giữ chính sách này cho baseline; đồng thời xem FP và Recall để giải thích đánh đổi. Nếu nhóm muốn Precision là mục tiêu ràng buộc, phải chốt trước giá trị/mức Recall tối thiểu và sửa selector cùng kiểm thử trước khi chọn; không đổi tiêu chí sau khi nhìn test.

Lệnh tune-tree chỉ chọn trong một archive K. Cần thêm bước tổng hợp giữa K=500 và K=1.000 để chọn pipeline+mô hình cuối bằng cùng chính sách và ghi config_id đầy đủ.

### E2. Cross-validation

Bổ sung runner CV 5-fold trên phần train để khảo sát độ ổn định. Mỗi fold có splitter theo ID/nhóm, fit mới tokenizer có học/vocabulary/Chi-square và model trên fold-train, chấm trên fold-validation. Không CV trực tiếp các archive K đã được selector fit trên toàn train.

Lịch 1 tháng: chạy CV trên baseline và số cấu hình rút gọn, tránh lồng 18 ứng viên × mọi biến thể tiền xử lý × mọi K. CV mean/std và kết quả validation trình bày riêng. Nếu dùng CV để chọn tham số, ghi rõ điểm CV đó là điểm lựa chọn cấu hình; kết quả test mới là đánh giá độc lập cuối cùng.

### E3. Ngưỡng phân loại

Baseline giữ quy tắc lá đa số, hòa chọn ham tương ứng `spam_fraction > 0.5`. Nếu khảo sát ngưỡng khác, chỉ dùng validation, lưu ngưỡng trong bundle/predictor và kiểm thử tính nhất quán. Chưa thêm tìm ngưỡng vào đợt đầu để tránh thay đổi đồng thời quá nhiều yếu tố.

**Nghiệm thu E:** Có bảng 18 ứng viên hoặc số đã chạy thực tế, lý do chọn K/tham số, số liệu train/validation, CV đã thực hiện và giới hạn; không tự ghi kết quả chưa chạy.

## 9. Giai đoạn F: Đánh giá cuối và phân tích lỗi

Chốt pipeline, tham số cây và quy tắc dự đoán; giữ mô hình học trên train như kế hoạch chung. Không refit trên train+validation mà vẫn giữ metadata/trọng số ensemble được tính cho mô hình cũ.

Lệnh **đã có**, chạy sau khi chọn run cuối:

```powershell
.venv\Scripts\python -m spam_detector.cli evaluate-tree --bundle runs/enron-tree-k500-tune/tree.json --data data/processed/tree/k500.test.npz --split test
.venv\Scripts\python -m spam_detector.cli evaluate-tree --bundle runs/enron-tree-k500-tune/tree.json --data data/processed/tree/k500.external.npz --split external_test
```

Ví dụ trên giả định pipeline K=500 thắng; nếu K=1.000 thắng, đổi cả bundle và archive, không trộn bộ từ. Lệnh hiện tại in summary JSON; cần bổ sung xuất file metrics, dự đoán theo ID và FP/FN cho test/external để truy vết và làm báo cáo.

Chỉ số: Precision spam, Recall spam, F1 spam, Accuracy, TN/FP/FN/TP, số mẫu/lớp, thời gian transform/fit/predict riêng. Có thể thêm PR/ROC từ tỷ lệ spam ở lá, nêu rõ score chưa hiệu chuẩn; không dùng nhãn cứng làm điểm số. Không đặt trước F1 bắt buộc đạt 95% khi chưa biết chất lượng và cách chia dữ liệu.

Phân tích ít nhất vài mẫu FP/FN nếu có: đọc nội dung, đường đi cây, từ OOV, lá nhỏ, từ đa nghĩa, thư công việc chứa money/URL. Không đưa mẫu sai vào train để cải thiện rồi báo lại kết quả trên cùng test như đánh giá độc lập.

**Nghiệm thu F:** Có số liệu test và external riêng, cấu hình đã khóa, bảng lỗi và giải thích giới hạn; kết luận dựa trên số liệu thật.

## 10. Giai đoạn G: Predictor văn bản, đóng gói và bàn giao ensemble

Gói cây phải gắn đúng bộ biến đổi và chính sách preprocessing. JSON cây hiện tại chỉ lưu cây/tên từ/metadata; chưa thể tự đọc email mới.

Bổ sung bundle cấp pipeline có:

```text
releases/tree-enron-v1/
  release.json                 # đường dẫn/checksum, cấu hình và phiên bản
  tree.json
  feature_pipeline.joblib      # vocabulary + selector đã fit trên train
  preprocessing.json
  feature_names.json
  metrics.test.json
  metrics.external.json
```

`predict_texts(texts)` dùng đúng thứ tự preprocess → transform → predict/predict_proba. Không fit lại trong demo. Kiểm thử email trắng, HTML, URL, toàn bộ từ ngoài vocabulary và vòng lưu/tải. Thông báo mô hình nhận ít tín hiệu nếu nhiều từ ngoài bộ từ; không gọi tỷ lệ lá là xác suất đã hiệu chuẩn.

Giao Dương: bundle đúng release + `sample_id,y_true,y_pred,spam_fraction` trên validation + config_id + thứ tự feature names. Dương ghép theo ID để tính trọng số ensemble; không tính trọng số bằng test F1. Hiệu dùng predictor cho demo, không chép riêng thuật toán cây vào giao diện.

## 11. Danh sách thay đổi cần lập trình

1. `data.py`: reader Enron/SpamAssassin, schema, nhãn và báo cáo dữ liệu.
2. `preprocessing.py`: canonical hash cho trùng lặp và biến đổi nội dung dùng chung.
3. Chức năng split/manifest: đóng băng ID và kiểm tra lớp/nhóm; thống nhất với Dương.
4. `features.py`: binary/Chi-square fit/transform/save/load; tích hợp với Minh.
5. `cli.py`: thêm `prepare-enron` và `build-tree-features` để tạo đầu vào thực cho train-tree.
6. `tree_workflow.py`: bổ sung metrics train, tổng hợp K, CV từ văn bản và xuất dự đoán/lỗi khi evaluate.
7. `artifacts.py`/`inference.py`: release văn bản, kiểm tra checksum/schema, predict_texts.
8. `tests`: kiểm thử nhãn/trùng lặp, split tái lập, leakage từ vựng/selector, raw-text round-trip và output theo ID.
9. `reports`: EDA thật, bảng ứng viên, CV/test/external, luật cây và lỗi FP/FN.

CLI cần bổ sung, cú pháp mục tiêu (chưa chạy được):

```text
prepare-enron --input <raw.csv> --output <processed-dir>
build-tree-features --records <records.csv> --splits <splits.csv> --k 500 --output <feature-dir>
predict-text --release <release-dir> --text <email>
```

Thông tin tiền xử lý và đường dẫn của các lệnh mới nên có cấu hình riêng hoặc mở rộng config loader có kiểm thử. Không đưa tham số mới vào baseline.yaml rồi giả định loader cũ hiểu chúng.

## 12. Lịch thực hiện và đầu ra

### Ngày 1: Dữ liệu thật và báo cáo chất lượng

Hiệu tải file/ghi nguồn; cùng Doanh xác thực schema/nhãn, trùng lặp, xuất records. Dương chốt split manifest. Đầu ra nghiệm thu: dữ liệu sạch + báo cáo + ID chia tập.

### Ngày 2: Đặc trưng K=500 và lưu bộ biến đổi

Minh xây binary/Chi-square, Doanh kiểm tra archive bằng loader hiện có. Đầu ra: development NPZ, pipeline và feature names, kiểm thử không học từ validation.

### Ngày 3: Baseline thật và benchmark

Doanh chạy cây, xem luật và metrics train/validation; Dương kiểm tra so sánh và cách ghép ID. Đầu ra: run baseline tái lập, lỗi cài đặt đã được xử lý.

### Ngày 4: K=1.000 và khảo sát tham số

Minh cung cấp pipeline K=1.000; Doanh chạy 18 ứng viên, tổng hợp theo chính sách đã chốt. Đầu ra: bảng cấu hình/metrics, lựa chọn ứng viên cuối.

### Ngày 5: CV rút gọn và kiểm tra độ ổn định

Dương/Doanh dùng splitter chung, Minh bảo đảm fit lại đặc trưng trong fold. Đầu ra: mean/std và ghi rõ phạm vi CV đã thực hiện; khóa cấu hình trước test.

### Ngày 6: Test Enron và SpamAssassin

Hiệu bàn giao external đã kiểm tra trùng; Doanh/Dương đánh giá, xuất dự đoán và FP/FN. Đầu ra: bảng test/external riêng và phân tích lỗi.

### Ngày 7-8: Predictor văn bản, bàn giao và báo cáo

Doanh/Minh ghép pipeline; Hiệu thử demo; Dương nhận dự đoán validation để tích hợp ensemble. Doanh cập nhật phần báo cáo, README và nghiệm thu lưu/tải.

Đây là đợt 6-8 ngày làm việc, có thể cần lâu hơn nếu các module nhóm chưa có. Nằm trong tuần 2-3 của kế hoạch tổng; giữ tuần cuối cho báo cáo, tích hợp bốn mô hình và diễn tập. Sau ngày 3 chưa có baseline đúng thì tạm hoãn mở rộng K/CV để sửa luồng dữ liệu trước.

## 13. Điều kiện hoàn thành đợt train

- [ ] Dataset và phiên bản/checksum được lưu, không đưa raw hoặc mô hình lớn vào Git.
- [ ] Có thống kê làm sạch, nhãn mâu thuẫn, trùng lặp, lớp và số mẫu mỗi tập.
- [ ] Train/validation/test/external tách đúng, feature selector không học từ holdout.
- [ ] Có baseline, bảng tham số, cấu hình được chọn và benchmark đúng đầu vào.
- [ ] Có CV thực sự fit lại pipeline theo fold hoặc ghi rõ phần này chưa hoàn thành.
- [ ] Có kết quả test/external và dự đoán theo ID, phân tích FP/FN nếu có.
- [ ] Mô hình cùng bộ biến đổi tải lại và dự đoán email mới tương đương trước lưu.
- [ ] Dương nhận đủ đầu ra validation để ghép ensemble, Hiệu nhận predictor cho demo.
- [ ] Báo cáo thay số liệu giả lập bằng số liệu thật ở phần thực nghiệm email; giữ demo giả lập trong phần kiểm thử thuật toán.

**Thứ tự ưu tiên:** dữ liệu/split đúng → đặc trưng không rò rỉ → baseline → chọn cấu hình → khóa mô hình → test/external → predictor và bàn giao. Điểm số cao chỉ có ý nghĩa sau khi các điều kiện dữ liệu và đánh giá đã được kiểm tra.
