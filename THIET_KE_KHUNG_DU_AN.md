# Thiết kế khung dự án phát hiện email spam

**Nhóm:** Nguyễn Quốc Doanh (trưởng nhóm, Decision Tree), Lê Hữu Hiệu (Naive Bayes), Thân Đức Minh (Logistic Regression), Nguyễn Thái Dương (Linear SVM).

**Thời gian:** 4 tuần; lịch tạm tính 29/10-25/11/2026, dự phòng 26-28/11. Hạn chính thức cần cập nhật khi có thông báo môn học.

**Trạng thái:** Bản thiết kế để xây dựng khung. Các đường dẫn, interface, cấu hình và lệnh dưới đây là đề xuất triển khai; chưa có mã nguồn mô hình, dữ liệu huấn luyện hoặc kết quả thực nghiệm trong workspace. Kế hoạch phân công tổng thể nằm trong `KE_HOACH_NHOM.md`.

## 1. Mục tiêu thiết kế và phạm vi 1 tháng

Khung phải giúp bốn người tự viết mô hình song song, sau đó dùng cùng quy trình để huấn luyện, đánh giá, kết hợp bỏ phiếu và chạy demo. Người dùng nhập email tiếng Anh; hệ thống trả nhãn ham/spam của bốn mô hình và ensemble. Báo cáo giải thích được dữ liệu, thuật toán, sai số và kết quả thực nghiệm.

Đầu ra bắt buộc của khung:

1. Bộ đọc dữ liệu và schema thống nhất cho Enron, SpamAssassin.
2. Tiền xử lý, bộ biến đổi đặc trưng và danh sách mẫu chia tập tái lập được.
3. Interface chung cho bốn mô hình tự cài đặt.
4. Luồng huấn luyện, chọn cấu hình, đánh giá và bỏ phiếu ensemble.
5. Cách lưu/tải mô hình đi cùng bộ biến đổi, danh sách từ và thông tin lần chạy.
6. Giao diện demo dùng lại luồng dự đoán đã kiểm tra.
7. Cấu hình, kiểm thử, dữ liệu mẫu, hướng dẫn chạy và đầu ra báo cáo.

Chọn ứng dụng Python chạy cục bộ, một package, một giao diện Streamlit. Không cần cơ sở dữ liệu, tài khoản người dùng, REST backend riêng hoặc triển khai cloud cho phạm vi bài tập này. Random Forest, SMS, tiếng Việt, hiệu chuẩn xác suất SVM và tự viết toàn bộ TF-IDF/Chi-square để ở phần mở rộng. Bốn thuật toán mô hình và bỏ phiếu vẫn tự viết.

## 2. Kiến trúc và luồng dữ liệu

### 2.1. Luồng phát triển mô hình

```text
Enron thô
  → đọc dữ liệu + thống nhất nhãn/nguồn
  → kiểm tra dữ liệu + xử lý trùng lặp
  → lưu danh sách ID train / validation / test
  → tiền xử lý văn bản và đặc trưng thống kê
  → fit bộ biến đổi trên train
  → transform train và validation
  → học bốn mô hình tự viết
  → chọn tham số bằng CV/validation
  → lấy F1 validation để tính trọng số ensemble
  → khóa cấu hình và lưu release
  → đánh giá test + SpamAssassin độc lập
  → xuất bảng, biểu đồ, dự đoán sai và báo cáo
```

CV là các lần học riêng trên phần train: mỗi fold fit lại vocabulary, IDF, chọn đặc trưng và scaler bằng dữ liệu huấn luyện của fold đó. Không tạo ma trận đã chọn đặc trưng một lần trên toàn bộ dữ liệu rồi đem chia fold.

### 2.2. Luồng demo

```text
Nội dung email người dùng nhập
  → kiểm tra đầu vào
  → tải release đã lưu (một lần, cache)
  → cùng hàm tiền xử lý như khi học
  → transform bằng bộ biến đổi tương ứng của từng mô hình
  → dự đoán bốn mô hình
  → weighted hard voting
  → hiển thị nhãn, phiếu và luật cây nếu có
```

Demo không gọi `fit`. Ensemble nhận nhãn theo cùng thứ tự mẫu, không cộng trực tiếp trọng số LR với margin SVM hoặc xác suất NB.

### 2.3. Chiều phụ thuộc

- `models` chỉ biết NumPy/SciPy và interface dữ liệu đầu vào, không đọc CSV, không nhập Streamlit.
- `preprocessing` không biết nhãn để làm sạch một email; hàm đặc trưng thống kê chỉ dùng văn bản gốc.
- `features` phụ trách biến văn bản thành ma trận, che giấu vocabulary/IDF/selector/scaler sau interface nhỏ.
- `training` phối hợp dữ liệu, đặc trưng, mô hình và lưu kết quả; không chứa thuật toán Gini/gradient/Bayes.
- `evaluation` tính metrics và xuất dự đoán; không tự thay đổi tham số hoặc huấn luyện lại.
- `inference` là nơi duy nhất ghép release, tiền xử lý, đặc trưng và dự đoán cho ứng dụng.
- `app` chỉ gọi `inference`; notebook sử dụng lại package, không giữ bản sao thuật toán.

## 3. Cấu trúc thư mục dự kiến

```text
spam-detector/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── baseline.yaml
│   └── experiments.yaml
├── data/
│   ├── README.md
│   ├── raw/                       # dữ liệu tải về, giữ nguyên
│   ├── processed/                 # records.csv và kiểm tra chất lượng
│   └── splits/                    # ID và nhóm trùng lặp theo tập
├── src/
│   └── spam_detector/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── naive_bayes.py
│       │   ├── logistic_regression.py
│       │   ├── svm.py
│       │   └── decision_tree.py
│       ├── training.py
│       ├── evaluation.py
│       ├── ensemble.py
│       ├── artifacts.py
│       └── inference.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── fixtures/                  # email giả lập nhỏ, không có dữ liệu cá nhân
│   ├── test_data_features.py
│   ├── test_models.py
│   ├── test_ensemble.py
│   └── test_pipeline.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results.ipynb
├── runs/                          # mỗi lần chạy một thư mục riêng
├── releases/                      # bộ mô hình đã nghiệm thu cho demo
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── report.md
│   └── slides-outline.md
└── docs/
    ├── contracts.md
    ├── experiment-protocol.md
    └── decisions.md
```

Giữ cấu trúc này nhỏ: chưa tách thêm factory, repository, event bus hoặc nhiều lớp adapter. Với bốn mô hình cố định, ánh xạ tên mô hình sang class trong `training.py` là đủ. Chỉ tách `feature_selection.py` khi phần chọn đặc trưng thực sự lớn hoặc cần hai người chỉnh độc lập.

`tests/fixtures` được quản lý bằng Git. Dữ liệu đầy đủ, ma trận đặc trưng, mô hình và đầu ra lớn đặt trong `.gitignore`; lưu hướng dẫn tải, checksum nguồn và cấu hình để tái lập. `src` là mã nguồn chính; `reports` là hồ sơ học thuật; `runs` là đầu ra máy sinh.

## 4. Công nghệ và quản lý môi trường

- Python: cả nhóm chốt cùng một phiên bản minor trong buổi đầu, ghi trong `pyproject.toml`.
- NumPy/SciPy: phép toán và ma trận thưa; pandas: dữ liệu bảng.
- scikit-learn: chia tập, metrics, biểu diễn/chọn đặc trưng được phép; thuật toán mô hình dựng sẵn chỉ được gọi từ nhánh benchmark của thực nghiệm.
- BeautifulSoup và công cụ chuẩn hóa tiếng Anh cần thiết: xử lý HTML/token/stemming. Nếu dùng NLTK, hướng dẫn tải tài nguyên bằng bước setup rõ ràng; tránh tự tải ngầm mỗi lần demo.
- Matplotlib/Seaborn: biểu đồ; Streamlit: giao diện; pytest: kiểm thử; PyYAML: đọc cấu hình; joblib: lưu các bộ biến đổi thư viện nếu cần.
- Ghi phiên bản dependencies đã chạy thành công sau lần tích hợp đầu; không chọn các phiên bản cụ thể chỉ từ phỏng đoán. Có thể xuất `requirements-lock.txt` từ môi trường chuẩn đã nghiệm thu, đồng thời giữ `pyproject.toml` làm khai báo dependencies.

Cài đặt dự kiến bằng `python -m pip install -e ".[dev]"` sau khi package đã được tạo. Các thành viên dùng môi trường ảo riêng. Notebook và CLI phải import cùng package đã cài.

## 5. Hợp đồng dữ liệu

### 5.1. Schema records.csv

- `sample_id`: chuỗi định danh ổn định theo nguồn và ID gốc hoặc hash nội dung; duy nhất trong bộ dữ liệu.
- `text`: nội dung email đầu vào, còn đủ thông tin để trích thống kê trước làm sạch; tách MIME/HTML theo cùng chính sách cho cả hai nguồn.
- `label`: số nguyên 0=ham, 1=spam; ánh xạ từ dữ liệu gốc được ghi rõ.
- `source`: `enron` hoặc `spamassassin`.
- `group_id`: nhóm nội dung trùng hoặc gần trùng để tránh nằm ở hai tập khác nhau.

File chia tập lưu `sample_id`, `group_id`, `split`, `dataset_fingerprint`; không dựa vào vị trí dòng CSV vì thứ tự dòng có thể thay đổi. Khi loại dữ liệu phải xuất số lượng và lý do. Mẫu thiếu nhãn hoặc cùng nội dung nhưng nhãn mâu thuẫn phải được thống kê và giải quyết trước khi học.

### 5.2. Đọc email và kiểm tra chất lượng

Hiệu thống nhất quy tắc lấy Subject + phần văn bản của body; xử lý multipart/encoding và HTML. Dữ liệu CSV đã có văn bản chỉ cần adapter đọc đúng các cột, không giả định mọi chuỗi đều là file MIME.

Kiểm tra bắt buộc: nhãn hợp lệ, ID duy nhất, văn bản không null, số mẫu mỗi lớp, số bản trùng, phân bố độ dài, nguồn và encoding. Email rỗng sau chuẩn hóa cần có số liệu riêng. Dữ liệu lỗi nghiêm trọng làm lệnh dừng với thông báo; không âm thầm bỏ hàng rồi làm lệch dự đoán và nhãn.

### 5.3. Chia tập và kiểm tra ngoài nguồn

- Enron: train/validation/test = 60/20/20, seed cố định 42, bảo toàn nhóm trùng lặp; tỷ lệ lớp có thể chỉ xấp xỉ nếu phải giữ nhóm.
- Cố gắng cân bằng lớp mà không phá nhóm; ghi tỷ lệ thực tế trong manifest.
- SpamAssassin: giữ làm external test. Kiểm tra trùng với Enron theo cùng quy tắc; báo số bị loại hoặc nhóm trùng, không đưa external vào vocabulary/chọn tham số.
- Định nghĩa canonicalization để tìm bản trùng phải cố định, không học từ nhãn. Giai đoạn đầu xử lý trùng chính xác sau chuẩn hóa cơ bản; rà soát gần trùng là bước bổ sung phải mô tả giới hạn nếu chưa làm.

## 6. Tiền xử lý và đặc trưng

### 6.1. Tiền xử lý một email

Interface dự kiến: `preprocess(text, options) -> PreparedEmail`.

`PreparedEmail` chứa token để vector hóa và các thống kê được tính từ văn bản gốc. Thứ tự xử lý: lấy subject/body → ghi thống kê → bỏ HTML theo chính sách → chuẩn hóa chữ thường và token URL/EMAIL/MONEY/NUMBER → tách từ → stopword/stemming theo cấu hình. Các token đặc biệt phải được giữ nhất quán qua cả bước làm sạch và tokenizer của vectorizer; tránh token viết hoa bị loại hoặc đổi tên ở bước sau.

Thống kê đề xuất: số ký tự, số từ, số URL, số dấu chấm than, tỷ lệ chữ in hoa. Tính tỷ lệ chữ in hoa trên phần chữ cái gốc, xử lý mẫu không có chữ cái bằng quy tắc 0. Các quyết định token hóa/chuẩn hóa lưu cùng release.

### 6.2. Hai nhánh đặc trưng ban đầu

**Nhánh văn bản cho NB/LR/SVM:** BoW hoặc TF-IDF không âm, ma trận CSR float32. Chi-square dùng đầu vào không âm. Với baseline có thể dùng TF-IDF cho cả ba mô hình; ghi rõ NB đang dùng trọng số TF-IDF thay cho số đếm nguyên. E2 thêm BoW để so sánh.

**Nhánh cây:** sự xuất hiện từ 0/1, chọn K=500 trước, dense uint8, ngưỡng chia 0,5. Chi-square fit trên train; giữ danh sách từ theo đúng thứ tự cột. K không vượt số từ thực có sau lọc; nếu vocabulary rỗng thì báo lỗi cấu hình/dữ liệu.

Khi bổ sung thống kê, LR/SVM có thể nối các cột được scale bằng tham số học trên train; lưu scaler. Không trộn cột chuẩn hóa có giá trị âm vào NB. Cây bản nhị phân chưa xử lý các cột thống kê liên tục; không nhị phân hóa tất cả rồi tuyên bố đã thử cây trên TF-IDF liên tục.

### 6.3. Interface bộ biến đổi

```python
feature_pipeline.fit(prepared_train, y_train) -> self
feature_pipeline.transform(prepared_batch) -> X
feature_pipeline.get_feature_names() -> list[str]
```

Hợp đồng: `transform` không học lại; thứ tự cột và số cột luôn cố định; thứ tự hàng giữ nguyên; không truyền y vào transform. Các trạng thái học gồm vocabulary, IDF, selector và scaler đều nằm trong bộ biến đổi đã lưu. Mỗi fold có bộ biến đổi riêng. Các mô hình dùng pipeline khác nhau phải được ghi rõ ở bảng kết quả.

## 7. Interface mô hình chung

`models/base.py` định nghĩa quy ước tối thiểu; bốn class thực hiện cùng interface. Không bắt mọi mô hình triển khai xác suất chỉ để đủ method.

```python
model.fit(X, y) -> self
model.predict(X) -> ndarray  # shape (n_samples,), giá trị 0 hoặc 1
```

Các khả năng bổ sung thực sự cần:

- NB/LR/Tree: `predict_proba(X)` trả shape `(n_samples, 2)`, cột 0=ham, cột 1=spam.
- SVM: `decision_function(X)` trả shape `(n_samples,)`; score càng lớn càng nghiêng về spam.
- Tree: `export_rules(feature_names)` để trình bày đường đi và số mẫu/tỷ lệ lớp ở lá.
- LR/SVM: lịch sử loss lưu như thuộc tính để vẽ, không để notebook tính lại thuật toán học.

Các điều kiện chung:

1. X phải hai chiều, số hàng khớp y; y đầu vào luôn là 0/1. SVM đổi -1/+1 nội bộ và trả về 0/1.
2. Fit yêu cầu tập có cả hai lớp; nút con của Tree có thể chỉ có một lớp. Khi dữ liệu quá ít để chia fold phải báo rõ, không xuất điểm giả.
3. NB/LR/SVM hỗ trợ CSR ở baseline; Tree nhận dense uint8 đúng nhánh của mình. Không gọi `.toarray()` trên toàn bộ vocabulary chỉ để dùng chung một kiểu dữ liệu.
4. Dự đoán trước fit hoặc sai số cột phải báo lỗi rõ ràng. NaN/Inf không được đi qua phép học mà không kiểm tra.
5. Seed và quy tắc hòa phải xác định. Gợi ý NB/Tree hòa chọn ham, LR score 0 tương ứng p=0,5 chọn ham, SVM margin 0 chọn ham; ensemble dùng quy tắc riêng công bố bên dưới.
6. Thứ tự dự đoán phải khớp thứ tự mẫu. Hàm tính metrics nhận ID cùng dự đoán để phát hiện lệch hàng.

### 7.1. Phần việc mỗi thuật toán

- **Hiệu / NB:** prior lớp, tổng trọng số đặc trưng theo lớp, Laplace alpha, log likelihood, chuẩn hóa xác suất bằng cách ổn định số học.
- **Minh / LR:** trọng số/bias, sigmoid ổn định, cross-entropy, gradient L2, học suất, số epoch và dừng; kiểm tra loss chứ không chỉ nhãn cuối.
- **Dương / SVM:** hinge loss, điều chuẩn, mask vi phạm margin và subgradient; thống nhất hệ số loss/điều chuẩn với mô tả trong báo cáo.
- **Doanh / Tree:** Node, Gini có trọng số, chọn cột nhị phân, dựng cây, điều kiện dừng, nhãn/xác suất lá, xuất luật. Bỏ phép chia có nhánh rỗng, cải thiện không dương hoặc vi phạm số mẫu lá.

## 8. Module huấn luyện và chọn cấu hình

`training.py` cung cấp một điểm vào cho học mô hình; bên trong chọn class theo tên và gọi interface chung. Mỗi lần chạy phải biết rõ dataset, split, pipeline và tham số.

Trình tự một lần baseline:

1. Đọc/kiểm tra cấu hình và fingerprint dữ liệu; tạo thư mục run mới.
2. Lấy mẫu theo ID train/validation đã khóa.
3. Tiền xử lý theo cấu hình; fit pipeline trên train, transform train/validation.
4. Học từng mô hình bằng train; đo thời gian fit và predict riêng.
5. Xuất dự đoán validation theo ID, metrics và lỗi; lưu pipeline + mô hình.
6. Khi đủ bốn mô hình, Dương tính trọng số từ validation trên cùng danh sách mẫu.
7. Chọn cấu hình, ghi quyết định; đóng gói release xác định đúng bốn bundle và trọng số.

Trong lịch 1 tháng, giữ mô hình cuối học trên train, validation dùng chọn cấu hình/trọng số, test dùng đánh giá cuối. Không tự refit trên train+validation rồi giữ trọng số cũ mà coi như release đã được đánh giá tương đương; nếu cần refit sau này phải thiết kế lại quy trình trọng số bằng dự đoán ngoài fold.

CV chỉ trên train, 5 fold theo nhóm/phân tầng khi khả thi, ghi fold thực tế. Dùng số cấu hình nhỏ trước; cấu hình không hợp lệ hoặc quá thời gian được ghi trạng thái thất bại/bỏ qua, không ghi F1=0 thay cho kết quả chưa chạy.

## 9. Ensemble

`WeightedHardVoting` cần ít interface:

```python
ensemble.fit_weights(validation_predictions, y_validation)
ensemble.predict(predictions) -> labels
ensemble.vote_score(predictions) -> scores
```

`predictions` là ma trận `(n_samples, 4)` theo danh sách tên mô hình cố định `[nb, lr, svm, tree]`. ID mẫu phải được đối chiếu trước khi ghép cột. Lưu cả thứ tự tên và trọng số.

Trọng số `w_k = F1_k / sum(F1)` từ validation; nếu tổng bằng 0 thì bốn trọng số bằng nhau. Điểm là tổng có trọng số của nhãn 0/1; điểm >=0,5 trả spam theo quy tắc trong kế hoạch gốc. Phải kiểm tra số cột, giá trị nhãn, tổng trọng số và giá trị hữu hạn. Khi so sánh với thư viện hoặc tái hiện, dùng cùng quy tắc hòa.

Điểm bỏ phiếu không phải xác suất spam được hiệu chuẩn. Demo ghi “điểm bỏ phiếu”, không ghi “độ tin cậy 95%”. Nếu thiếu một mô hình trong release thì báo không đầy đủ, không tự đổi trọng số và hiển thị như ensemble chuẩn.

## 10. Đánh giá và giao thức thực nghiệm

`evaluation.py` nhận y thật, nhãn dự đoán và tùy chọn score; trả metrics có cấu trúc để CLI, notebook và báo cáo dùng chung.

Các trường tối thiểu: số mẫu, số ham/spam, Accuracy, Precision spam, Recall spam, F1 spam, TN/FP/FN/TP. Quy định ma trận theo thứ tự lớp `[0, 1]`, lớp dương=spam. Nếu mẫu số metric bằng 0, ghi chính sách tính 0 và số lượng dự đoán mỗi lớp để người đọc hiểu. ROC/PR chỉ xuất khi có score phù hợp và đủ hai lớp; không giả định mọi mô hình có xác suất.

Tách các loại kết quả `cv`, `validation`, `test`, `external_test`. Không tính trung bình lẫn chúng. CV báo mean/std; test/external báo số liệu đúng tập đánh giá. Thời gian phải nêu rõ có bao gồm transform hay không; đề xuất lưu `fit_seconds`, `transform_seconds`, `predict_seconds` riêng.

### 10.1. Các nhóm thí nghiệm

- **E1:** Bốn mô hình, cùng fold, Precision/Recall/F1 và thời gian; baseline thống nhất và ghi rõ nhánh cây.
- **E2:** BoW, TF-IDF, TF-IDF+thống kê trên mô hình hỗ trợ. Cây nhị phân là biến thể riêng; không ép chạy thí nghiệm không đúng biểu diễn.
- **E3:** Thay đổi từng yếu tố stopword/stemming/lemmatization ở quy mô giới hạn, giữ những yếu tố còn lại cố định.
- **E4:** Học Enron, kiểm tra SpamAssassin bằng pipeline Enron đã lưu.
- **E5:** K=500/1.000/3.000/toàn bộ từ; bắt đầu từ K nhỏ, ghi trường hợp không chạy vì tài nguyên.
- **E6:** Ensemble và từng mô hình, cùng mẫu đánh giá; trọng số lấy từ validation.
- **E7:** Mô hình tự viết và sklearn cùng X, nhãn, split và cấu hình gần tương đương; phần gọi mô hình sklearn chỉ nằm trong nhánh benchmark thực nghiệm.

Ưu tiên triển khai E1/E6/E7, rồi E4, sau đó mở rộng E2/E3/E5. Đây là thứ tự làm việc; nếu giảng viên yêu cầu đầy đủ E1-E7, phải bố trí đủ hoặc báo rõ phần chưa đạt. Không gọi một kết quả gần sklearn là bằng chứng đầy đủ rằng thuật toán đúng; vẫn cần ví dụ tính tay và kiểm thử hành vi.

### 10.2. Kết quả để truy vết

`metrics.csv` nên có `run_id, experiment_id, model, split, fold, config_id, n_samples, precision_spam, recall_spam, f1_spam, fp, fn, fit_seconds, predict_seconds` và các metrics còn lại.

`predictions.csv` có `sample_id, model, y_true, y_pred, score, score_kind`; `score_kind` phân biệt probability, margin hoặc vote. Nội dung email sai chỉ đưa vào file phân tích cục bộ nếu cần, hạn chế chép hàng loạt vào log và báo cáo.

Mỗi bảng/hình phải chỉ về run/config và tập đánh giá. Test chỉ mở sau khi chốt cấu hình; nếu sửa lỗi cài đặt sau đánh giá phải ghi lại và chạy lại phần liên quan, không lặng lẽ chọn kết quả test cao nhất.

## 11. Lưu mô hình, lần chạy và release

```text
runs/<run_id>/
  config.resolved.yaml
  manifest.json
  metrics.csv
  predictions.validation.csv
  figures/
  bundles/<model>/
    model_state.npz hoặc model_state.json
    feature_pipeline.joblib
    metadata.json

releases/<release_id>/
  release.json
  ensemble.json
  bundles/nb, lr, svm, tree/...
  evaluation/metrics.test.csv
  evaluation/metrics.external.csv
```

`manifest` lưu phiên bản Python/dependencies, seed, commit nếu có, fingerprint dữ liệu và split, thời gian chạy, cấu hình đã giải quyết đầy đủ. `metadata` lưu tên/lớp mô hình, thứ tự lớp, loại/shape đầu vào, tên đặc trưng hoặc checksum của danh sách đó, phiên bản format và cấu hình tiền xử lý. Tree lưu cấu trúc nút; LR/SVM lưu trọng số/bias; NB lưu prior/log likelihood và các thông tin cần dự đoán. Có thể chọn một cơ chế serializer đơn giản hơn cho toàn bộ bundle khi triển khai, nhưng phải kiểm tra round-trip và giữ version format.

Đặt tên run duy nhất; không ghi đè mô hình của cấu hình khác. `release.json` chỉ ra đúng bundle nào đã dùng tính ensemble và đánh giá. Demo nhận đường dẫn release cụ thể qua cấu hình, không tự lấy thư mục mới nhất theo tên.

Nếu cache đặc trưng, cache key phải gồm fingerprint dữ liệu/split, fold, tiền xử lý và tham số đặc trưng. Giai đoạn đầu có thể bỏ cache để tránh dùng nhầm ma trận; tối ưu sau khi đo thời gian thực tế.

## 12. Inference và thiết kế demo

`load_predictor(release_dir)` tải và kiểm tra bundle; đối tượng trả về cung cấp `predict_texts(texts)` để ứng dụng chỉ cần gọi một lần.

Kết quả mỗi email gồm nhãn từng mô hình, score có loại rõ ràng, nhãn ensemble, điểm bỏ phiếu và tùy chọn đường đi cây. Không trả cả vector TF-IDF hoặc hàng nghìn trọng số cho ứng dụng.

Một màn hình demo đủ dùng:

1. Tiêu đề và mô tả phạm vi email tiếng Anh.
2. Ô nhập email, nút phân loại, vài ví dụ cố định.
3. Kết luận ensemble và bảng nhãn bốn mô hình.
4. Mục mở rộng xem luật cây và trọng số bỏ phiếu.
5. Tên release/cấu hình rút gọn trong phần thông tin thực nghiệm.

Đầu vào trắng bị từ chối trước dự đoán. Nếu email hợp lệ nhưng toàn bộ từ ngoài vocabulary, vẫn giữ số dòng và trả dự đoán theo trạng thái mô hình; có thông báo ít tín hiệu từ vựng. Với nhiều nhánh đặc trưng, kiểm tra trạng thái này theo từng pipeline. Lỗi chưa có release, sai phiên bản, thiếu mô hình hoặc sai số cột phải được hiển thị dễ hiểu; log kỹ thuật lưu để nhóm sửa.

## 13. CLI và cấu hình dự kiến

Các lệnh sau chỉ là giao diện mục tiêu sau khi khung được triển khai:

```powershell
python -m spam_detector.cli prepare --config configs/baseline.yaml
python -m spam_detector.cli train --config configs/baseline.yaml --model all
python -m spam_detector.cli experiment --config configs/experiments.yaml --id E1
python -m spam_detector.cli release --run-dir runs/<run_id>
python -m spam_detector.cli evaluate --release-dir releases/<release_id> --split test
python -m spam_detector.cli predict --release-dir releases/<release_id> --text "Team meeting tomorrow"
python -m streamlit run app/streamlit_app.py
python -m pytest
```

`prepare` tạo records và split; `train` chỉ dùng train/validation; `release` yêu cầu đủ bundle và dự đoán validation để tính trọng số; `evaluate` dùng release đã khóa. `experiment` ghi lại từng lần học CV và benchmark. Mỗi lệnh kiểm tra điều kiện đầu vào và trả mã lỗi khi thất bại.

Cấu hình baseline minh họa, không phải tham số đã tối ưu:

```yaml
seed: 42
data:
  enron_path: data/raw/enron
  external_path: data/raw/spamassassin
  split_ratios: [0.60, 0.20, 0.20]
preprocessing:
  lowercase: true
  replace_special_tokens: true
  remove_stopwords: true
  stemming: porter
features:
  text:
    representation: tfidf
    select_k: 1000
    add_statistics: false
  tree:
    representation: binary
    select_k: 500
models:
  nb: {alpha: 1.0}
  lr: {learning_rate: 0.1, epochs: 200, l2: 0.001}
  svm: {learning_rate: 0.01, epochs: 200, l2: 0.001}
  tree: {max_depth: 10, min_samples_split: 10, min_samples_leaf: 5}
ensemble:
  weighting: validation_f1
  threshold: 0.5
  tie_label: 1
evaluation:
  cv_folds: 5
  positive_label: 1
```

`config.py` kiểm tra tỷ lệ cộng bằng 1, K/số epoch dương, learning rate hợp lệ, tên mô hình đúng và điều kiện lá của cây. Không đưa đường dẫn tuyệt đối máy Doanh vào cấu hình dùng chung. Các giá trị LR/SVM cần thử trên validation; không cam kết loss giảm chỉ vì dùng cấu hình minh họa.

## 14. Kiểm thử có giá trị trước khi tích hợp

### Dữ liệu và đặc trưng

- Bộ email nhỏ có URL/HTML/viết hoa, bản trùng, null và nhãn mâu thuẫn: kiểm tra báo cáo chất lượng và ID chia tập.
- Từ chỉ xuất hiện trong validation không được nằm trong vocabulary học từ train.
- Từ xuất hiện nhiều ở test không được làm đổi IDF/Chi-square; transform trước và sau dự đoán giữ nguyên state.
- Trộn lại thứ tự dòng records không làm đổi thành viên mỗi split hoặc ghép sai nhãn.
- Tên cột và shape đúng khi fit/transform/lưu/tải; token đặc biệt không mất qua vectorizer.

### Mô hình

- NB: ví dụ đếm nhỏ có kết quả tính tay, gồm từ không gặp ở một lớp và làm trơn.
- LR: gradient trên dữ liệu nhỏ đối chiếu sai phân hữu hạn; dự đoán đúng chiều điểm số.
- SVM: mẫu nằm ngoài margin và trong margin tạo cập nhật phù hợp, bias không bị điều chuẩn nhầm nếu công thức chọn không điều chuẩn bias.
- Tree: ví dụ có phép chia tốt nhất tính tay, lá thuần, cột hằng, nhánh không hợp lệ, giới hạn độ sâu và số mẫu tối thiểu.
- Mỗi mô hình phát hiện predict trước fit, sai chiều và đầu vào không hữu hạn.

### Ensemble và tích hợp

- Cùng bộ nhãn nhưng đảo thứ tự ID phải bị phát hiện khi ghép phiếu.
- Trường hợp hòa 2-2, trọng số khác nhau và toàn bộ F1=0 có đầu ra đúng quy tắc.
- Bộ dữ liệu giả lập nhỏ chạy qua prepare → train → release → predict; kiểm tra số hàng, nhãn và bundle đầy đủ.
- Dự đoán trước và sau lưu/tải tương đương, score sai số trong tolerance đã định.
- CLI và demo gọi cùng predictor phải cho cùng kết quả trên cùng email và release.
- Thử tải release thiếu bundle hoặc sai số đặc trưng: lỗi rõ ràng, không trả kết quả một phần như bản hoàn chỉnh.

Không cần viết kiểm thử cho mọi dòng cấu hình hoặc giao diện tĩnh. Tập trung các lỗi có thể làm sai thí nghiệm hoặc làm hỏng tích hợp.

## 15. Phân công file và người duyệt

- **Doanh sở hữu:** khung package, `models/base.py`, `decision_tree.py`, `config.py`, hợp đồng trong `docs`, rà soát `training.py`/release và đóng gói. Minh duyệt interface và phần cây.
- **Hiệu sở hữu:** `data.py`, `preprocessing.py`, `naive_bayes.py`, EDA, hướng dẫn nguồn và `app/streamlit_app.py`. Dương duyệt dữ liệu/tập chia và luồng demo.
- **Minh sở hữu:** `features.py`, `logistic_regression.py`, lưu state bộ biến đổi; phối hợp Doanh xây `artifacts.py`. Doanh duyệt thứ tự đặc trưng và nhánh Tree.
- **Dương sở hữu:** `svm.py`, `evaluation.py`, `ensemble.py`, khung thí nghiệm trong CLI/training; phối hợp Hiệu xây `inference.py`. Hiệu duyệt ghép mẫu và đánh giá ngoài nguồn.
- **File dùng chung `training.py`, `cli.py`, `artifacts.py`, `inference.py`:** ghi người đang chỉnh trên bảng việc; một người chủ trì mỗi đầu việc, người khác gửi thay đổi nhỏ qua review. Không để cả bốn người cùng sửa entry point.
- **Mỗi người:** kiểm thử mô hình mình, phần báo cáo, hình minh họa, cấu hình và phân tích lỗi; không giao toàn bộ kiểm thử cho Dương hoặc toàn bộ báo cáo cho Doanh.

## 16. Thứ tự xây khung trong 5 ngày đầu

### Ngày 1: Chốt hợp đồng, cài được package

Doanh tạo repo/package, pyproject, cấu hình ban đầu, README và interface mô hình. Cả nhóm chốt tên nhãn, shape, schema, cách báo lỗi. Hiệu cung cấp fixtures email; Minh chốt kiểu ma trận hai nhánh; Dương chốt danh sách đầu ra đánh giá. Mỗi người kiểm tra import được package trên máy mình.

**Nghiệm thu:** Cài editable thành công, đọc được cấu hình, fixtures hợp lệ và hợp đồng được ghi vào docs. Chưa có mô hình thật thì entry point phải báo chưa triển khai, không trả dự đoán giả.

### Ngày 2: Có dữ liệu và tập chia dùng chung

Hiệu hoàn thành reader/kiểm tra cơ bản; Dương làm split manifest; Doanh và Minh rà schema. Bốn người bắt đầu thuật toán bằng dữ liệu số nhỏ do mình kiểm soát, không chờ dữ liệu đầy đủ.

**Nghiệm thu:** Có records, thống kê dữ liệu, split không trùng ID/nhóm, đủ lớp cho quy trình dự kiến.

### Ngày 3: Có đường đi đặc trưng

Minh hoàn thành fit/transform cho TF-IDF và binary K=500; Hiệu chốt preprocessing; Doanh nhận X binary/y; Dương nối metrics với nhãn dự đoán mẫu trong kiểm thử.

**Nghiệm thu:** X train/validation đúng shape và kiểu; test chứng minh vocabulary không học từ validation.

### Ngày 4: Có một luồng học thật xuyên suốt

Hiệu bàn giao NB bản nhỏ; Dương ghép vào training và xuất metrics validation; Minh/Doanh thử fit mô hình mình qua cùng interface. Đây là đường chạy đầu tiên dùng mô hình thật từ văn bản đến kết quả lưu được.

**Nghiệm thu:** Có một run thực sự, cấu hình resolved, ID dự đoán và bundle có thể tải lại.

### Ngày 5: Dùng luồng đã chạy để mở cho cả nhóm

Doanh rà entry point/interface và phân nhiệm sửa lỗi; mỗi thành viên thay mô hình mình vào mà không sửa data/features/evaluation. Hiệu tạo khung demo gọi predictor; Dương kiểm tra ghép phiếu bằng fixtures; Minh hoàn thiện metadata đặc trưng.

**Nghiệm thu:** Khung có đường tích hợp rõ; phần chưa hoàn thành được liệt kê theo người và mốc ngày 7, không bị che bằng kết quả mẫu.

## 17. Mốc nghiệm thu trong 4 tuần

- **04/11:** Khung package, dữ liệu, split, đặc trưng, một luồng học thật và bốn mô hình chạy trên bài toán nhỏ.
- **11/11:** Bốn mô hình chạy dữ liệu thật, kiểm thử chính, metrics validation, lưu/tải bundle và demo đầu tiên.
- **18/11:** Cấu hình chốt, ensemble, thực nghiệm, kết quả test/external và phân tích lỗi; dừng mở rộng tính năng.
- **21/11:** Mỗi người giao báo cáo/slide/hình/số liệu của mình.
- **25/11:** Bản nộp tái lập được từ hướng dẫn; diễn tập xong, video dự phòng và file nộp đầy đủ.

Nếu chậm, giảm số cấu hình và tính năng mở rộng trước. Không giảm kiểm tra rò rỉ, không thay mô hình tự viết bằng sklearn và không điền kết quả chưa chạy. Thực nghiệm bắt buộc của giảng viên cần được giữ hoặc ghi rõ phần chưa đạt.

## 18. Checklist hoàn thành khung

- [ ] Cả bốn thành viên cài và import được cùng package.
- [ ] Có schema, quy ước nhãn, split manifest và fixtures.
- [ ] Có interface fit/predict, scores theo khả năng, shape và lỗi rõ ràng.
- [ ] Có hai nhánh đặc trưng, không rò rỉ và đúng thứ tự cột.
- [ ] Có một luồng chạy thật từ email đến metrics và bundle.
- [ ] Thêm mô hình thứ hai không phải chép lại tiền xử lý hoặc evaluation.
- [ ] Lưu/tải và CLI/demo dự đoán tương đương.
- [ ] Ensemble ghép theo ID và thứ tự tên mô hình cố định.
- [ ] Mỗi run truy được cấu hình, dữ liệu, split và metrics.
- [ ] Mỗi module có người phụ trách, người duyệt và hạn bàn giao.
- [ ] Mỗi phần chưa triển khai báo trạng thái rõ ràng; chưa có số liệu thì không hiển thị như kết quả thật.

**Quyết định nên chốt ngay:** giữ baseline đơn giản (TF-IDF không âm cho NB/LR/SVM, binary K=500 cho Tree), đóng băng interface trong ngày đầu, có một luồng học thật trước khi thêm nhiều tính năng. Đây là cơ sở để Doanh quản lý tiến độ bằng sản phẩm chạy được.
