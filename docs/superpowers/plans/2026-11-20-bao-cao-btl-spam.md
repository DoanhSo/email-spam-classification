# Báo cáo BTL phát hiện email spam (NB chính + DT so sánh) – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thành báo cáo Word cho BTL Khai phá dữ liệu "Hệ thống phát hiện email spam/rác"
theo khung báo cáo mẫu. Kèm theo là phần code để tạo số liệu: xuất ARFF cho Weka, Multinomial
Naive Bayes tự viết, đối chiếu với Weka, demo Streamlit.

**Architecture:** Weka (người dùng tự thao tác) định nghĩa dữ liệu sạch, tập 70/30, kết quả
NB/J48 và kiểm định. Python trong `src/spam_detector` làm ba việc: đọc/ghi ARFF, chứng minh NB
tự viết khớp Weka trên cùng vector, tạo release NB/DT cho demo. Báo cáo được viết bằng Markdown
theo từng chương, sau đó dựng thành `.docx` bằng pandoc (qua `pypandoc_binary`) với reference
doc và hậu xử lý bằng python-docx để chèn bìa.

**Tech Stack:** Python ≥3.11, NumPy, liac-arff, scikit-learn (chỉ để vector hóa trong pipeline
đã có), Streamlit, matplotlib, pypandoc_binary, python-docx, Weka 3.8.7.

**Spec:** `docs/superpowers/specs/2026-11-20-bao-cao-btl-spam-design.md`

## Global Constraints

- Báo cáo bằng tiếng Việt, định dạng Word (.docx). Đề mục theo đúng mục 5 của spec.
- Bìa: TRƯỜNG ĐẠI HỌC THỦY LỢI / KHOA CÔNG NGHỆ THÔNG TIN / KHAI PHÁ DỮ LIỆU /
  Đề tài: "Hệ thống phát hiện email spam/rác" / Giảng viên: Trần Mạnh Tuấn / Nhóm 12 – lớp
  65CNTT / 1. Nguyễn Quốc Doanh – 2351060429, 2. Nguyễn Thái Dương – 2351060435,
  3. Lê Hữu Hiệu – 2351060446, 4. Thân Đức Minh – 2351060467 / HÀ NỘI, 2026.
- Hạn nộp 30/11/2026. Bản hoàn chỉnh trước 28/11.
- Nhãn: `ham` = 0, `spam` = 1. Lớp dương là spam. ARFF luôn khai báo `class {ham,spam}`.
- Seed 42 ở mọi chỗ cần ngẫu nhiên (`Randomize -S 42`, CV).
- Mô hình tự viết chỉ dùng NumPy. sklearn không được dùng để huấn luyện NB/DT.
- Weka 3.8.7 bản `bellsoft-x64-windows`. Mọi thao tác GUI do người dùng làm theo
  `reports/weka/CHECKLIST.md`.
- Không được rò rỉ dữ liệu: từ vựng và lựa chọn thuộc tính chỉ học trên train
  (`FilteredClassifier` hoặc batch filtering). Experimenter chỉ chạy trên `train.arff`.
- Không trích dẫn bài báo nào chưa tra được DOI hoặc trang chính thức.
- Không sửa hay xóa `KE_HOACH_NHOM.md`, `THIET_KE_KHUNG_DU_AN.md`, `artifacts/tree-enron-v1`.

## Review Focus

1. **Chuỗi trong ARFF do Weka lưu** (dấu `'` được escape, `\n`, `\\`, ký tự non-ASCII) phải đọc
   lại đúng nguyên văn bản. Test nằm ở Task 1 (`test_read_weka_escaped_strings`).
2. **Email không có từ nào thuộc từ vựng NB, hoặc email rỗng** phải hiện "Không dự đoán được"
   thay vì trả về nhãn theo prior. Test ở Task 7 (`test_nb_unpredictable_when_no_vocabulary`).
3. **Email rất dài** (gần 100.000 ký tự, số đếm lớn) không được làm xác suất NB bị underflow
   hoặc NaN. Test ở Task 2 (`test_long_document_is_finite`).
4. **ARFF vector của Weka khai báo lớp theo thứ tự `{spam,ham}` hoặc thuộc tính lớp không nằm
   cuối** vẫn phải ánh xạ nhãn theo tên. Test ở Task 1 (`test_vector_arff_class_by_name`).
5. **File CSV dự đoán của Weka** có cột error là `+`, có dòng trống ở cuối, nhãn dạng `2:spam`
   phải parse được. Test ở Task 4 (`test_read_weka_predictions_variants`).

## Lịch đề xuất

| Ngày | Việc |
|---|---|
| 21/11 | Task 1–3. Người dùng cài Weka. |
| 22/11 | Người dùng chạy CHECKLIST trên Weka. Song song làm Task 4–5 và Task 8. |
| 23/11 | Task 6–7, Task 9 (tổng quan tài liệu). |
| 24–26/11 | Task 10–11 (viết chương I–IV, Kết luận). |
| 27/11 | Task 12 (dựng bản cuối, rà soát theo mẫu). Nhóm đọc duyệt. |
| 28–30/11 | Dự phòng, sửa theo góp ý, nộp. |

## File Structure

| File | Trách nhiệm |
|---|---|
| `src/spam_detector/weka_io.py` (mới) | Ghi ARFF Enron, đọc ARFF văn bản và ARFF vector (dense/sparse) |
| `src/spam_detector/models/naive_bayes.py` (mới) | Multinomial NB tự viết, cùng cách làm trơn với Weka |
| `src/spam_detector/weka_compare.py` (mới) | Đọc CSV dự đoán của Weka, so NB Python với NB Weka |
| `src/spam_detector/nb_release.py` (mới) | Lưu/nạp release NB, `NBTextPredictor` cho demo |
| `src/spam_detector/report_experiments.py` (mới) | Train NB/DT Python trên 70/30 của Weka, đánh giá test và SpamAssassin, xuất hình |
| `src/spam_detector/cli.py` (sửa) | Thêm lệnh `export-arff`, `compare-weka`, `report-run` |
| `src/spam_detector/web_service.py`, `app/streamlit_app.py`, `configs/web.yaml` (sửa) | Adapter `nb`, NB mặc định, giải thích top từ, quy tắc "Không dự đoán được" |
| `reports/weka/CHECKLIST.md` (mới) | Hướng dẫn thao tác Weka từng bước, đánh số hình và đường dẫn lưu file |
| `reports/bao-cao/build_docx.py`, `reports/bao-cao/content/*.md` (mới) | Nội dung báo cáo và script dựng .docx |
| `pyproject.toml` (sửa) | Thêm `liac-arff>=2.5` vào dependencies, extra `report = ["pypandoc_binary>=1.13", "python-docx>=1.1", "matplotlib>=3.8"]` |

---

### Task 1: Đọc và ghi ARFF

**Files:**
- Create: `src/spam_detector/weka_io.py`
- Modify: `pyproject.toml` (thêm `liac-arff>=2.5`)
- Test: `tests/test_weka_io.py`

**Interfaces:**
- Produces:
  - `write_enron_arff(csv_path: str | Path, out_path: str | Path) -> int`: trả về số dòng đã ghi.
    Relation `enron_spam`. Các thuộc tính theo thứ tự: `row_index NUMERIC` (cột đầu không tên
    của CSV), `label_num NUMERIC`, `text STRING`, `class {ham,spam}` (lấy từ cột `label`).
  - `read_text_arff(path) -> list[dict]`: mỗi phần tử là `{"text": str, "label": int}`. Hàm
    tìm thuộc tính `text` và `class` theo tên, bỏ qua các thuộc tính khác.
  - `read_vector_arff(path) -> tuple[np.ndarray, np.ndarray, list[str]]`: trả về `X`
    (float64, dense, theo thứ tự cột), `y` (int64, ánh xạ theo tên lớp: ham=0, spam=1) và tên
    các thuộc tính đặc trưng. Hỗ trợ cả ARFF sparse. Báo `ValueError` nếu thiếu thuộc tính
    `class`, lớp không phải {ham, spam}, hoặc có thuộc tính đặc trưng không phải số.

- [ ] **Step 1: Viết các test lỗi**
  - `test_enron_roundtrip`: CSV 3 dòng có dấu nháy, xuống dòng, `\`, ký tự `é`. Ghi xong thì
    `write_enron_arff(...) == 3`, và `read_text_arff` trả lại đúng nguyên văn cùng nhãn `[0, 1, 0]`.
  - `test_read_weka_escaped_strings`: fixture viết tay theo kiểu Weka, có dòng
    `'it\'s a \\n test\nline2',spam`. Kết quả đọc được là `"it's a \\n test\nline2"` với label 1.
    Phải đối chiếu với một file Weka lưu thật trong lần chạy CHECKLIST, và cập nhật fixture nếu khác.
  - `test_vector_arff_sparse`: ARFF sparse 2 dòng `{0 2,2 1,3 spam}` và `{1 3,3 ham}`, có 3
    thuộc tính số và `class` nằm cuối. Kết quả là `X == [[2,0,1],[0,3,0]]` và `y == [1,0]`.
  - `test_vector_arff_class_by_name`: `class {spam,ham}` đặt ở cột đầu. Nhãn vẫn đúng ham=0, spam=1.
  - `test_vector_arff_rejects_missing_class`: thiếu `class` thì báo `ValueError`.
- [ ] **Step 2:** Chạy `.venv\Scripts\python -m pytest tests/test_weka_io.py -v`. Kết quả mong đợi: FAIL (chưa có module).
- [ ] **Step 3:** `pip install -e ".[dev]"` sau khi thêm dependency, rồi cài đặt ba hàm trên bằng `arff.load`/`arff.dumps` của liac-arff. Đọc CSV bằng module `csv` với `field_size_limit` lớn.
- [ ] **Step 4:** Chạy lại test. Kết quả mong đợi: PASS. Sau đó chạy toàn bộ `pytest -q` và không có test cũ nào hỏng.
- [ ] **Step 5:** Commit `feat: ARFF export/import for Weka workflow`.

### Task 2: Multinomial Naive Bayes tự viết

**Files:**
- Create: `src/spam_detector/models/naive_bayes.py`
- Test: `tests/test_naive_bayes.py`

**Interfaces:**
- Consumes: `NotFittedError` từ `models/base.py`.
- Produces: class `MultinomialNaiveBayes(alpha: float = 1.0)`, gồm:
  - `fit(X, y) -> Self`, trong đó X là ma trận đếm không âm và y phải có đủ cả lớp 0 và 1.
  - `predict(X) -> NDArray[int64]` và `predict_proba(X) -> NDArray[float64]` (cột 0 là ham,
    cột 1 là spam).
  - `explain(x_row, feature_names, n=5) -> list[tuple[str, float]]`: liệt kê các từ có mặt
    trong email, xếp theo đóng góp `count * (log P(w|spam) - log P(w|ham))` giảm dần.
  - `to_dict() -> dict` và `from_dict(state) -> MultinomialNaiveBayes`.
  - Thuộc tính `class_log_prior_` có shape (2,), `feature_log_prob_` có shape (2, n_features).
- Công thức phải khớp `NaiveBayesMultinomial` của Weka:
  `P(c) = (N_c + 1) / (N + 2)` và `P(w|c) = (count_wc + alpha) / (Σ_w count_wc + alpha·V)`.
  Tính toán trong không gian log, chuẩn hóa bằng log-sum-exp, không có hệ số multinomial.

- [ ] **Step 1: Viết các test lỗi** với `X = [[2,1,0],[0,1,3],[1,0,0]]`, `y = [0,1,0]`:
  - `test_hand_computed_parameters`: `exp(class_log_prior_) ≈ [3/5, 2/5]`,
    `exp(feature_log_prob_[0]) ≈ [4/7, 2/7, 1/7]`, `exp(feature_log_prob_[1]) ≈ [1/7, 2/7, 4/7]`.
  - `test_hand_computed_proba`: `predict_proba([[1,0,1]]) ≈ [[0.6, 0.4]]` và
    `predict_proba([[0,0,2]])[0,1] ≈ 6.4/7.0`.
  - `test_long_document_is_finite`: với `[[100000,0,100000]]`, kết quả hữu hạn và tổng mỗi dòng bằng 1.
  - `test_rejects_negative_and_single_class`: báo `ValueError`.
  - `test_not_fitted`: gọi `predict` trước `fit` thì báo `NotFittedError`.
  - `test_dict_roundtrip`: dự đoán của mô hình nạp từ `from_dict(to_dict())` giống hệt bản gốc.
  - `test_explain_orders_by_spam_contribution`: `explain([0,1,3], ["a","b","c"])[0][0] == "c"`.
    Từ `"a"` vắng mặt trong email nên không có trong kết quả.
- [ ] **Step 2:** Chạy `pytest tests/test_naive_bayes.py -v`. Kết quả mong đợi: FAIL.
- [ ] **Step 3:** Cài đặt class trong `naive_bayes.py` theo phong cách validate của `decision_tree.py`.
- [ ] **Step 4:** Chạy lại test. Kết quả mong đợi: PASS.
- [ ] **Step 5:** Commit `feat: self-written multinomial naive bayes matching Weka smoothing`.

### Task 3: Lệnh `export-arff` và CHECKLIST Weka

**Files:**
- Modify: `src/spam_detector/cli.py`
- Create: `reports/weka/CHECKLIST.md`, `reports/weka/data/enron_raw.arff` (file sinh ra, có commit vì nhỏ hơn 10MB; nếu lớn hơn thì thêm vào `.gitignore` và ghi lệnh tạo lại)
- Test: `tests/test_weka_io.py::test_cli_export_arff`

**Interfaces:**
- Consumes: `write_enron_arff` (Task 1).
- Produces: lệnh `spam-detector export-arff --csv <path> --output <path>`, in ra `wrote N rows`.
  Ngoài ra là các đường dẫn file mà các task sau dùng (CHECKLIST yêu cầu người dùng lưu đúng tên):
  `reports/weka/data/{enron_dedup,train,test,train_vec,test_vec}.arff`,
  `reports/weka/results/{nb,j48}_{training,cv,test}.txt` (toàn văn Classifier output),
  `reports/weka/results/nb_test_predictions.csv`, `reports/weka/results/experimenter.txt`,
  và ảnh `reports/weka/figures/hinh-XX.png`.

- [ ] **Step 1:** Viết test CLI: chạy trên CSV fixture, kiểm tra exit code 0 và file ARFF đọc lại được 3 dòng.
- [ ] **Step 2:** Chạy test. Kết quả mong đợi: FAIL.
- [ ] **Step 3:** Thêm subcommand vào `cli.py` theo mẫu các lệnh hiện có.
- [ ] **Step 4:** Chạy test. Kết quả mong đợi: PASS. Sau đó chạy lệnh thật
  `spam-detector export-arff --csv data/raw/enron/spam_ham_dataset.csv --output reports/weka/data/enron_raw.arff`,
  kết quả mong đợi `wrote 5171 rows`.
- [ ] **Step 5:** Viết `CHECKLIST.md`. Mỗi bước ghi số hình, đường dẫn menu Weka, tham số chính xác
  và file cần lưu. Thứ tự hình bám theo mục II–III của spec:
  1. Mở `enron_raw.arff` (H1 dữ liệu thô, H2 chọn file).
  2. Remove `row_index`, `label_num` (H3, H4).
  3. `RemoveDuplicates` (H5, ghi số bản ghi trước/sau). Kiểm tra Missing của `text` (H6).
     Lưu `enron_dedup.arff`.
  4. Minh họa trên bản sao: `StringToWordVector -C -L -M 2 -W 2000`, stopwords handler
     `Rainbow` (H7, H8). `AttributeSelection` với `InfoGainAttributeEval` + `Ranker -N 500`
     (H9, H10). Visualize phân bố lớp và 2 từ có Info Gain cao nhất (H11). Undo về `enron_dedup`.
  5. `Randomize -S 42` (H12). `RemovePercentage -P 30` rồi lưu `train.arff` (H13). Undo,
     `RemovePercentage -P 30 -V` rồi lưu `test.arff` (H14). Thứ tự Randomize phải giống hệt ở hai lần.
  6. Classify: `FilteredClassifier` với filter là `MultiFilter[StringToWordVector (tham số như
     bước 4), AttributeSelection (như bước 4)]`, classifier `NaiveBayesMultinomial` (H15).
     Chạy Use training set (H16), Cross-validation 10 folds (H17), Supplied test set = `test.arff`
     (H18). Mỗi lần chạy lưu toàn văn output vào `results/nb_*.txt`. Riêng lần Supplied test set
     bật More options → Output predictions → CSV và lưu `nb_test_predictions.csv`.
  7. Lặp lại bước 6 với `J48 -C 0.25 -M 2` (H19–H21), và Visualize tree (H22).
  8. Batch filtering trên Simple CLI, ghi nguyên văn hai lệnh vào checklist:
     `java weka.filters.unsupervised.attribute.StringToWordVector -b -i train.arff -o train_s2w.arff -r test.arff -s test_s2w.arff -C -L -M 2 -W 2000 -stopwords-handler weka.core.stopwords.Rainbow -c last`
     rồi `java weka.filters.supervised.attribute.AttributeSelection -b -i train_s2w.arff -o train_vec.arff -r test_s2w.arff -s test_vec.arff -E weka.attributeSelection.InfoGainAttributeEval -S "weka.attributeSelection.Ranker -N 500" -c first`.
     Lệnh thứ hai dùng `-c first` vì sau `StringToWordVector`, thuộc tính `class` chuyển lên đầu
     còn các từ được nối vào sau. Mở `train_s2w.arff` trong Preprocess để xác nhận trước khi chạy.
  9. Experimenter: dataset `train.arff`, 10 lần lặp, 10-fold CV, hai FilteredClassifier như bước
     6 và 7 (H23). Analyse dùng `Paired T-Tester (corrected)`, significance 0.05, các chỉ số
     `IR_precision`/`F_measure` (lớp spam), `Percent_correct`, `Area_under_ROC` (H24). Lưu `experimenter.txt`.

  Ghi chú tối ưu tùy chọn: thử `Ranker -N` với các giá trị {200, 500, 1000} bằng CV trên `train.arff`
  và chọn F-Measure spam cao nhất. Nếu đổi N thì phải đổi đồng nhất ở bước 4, 6, 7, 8, 9.
- [ ] **Step 6:** Commit `feat: export-arff command and Weka checklist`.

> **Điểm chờ người dùng:** bắt đầu từ Task 4, cần có file trong `reports/weka/data` và `reports/weka/results`.
> Các task không phụ thuộc Weka (5, 8, 9) có thể làm trước.

### Task 4: Đối chiếu NB Python với Weka

**Files:**
- Create: `src/spam_detector/weka_compare.py`
- Modify: `src/spam_detector/cli.py` (lệnh `compare-weka`)
- Test: `tests/test_weka_compare.py`

**Interfaces:**
- Consumes: `read_vector_arff` (Task 1), `MultinomialNaiveBayes` (Task 2),
  `classification_metrics(y, prediction)` có sẵn trong `tree_workflow.py`.
- Produces:
  - `read_weka_predictions(csv_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: trả về
    nhãn thật, nhãn dự đoán và xác suất của nhãn dự đoán, theo thứ tự `inst#`.
  - `compare_with_weka(train_vec, test_vec, weka_predictions_csv, output_dir) -> dict`: ghi
    `output_dir/nb_python_vs_weka.json` gồm `{python_metrics, weka_metrics, n_test, agreement,
    max_abs_proba_diff, disagreements: [inst#...]}`.
  - Lệnh `spam-detector compare-weka --train-vec … --test-vec … --weka-pred … --output reports/bao-cao/results`.

- [ ] **Step 1: Viết các test lỗi**
  - `test_read_weka_predictions_variants`: CSV có header `inst#,actual,predicted,error,prediction`,
    có dòng `2,2:spam,1:ham,+,0.7`, và dòng trống ở cuối. Kết quả parse đúng.
  - `test_compare_identical_on_fixture`: dùng ARFF vector nhỏ cùng CSV dự đoán tạo từ chính
    NB Python. Kết quả `agreement == n_test`, `disagreements == []`.
- [ ] **Step 2:** Chạy test. Kết quả mong đợi: FAIL.
- [ ] **Step 3:** Cài đặt module và lệnh CLI.
- [ ] **Step 4:** Chạy test. Kết quả mong đợi: PASS. Sau đó chạy trên dữ liệu thật. Tiêu chí chấp
  nhận là `agreement == n_test` và `max_abs_proba_diff < 1e-3`. Nếu không đạt, chẩn đoán bằng
  superpowers:systematic-debugging. Những nghi vấn đầu tiên: prior Laplace, thứ tự lớp,
  thuộc tính lớp lọt vào X. **Không được chỉnh code chỉ để khớp số liệu nếu chưa hiểu nguyên nhân.**
- [ ] **Step 5:** Commit `feat: compare self-written NB with Weka predictions`.

### Task 5: Release NB và predictor văn bản

**Files:**
- Create: `src/spam_detector/nb_release.py`
- Test: `tests/test_nb_release.py`

**Interfaces:**
- Consumes: `MultinomialNaiveBayes` (Task 2), `BinaryTextFeatures` (`features.py`), `sha256`,
  `write_json` (`data.py`), `PREPROCESSING_VERSION`.
- Produces:
  - `save_nb_release(release_dir, model, features, metadata) -> None`: ghi `nb.json`,
    `features.json` và `release.json` (`format_version: 1`, `preprocessing_version`,
    `checksums` cho `nb.json` và `features.json`, `model: "self_written_multinomial_nb"`).
    Cấu trúc theo `enron_training.py:251-256`.
  - `NBTextPredictor(release_dir)`, có `predict_texts(texts) -> list[dict]`. Mỗi dict gồm
    `label`, `class_name`, `score_kind: "probability_uncalibrated"`, `score_value` (P(spam)),
    `matched_features`, `unique_tokens`, `top_words: [{"word", "contribution"}]` (5 phần tử) và
    `warning` (khác None khi `matched_features == 0`).
- Đặc trưng: NB demo dùng `BinaryTextFeatures` của repo, tức đếm nhị phân có/không. Báo cáo gọi
  đây là "binarized multinomial NB", biến thể Metsis 2006 đánh giá tốt.

- [ ] **Step 1: Viết các test lỗi**
  - `test_release_roundtrip`: train trên 6 email giả, lưu, nạp lại. Kết quả `predict_texts` giống
    hệt trước khi lưu.
  - `test_checksum_mismatch_rejected`: sửa 1 byte trong `nb.json` thì nạp lại báo `ValueError`.
  - `test_no_vocabulary_warning`: email `"zzzz qqqq"` cho `matched_features == 0` và `warning` khác None.
- [ ] **Step 2:** Chạy test. Kết quả mong đợi: FAIL.
- [ ] **Step 3:** Cài đặt `nb_release.py`.
- [ ] **Step 4:** Chạy test. Kết quả mong đợi: PASS.
- [ ] **Step 5:** Commit `feat: NB release bundle and text predictor`.

### Task 6: Thực nghiệm Python cho báo cáo

**Files:**
- Create: `src/spam_detector/report_experiments.py`
- Modify: `src/spam_detector/cli.py` (lệnh `report-run`)
- Test: `tests/test_report_experiments.py`

**Interfaces:**
- Consumes: `read_text_arff` (Task 1), `save_nb_release` (Task 5), `BinaryTextFeatures`,
  `DecisionTreeClassifier`, `save_tree_bundle`, `clean_csv(path, "spamassassin")`,
  `classification_metrics`.
- Produces:
  - `run_report_experiments(train_arff, test_arff, external_csv, output_dir, artifacts_dir, k=1000) -> dict`.
    Hàm học `BinaryTextFeatures(k=1000, min_df=2)` trên train, rồi train NB (`alpha=1`) và DT
    (`max_depth=15, min_samples_split=10, min_samples_leaf=1`). Đánh giá trên `test` và
    `external` (SpamAssassin). Kết quả ghi ra:
    - `output_dir/results.json` với cấu trúc `{model: {split: metrics}}`.
    - Hình `output_dir/figures/confusion_{nb,tree}_{test,external}.png`.
    - `output_dir/figures/top_words_nb.png` (top 20 từ theo log-odds).
    - Hai release `artifacts_dir/nb-weka7030-v1/` và `artifacts_dir/tree-weka7030-v1/`. Release
      tree có cùng định dạng với `tree-enron-v1` để `TextPredictor` nạp được.
  - Lệnh `spam-detector report-run --train reports/weka/data/train.arff --test reports/weka/data/test.arff --external data/raw/spamassassin/completeSpamAssassin.csv --output reports/bao-cao/results --artifacts artifacts`.

- [ ] **Step 1:** Viết test `test_report_run_small`: chạy với ARFF và CSV fixture khoảng 20 email,
  `k=20`. Kiểm tra `results.json` có đủ hai khóa `nb` và `tree`, mỗi khóa có `test` và
  `external`, đủ 4 hình, và hai release nạp được bằng `NBTextPredictor` / `TextPredictor`.
- [ ] **Step 2:** Chạy test. Kết quả mong đợi: FAIL.
- [ ] **Step 3:** Cài đặt module và lệnh CLI.
- [ ] **Step 4:** Chạy test. Kết quả mong đợi: PASS. Sau đó chạy lệnh thật (cần `train.arff` và `test.arff` từ Weka).
- [ ] **Step 5:** Commit `feat: report experiments on Weka 70/30 split and SpamAssassin`. Có commit
  cả `results.json`, các hình và hai release.

### Task 7: Demo Streamlit dùng NB

**Files:**
- Modify: `src/spam_detector/web_service.py`, `app/streamlit_app.py`, `configs/web.yaml`
- Test: `tests/test_web_service.py`, `tests/test_web_app.py`

**Interfaces:**
- Consumes: `NBTextPredictor` (Task 5), các release từ Task 6.
- Produces:
  - `default_loader` hỗ trợ `adapter: nb`.
  - `configs/web.yaml`: `default_model: nb`, `nb.release: artifacts/nb-weka7030-v1`,
    `tree.release: artifacts/tree-weka7030-v1`.
  - `registry_fingerprint` băm thêm `nb.json`.
  - Khi `model_id == "nb"` và `matched_features == 0`, `PredictionService.predict` báo
    `ValueError("Không dự đoán được: email không chứa từ nào trong bộ từ vựng của mô hình.")`.
    Email rỗng vẫn dùng thông báo hiện có.
  - Streamlit hiển thị bảng 5 `top_words` cho NB. Phần hiển thị cho tree giữ nguyên.

- [ ] **Step 1: Viết các test lỗi**
  - `test_nb_service_matches_predictor`: kết quả qua service bằng kết quả `NBTextPredictor`
    trực tiếp. Dùng fixture release tạo trong test.
  - `test_nb_unpredictable_when_no_vocabulary`: `service.predict("", "zzzz qqqq", model_id="nb")`
    báo lỗi có chứa `"Không dự đoán được"`.
  - Sửa `test_web_app` để mô hình mặc định là NB và có hiển thị "Từ đóng góp".
- [ ] **Step 2:** Chạy `pytest tests/test_web_service.py tests/test_web_app.py -v`. Kết quả mong đợi: các test mới FAIL.
- [ ] **Step 3:** Cài đặt các thay đổi.
- [ ] **Step 4:** Chạy toàn bộ `pytest -q`. Kết quả mong đợi: PASS. Sau đó chạy app, chụp 3 ảnh
  `reports/bao-cao/figures/demo-{spam,ham,khong-du-doan}.png`.
- [ ] **Step 5:** Commit `feat: NB as default model in Streamlit demo`.

### Task 8: Khung dựng báo cáo Word

**Files:**
- Create: `reports/bao-cao/build_docx.py`, `reports/bao-cao/content/00-loi-cam-on.md`,
  `01-mo-ta-bai-toan.md`, `02-quy-trinh-kpdl.md`, `03-phan-lop-naive-bayes.md`,
  `04-trien-khai.md`, `05-ket-luan.md`, `06-tai-lieu-tham-khao.md`
- Modify: `pyproject.toml` (extra `report`)
- Test: `tests/test_report_build.py`

**Interfaces:**
- Produces: lệnh `python reports/bao-cao/build_docx.py [--allow-missing-figures]`, ghi ra
  `reports/bao-cao/BaoCao_Nhom12_SpamEmail.docx`.
  - Khổ A4, lề trái 3 cm, phải 2 cm, trên và dưới 2 cm. Font Times New Roman 13, giãn dòng 1.3.
  - Header trang chẵn là "Khai phá dữ liệu", trang lẻ là "Nhóm 12", giống mẫu. Footer có số trang.
  - Thứ tự trong file: bìa (Global Constraints), MỤC LỤC (TOC field), LỜI CẢM ƠN, các chương.
  - Ảnh viết theo cú pháp `![Hình N: chú thích](path)`. Công thức viết LaTeX `$...$`, pandoc
    chuyển thành công thức Word.
  - Nếu thiếu ảnh: mặc định báo lỗi và liệt kê đường dẫn. Khi bật `--allow-missing-figures`,
    chèn đoạn chữ "[Chèn Hình N]" thay cho ảnh.
- Nội dung ban đầu của mỗi file `.md` là toàn bộ đề mục của spec mục 5, đánh số như mẫu
  (I., 1., 1.1., 1.1.1.), dưới mỗi đề mục để `TODO`.

- [ ] **Step 1:** Viết test `test_build_docx_skeleton`: build với `--allow-missing-figures` vào thư
  mục tạm. Mở bằng python-docx và kiểm tra:
  - Có chuỗi "Nhóm 12", đủ 4 MSSV, "HÀ NỘI, 2026".
  - Có các heading "I. Mô tả bài toán", "II. Quy trình khai phá dữ liệu", "III. Phương pháp
    phân lớp và thuật toán Naive Bayes", "IV. Triển khai thuật toán (Python)", "KẾT LUẬN",
    "TÀI LIỆU THAM KHẢO".
  - Có một TOC field.
- [ ] **Step 2:** Chạy test. Kết quả mong đợi: FAIL.
- [ ] **Step 3:** Cài đặt `build_docx.py`. Tạo reference.docx bằng python-docx lúc chạy, gọi
  `pypandoc.convert_file(..., "docx", extra_args=["--reference-doc", ref])`, rồi dùng python-docx
  chèn bìa và mục lục vào đầu file.
- [ ] **Step 4:** Chạy test. Kết quả mong đợi: PASS. Mở file bằng Word và xác nhận font, header và mục lục.
- [ ] **Step 5:** Commit `feat: Word report build pipeline and chapter skeleton`.

### Task 9: Tổng quan tài liệu (mục I.3) và tài liệu tham khảo

**Files:**
- Modify: `reports/bao-cao/content/01-mo-ta-bai-toan.md` (mục I.3), `06-tai-lieu-tham-khao.md`
- Create: `reports/bao-cao/references.csv` (cột `key, authors, year, title, venue, doi_or_url, verified_on, in_scholar_list`)

- [ ] **Step 1:** Tìm 6–10 bài về lọc spam/phishing bằng ML. Ít nhất 4 bài phải thuộc venue
  trong phụ lục A của spec. Có thể đưa thêm các bài nền tảng: Androutsopoulos SIGIR 2000,
  Metsis CEAS 2006, Sahami 1998.
- [ ] **Step 2:** Xác minh từng bài bằng cách mở trang DOI hoặc trang chính thức, rồi ghi
  `verified_on`. Bài nào không xác minh được thì bỏ.
- [ ] **Step 3:** Viết mục I.3 gồm một đoạn tổng quan và bảng tóm tắt (tác giả, năm, venue, dữ liệu,
  phương pháp, kết quả). Thêm đoạn chốt "khoảng trống" dẫn sang lý do chọn NB và DT.
- [ ] **Step 4:** Kiểm tra: build docx thành công; mỗi trích dẫn `[n]` trong nội dung có mục tương
  ứng trong `06-…`, và ngược lại.
- [ ] **Step 5:** Commit `docs: literature review with verified references`.

### Task 10: Viết chương I và II

**Files:** `reports/bao-cao/content/00-loi-cam-on.md`, `01-mo-ta-bai-toan.md`, `02-quy-trinh-kpdl.md`

**Interfaces:** Consumes: ảnh H1–H14 và số bản ghi ghi trong CHECKLIST (Task 3); giả thuyết theo spec mục 7.

- [ ] **Step 1:** Viết Lời cảm ơn và các mục I.1, I.2, I.4 (H0/H1 nguyên văn spec mục 7), I.5, I.6.
  I.6 có bảng thuộc tính `row_index`, `label_num`, `text`, `class` kèm ý nghĩa.
- [ ] **Step 2:** Viết chương II theo spec mục 5, chèn H1–H14 kèm chú thích. Số bản ghi lấy từ ảnh
  Weka (trước/sau `RemoveDuplicates`, số train/test). Nêu rõ ảnh `StringToWordVector`/`AttributeSelection`
  trên toàn bộ dữ liệu chỉ để minh họa, còn số liệu dùng `FilteredClassifier`.
- [ ] **Step 3:** Kiểm tra: build không cần `--allow-missing-figures` cho H1–H14. Mọi con số trong
  chương II khớp với ảnh, đối chiếu từng con số một.
- [ ] **Step 4:** Commit `docs: report chapters I-II`.

### Task 11: Viết chương III, IV và Kết luận

**Files:** `03-phan-lop-naive-bayes.md`, `04-trien-khai.md`, `05-ket-luan.md`

**Interfaces:** Consumes: `reports/weka/results/*.txt`, `experimenter.txt`, ảnh H15–H24,
`reports/bao-cao/results/{results.json, nb_python_vs_weka.json, figures/*}`, ảnh demo (Task 7).

- [ ] **Step 1:** Viết III.1–III.2:
  - Lý thuyết NB: công thức Bayes, multinomial, Laplace, log.
  - Quy trình 6 bước như mẫu.
  - Bảng kết quả 3 cách kiểm thử, lấy số từ `nb_*.txt`.
  - Giải thích các thống kê và độ đo, đủ danh sách như mẫu.
- [ ] **Step 2:** Viết III.3 (J48), III.4 (kiểm định giả thuyết: bảng Experimenter, kết luận bác
  bỏ hoặc không bác bỏ H0 ở α = 0,05), III.5 (nhận xét, 3–5 email bị phân loại sai lấy từ
  `nb_test_predictions.csv`), III.6 (SpamAssassin, lấy từ `results.json`).
- [ ] **Step 3:** Viết chương IV:
  - Trích code `fit`/`predict_proba` của NB.
  - Bảng đối chiếu Python và Weka lấy từ `nb_python_vs_weka.json`.
  - Bảng 4 mô hình trên test 30%.
  - Ảnh demo spam, ham và trường hợp "Không dự đoán được".
- [ ] **Step 4:** Viết Kết luận theo giọng của mẫu: kết quả đạt được, hạn chế (lệch miền dữ liệu
  SpamAssassin, chỉ có tiếng Anh), hướng phát triển.
- [ ] **Step 5:** Kiểm tra: build không cần `--allow-missing-figures`. Mọi số liệu trong III/IV khớp
  file nguồn (đối chiếu từng số). Không còn `TODO` nào (`grep -n TODO content/` trả về rỗng).
- [ ] **Step 6:** Commit `docs: report chapters III-IV and conclusion`.

### Task 12: Dựng bản cuối và rà soát theo mẫu

**Files:** `reports/bao-cao/BaoCao_Nhom12_SpamEmail.docx`, `reports/bao-cao/REVIEW.md`

- [ ] **Step 1:** Build bản cuối. Mở bằng Word, Update Field để cập nhật mục lục, rồi xuất thêm PDF để đọc thử.
- [ ] **Step 2:** Viết `REVIEW.md`: bảng đối chiếu từng mục của báo cáo mẫu
  (`C:\Users\mrdoanh\Downloads\Bai tap lon.pdf`) với mục tương ứng trong báo cáo nhóm, cột "có/không".
  Thêm các mục ★ của spec. Rà lại danh sách phản mục tiêu ở spec mục 2.
- [ ] **Step 3:** Chạy `pytest -q`. Kết quả mong đợi: tất cả PASS.
- [ ] **Step 4:** Commit `docs: final report build and review against sample`. Gửi file .docx cho nhóm đọc duyệt.
