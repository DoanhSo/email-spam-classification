# Checklist thao tác Weka 3.8.7 (Nhóm 12 – Email spam)

Mục đích: tạo đủ ảnh minh họa, tập dữ liệu và kết quả cho chương II–III của báo cáo.
Mỗi bước ghi rõ: **hình cần chụp**, **đường dẫn menu Weka**, **tham số** và **file cần lưu**.

## Quy ước chung

- Thư mục gốc: `C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\`.
  - Ảnh lưu vào `figures\hinh-01.png`, `hinh-02.png`, … đúng số thứ tự ghi ở từng bước.
  - Dữ liệu lưu vào `data\`. Kết quả lưu vào `results\` (tạo thư mục nếu chưa có).
- Chụp màn hình: nhấn **Win + Shift + S**, chọn vùng cửa sổ Weka. Mở Paint, dán, rồi lưu dạng PNG.
  Có thể chụp cả cửa sổ để nhìn rõ.
- Mở Weka: **Weka GUI Chooser → Explorer**. Riêng bước 9 dùng **Experimenter**, bước 8 dùng **Simple CLI**.
- Ở mọi lần chạy, thuộc tính lớp phải là **(Nom) class**. Kiểm tra ô chọn lớp ở tab Classify,
  và ở góc phải tab Preprocess.
- Sau mỗi bước, ghi các con số vào **bảng số liệu ở cuối file**.

---

## Bước 1 – Đọc dữ liệu thô (H1, H2)

1. Tab **Preprocess → Open file…**, chọn `data\enron_raw.arff`. Chụp hộp thoại chọn file → **H2**.
2. Bấm **Edit…** để xem bảng dữ liệu, kéo cho thấy các cột `row_index`, `label_num`, `text`,
   `class` → chụp **H1**. Đóng cửa sổ Edit (Cancel).
3. Ghi số bản ghi (Instances) vào bảng. Giá trị mong đợi: **5171**.

## Bước 2 – Loại bỏ thuộc tính dư thừa (H3, H4)

1. Tick `row_index` (chỉ là số thứ tự) và `label_num` (trùng nghĩa với `class`, nếu giữ sẽ lộ nhãn).
2. Bấm **Remove**. Chụp ngay trước khi bấm, khi đang tick hai ô → **H3**.
3. Chụp màn hình sau khi xóa (còn `text`, `class`) → **H4**.

## Bước 3 – Loại bản ghi trùng, kiểm tra thiếu (H5, H6)

1. **Filter → Choose → weka → filters → unsupervised → instance → RemoveDuplicates → Apply.**
   Chụp màn hình sau khi Apply, ô Instances phải hiện số mới → **H5**. Ghi số bản ghi trước và sau vào bảng.
2. Bấm vào thuộc tính `text`, xem dòng **Missing** (mong đợi 0 (0%)). Bấm `class` để xem phân bố
   ham/spam → chụp **H6**. Ghi số ham và spam.
3. **Save…** → `data\enron_dedup.arff`.

## Bước 4 – Minh họa biến đổi và chọn thuộc tính (H7–H11)

> Các phép biến đổi ở bước này áp lên **toàn bộ** dữ liệu và **chỉ để minh họa**. Số liệu đánh giá
> lấy ở bước 6–9, khi bộ lọc nằm trong FilteredClassifier.

1. **Filter → Choose → unsupervised → attribute → StringToWordVector.** Bấm vào tên bộ lọc để mở
   bảng tham số và đặt:
   - `attributeNamePrefix = W_` (bắt buộc: email có chứa từ "class", nếu không đặt tiền tố thì từ này trùng tên với thuộc tính lớp và Weka báo lỗi "Attribute names are not unique")
   - `outputWordCounts = True`
   - `lowerCaseTokens = True`
   - `minTermFreq = 2`
   - `wordsToKeep = 2000`
   - `stopwordsHandler = Rainbow`
   - Các tham số khác giữ mặc định.
   Chụp bảng tham số → **H7**. Bấm OK, rồi Apply. Chụp danh sách thuộc tính sau khi biến đổi → **H8**.
   Ghi số thuộc tính.
2. Ô chọn lớp (ngay trên nút Visualize All) chọn **(Nom) class**.
3. **Filter → Choose → supervised → attribute → AttributeSelection**, đặt:
   - `evaluator = InfoGainAttributeEval`
   - `search = Ranker`, `numToSelect = 500`
   Chụp bảng tham số → **H9**. Apply, rồi chụp kết quả còn 501 thuộc tính (500 từ và class) → **H10**.
4. **Visualize All**, hoặc bấm từng thuộc tính: chụp phân bố lớp và 2 từ đứng đầu danh sách
   (Info Gain cao nhất), tô màu theo class → **H11**. Ghi tên 2 từ đó.
5. Bấm **Undo** 2 lần, hoặc Open lại `data\enron_dedup.arff`, để quay về dữ liệu chỉ có `text`, `class`.

## Bước 5 – Phân tách dữ liệu 70/30 (H12–H14)

Làm trên `enron_dedup.arff` (chỉ có `text` và `class`).

1. **Filter → unsupervised → instance → Randomize**, đặt `randomSeed = 42` → chụp bảng tham số
   → **H12**. Apply.
2. **Filter → unsupervised → instance → RemovePercentage**, đặt `percentage = 30`,
   `invertSelection = False`. Apply. Phần còn lại (70%) là tập **train**. Chụp → **H13**.
   **Save…** → `data\train.arff`. Ghi số bản ghi.
3. Bấm **Undo** một lần để quay về dữ liệu đã Randomize. **Không Randomize lại.**
   Mở RemovePercentage, đặt `invertSelection = True`. Apply. Phần còn lại (30%) là tập **test**.
   Chụp → **H14**. **Save…** → `data\test.arff`. Ghi số bản ghi.
4. Kiểm tra: số bản ghi train + test = số bản ghi sau RemoveDuplicates.

## Bước 6 – Naive Bayes Multinomial (H15–H18)

1. Preprocess → Open file `data\train.arff`.
2. Tab **Classify → Choose → meta → FilteredClassifier**. Bấm vào tên để mở bảng tham số:
   - `filter` → Choose **unsupervised → attribute → … MultiFilter** (nằm trong `weka.filters`).
     Mở MultiFilter, sửa `filters` để danh sách gồm đúng 2 bộ lọc theo thứ tự:
     1. `StringToWordVector`, tham số giống hệt bước 4.1.
     2. `supervised → attribute → AttributeSelection`, tham số giống hệt bước 4.3.
   - `classifier` → **bayes → NaiveBayesMultinomial** (giữ mặc định).
   Chụp bảng tham số FilteredClassifier (nếu cần, chụp cả cửa sổ MultiFilter) → **H15**.
   Cấu hình này dùng lại ở bước 7 và 9. Có thể chuột phải ô cấu hình → **Copy configuration
   to clipboard** để dán vào `results\config_nb.txt`.
3. **Use training set → Start.** Chụp cửa sổ kết quả → **H16**. Chuột phải dòng kết quả trong
   Result list → **Save result buffer** → `results\nb_training.txt`.
4. **Cross-validation, Folds = 10 → Start.** Chụp → **H17**. Save result buffer → `results\nb_cv.txt`.
5. **Supplied test set → Set… → Open file** `data\test.arff` (Class: class). Trước khi chạy, bấm
   **More options… → Output predictions → Choose → CSV** và đặt `outputFile` =
   `C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\results\nb_test_predictions.csv`. Start.
   Chụp → **H18**. Save result buffer → `results\nb_test.txt`.
   Nếu thấy file CSV rỗng, đặt Output predictions lại về **Null** rồi chạy lại cách khác: chọn
   CSV nhưng để trống outputFile, sau đó copy phần predictions trong output và lưu vào file CSV.
   Sau khi xong, đặt Output predictions về **Null** để các lần chạy sau không in quá dài.

## Bước 7 – Decision Tree J48 (H19–H22)

1. Vẫn dùng FilteredClassifier, giữ nguyên `filter` như bước 6. Đổi `classifier` →
   **trees → J48** (`confidenceFactor = 0.25`, `minNumObj = 2`, đây là mặc định).
2. Chạy 3 lần giống bước 6.3–6.5 (không cần xuất predictions):
   - Use training set → chụp **H19**, lưu `results\j48_training.txt`.
   - Cross-validation 10 → chụp **H20**, lưu `results\j48_cv.txt`.
   - Supplied test set → chụp **H21**, lưu `results\j48_test.txt`.
3. Chuột phải kết quả Supplied test set → **Visualize tree**. Cây lớn thì chuột phải → *Fit to
   screen*, hoặc chụp phần gốc cây → **H22**.

## Bước 8 – Xuất vector cho đối chiếu Python (không cần chụp)

Mở **Weka GUI Chooser → Simple CLI**. Gõ lần lượt hai lệnh dưới đây, mỗi lệnh một dòng rồi Enter,
và chờ chữ "Finished" hoặc dấu nhắc mới:

```
java weka.filters.unsupervised.attribute.StringToWordVector -b -i C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\data\train.arff -o C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\data\train_s2w.arff -r C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\data\test.arff -s C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\data\test_s2w.arff -C -L -M 2 -W 2000 -P W_ -stopwords-handler weka.core.stopwords.Rainbow -c last
```

Mở `train_s2w.arff` trong Explorer để xem `class` có nằm ở **đầu** danh sách thuộc tính không.
Nếu có, dùng `-c first` như lệnh dưới. Nếu `class` vẫn ở cuối, đổi thành `-c last`.

```
java weka.filters.supervised.attribute.AttributeSelection -b -i C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\data\train_s2w.arff -o C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\data\train_vec.arff -r C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\data\test_s2w.arff -s C:\Users\mrdoanh\Downloads\KhaiPhaDL\reports\weka\data\test_vec.arff -E weka.attributeSelection.InfoGainAttributeEval -S "weka.attributeSelection.Ranker -N 500" -c first
```

Kết quả cần có: `data\train_vec.arff` và `data\test_vec.arff`.

## Bước 9 – Kiểm định giả thuyết bằng Experimenter (H23, H24)

1. **Weka GUI Chooser → Experimenter → Setup → New.** Chuyển sang chế độ **Advanced**
   (ô "Experiment Configuration Mode") để chỉnh được lớp tính độ đo.
2. Result destination: **ARFF file** → `results\experiment.arff`.
3. Result generator: `CrossValidationResultProducer`, `numFolds = 10`. Bấm vào tên →
   `splitEvaluator` = `ClassifierSplitEvaluator` → đặt **`classForIRStatistics`** về lớp
   **spam**. Lớp spam là giá trị thứ 2 của `class {ham,spam}`. Xem tooltip để biết ô này nhận chỉ số
   đếm từ 0 hay từ 1: đếm từ 0 thì nhập 1, đếm từ 1 thì nhập 2.
4. Runs: from **1** to **10**.
5. Datasets → **Add new…** → `data\train.arff`. **Chỉ dùng train**, không dùng test.
6. Generator properties → Enabled → chọn `classifier`. Thêm 2 cấu hình FilteredClassifier giống
   bước 6 (NaiveBayesMultinomial) và bước 7 (J48). Có thể dán từ `results\config_nb.txt`.
   Chụp màn hình Setup → **H23**. **Save…** cấu hình thành `results\experiment.exp`.
7. Tab **Run → Start** và chờ chạy xong. Mất vài phút vì có 2 mô hình × 10 lần × 10 fold.
8. Tab **Analyse → Experiment** (lấy kết quả vừa chạy):
   - `Testing with = Paired T-Tester (corrected)`, `Significance = 0.05`.
   - Test base = mô hình NaiveBayesMultinomial.
   - `Comparison field = F_measure` → **Perform test**. Chụp → **H24**.
   Copy toàn bộ ô kết quả và dán vào `results\experimenter.txt`. Lặp lại với `Percent_correct`,
   `Area_under_ROC`, `IR_precision`, `IR_recall`, dán tiếp vào cùng file và ghi tiêu đề từng phần.

---

## Bảng số liệu (điền trong lúc làm)

| Mục | Giá trị |
|---|---|
| Số bản ghi dữ liệu thô | |
| Số bản ghi sau RemoveDuplicates | |
| Số ham / spam sau khử trùng | |
| Missing của `text` | |
| Số thuộc tính sau StringToWordVector | |
| 2 từ có Info Gain cao nhất | |
| Số bản ghi train / test | |
| Ghi chú bất thường (nếu có) | |

## Danh sách file phải có khi xong

- `figures\hinh-01.png` … `figures\hinh-24.png`
- `data\enron_dedup.arff`, `data\train.arff`, `data\test.arff`, `data\train_vec.arff`, `data\test_vec.arff`
- `results\nb_training.txt`, `nb_cv.txt`, `nb_test.txt`, `nb_test_predictions.csv`
- `results\j48_training.txt`, `j48_cv.txt`, `j48_test.txt`
- `results\experimenter.txt` (và `experiment.exp`, `experiment.arff`)
- Bảng số liệu ở trên đã điền đầy đủ
