# II. Quy trình khai phá dữ liệu

## 1. Tiền xử lý dữ liệu

Tiền xử lý là quá trình xử lý dữ liệu thô/gốc nhằm cải thiện chất lượng dữ liệu và chất lượng của kết quả khai phá dữ liệu.

### 1.1. Làm sạch dữ liệu

Làm sạch là quá trình xử lý dữ liệu bị thiếu, nhận diện phần tử trùng lặp hoặc gây nhiễu, và xử lý dữ liệu không nhất quán.

#### 1.1.1. Loại bỏ thuộc tính dư thừa

- Đọc dữ liệu vào Weka: tab **Preprocess → Open file…**, chọn tệp `enron_raw.arff`. Weka đọc được 5171 bản ghi và 4 thuộc tính.

![Hình 2: Chọn file dữ liệu](../weka/figures/hinh-02.png)

- Loại bỏ thuộc tính dư thừa, không cần thiết:
  - `row_index`: chỉ là số thứ tự của bản ghi trong tệp gốc, không mang thông tin về nội dung thư.
  - `label_num`: là nhãn ở dạng số (0/1), trùng nghĩa hoàn toàn với thuộc tính lớp `class`. Nếu giữ lại, mô hình sẽ "nhìn thấy" đáp án và cho kết quả sai lệch (rò rỉ nhãn).
- Tick chọn hai thuộc tính `row_index` và `label_num`, nhấn **Remove** để loại bỏ:

![Hình 3: Xóa bỏ thuộc tính row_index và label_num](../weka/figures/hinh-03.png)

➢ Ta thu được kết quả: dữ liệu còn 2 thuộc tính là `text` (nội dung email) và `class` (nhãn).

![Hình 4: Dữ liệu sau khi loại bỏ thuộc tính dư thừa](../weka/figures/hinh-04.png)

#### 1.1.2. Loại bỏ dữ liệu trùng lặp

- Trong tập dữ liệu có những email bị lặp lại nguyên văn (cùng nội dung và cùng nhãn), ví dụ các thư thông báo gửi nhiều lần. Bản ghi trùng làm mô hình "học thuộc" một số mẫu và nếu một bản nằm ở tập huấn luyện, bản còn lại nằm ở tập kiểm thử thì kết quả kiểm thử sẽ bị cao giả tạo.
- Xử lý bằng bộ lọc: **Filter → Choose → unsupervised → instance → RemoveDuplicates → Apply**.

➢ Số bản ghi giảm từ **5171** xuống **4993**, tức đã loại **178** bản ghi trùng (141 thư ham và 37 thư spam).

![Hình 5: Loại bỏ bản ghi trùng lặp bằng RemoveDuplicates](../weka/figures/hinh-05.png)

#### 1.1.3. Xử lý dữ liệu bị thiếu

- Kiểm tra từng thuộc tính trong khung **Selected attribute**: thuộc tính `text` có **Missing: 0 (0%)** và thuộc tính `class` cũng có **Missing: 0 (0%)**.
- Phân bố lớp sau khi làm sạch: **3531 thư ham (70,7%)** và **1462 thư spam (29,3%)**.

➢ Dữ liệu không có giá trị bị thiếu nên không cần áp dụng bộ lọc `ReplaceMissingValues`.

![Hình 6: Thuộc tính class không bị thiếu, phân bố ham/spam sau khi làm sạch](../weka/figures/hinh-06.png)

#### 1.1.4. Xử lý dữ liệu không nhất quán

- Dữ liệu không nhất quán ở bài toán này là trường hợp cùng một nội dung email nhưng được gán hai nhãn khác nhau.
- Sau khi loại trùng, thuộc tính `text` có 4993 giá trị phân biệt trên 4993 bản ghi (**Distinct: 4993, Unique: 100%**, Hình 5), tức không có nội dung nào xuất hiện hai lần với hai nhãn khác nhau. Nhãn chỉ nhận hai giá trị hợp lệ là `ham` và `spam`.

➢ Tập dữ liệu đã nhất quán. Dữ liệu sau khi làm sạch được lưu thành tệp `enron_dedup.arff`.

### 1.2. Tích hợp dữ liệu

- Tích hợp là quá trình trộn dữ liệu từ các nguồn khác nhau vào một kho dữ liệu sẵn sàng cho quá trình khai phá dữ liệu.
- Quá trình này liên quan đến cấu trúc và tính không thuần nhất (heterogeneity) về ngữ nghĩa (semantics) của dữ liệu.
- Tích hợp giúp giảm và tránh dư thừa, không nhất quán về dữ liệu, từ đó cải thiện độ chính xác và tốc độ của quá trình khai phá.

➢ Dữ liệu huấn luyện chỉ lấy từ một nguồn (Enron) nên bỏ qua bước này. Bộ SpamAssassin [10] không trộn vào dữ liệu huấn luyện mà chỉ dùng để kiểm thử trên nguồn khác (mục III.6).

### 1.3. Biến đổi dữ liệu (vector hóa văn bản)

- Biến đổi dữ liệu là quá trình biến đổi hay kết hợp dữ liệu vào những dạng thích hợp cho quá trình khai phá dữ liệu, bao gồm: làm trơn dữ liệu, kết hợp dữ liệu, tổng quát hóa, chuẩn hóa, xây dựng thuộc tính và thu giảm dữ liệu.
- Các thuật toán phân lớp không làm việc trực tiếp với văn bản, vì vậy cần chuyển mỗi email thành một vector số. Nhóm dùng bộ lọc **StringToWordVector** (Filter → unsupervised → attribute) theo mô hình túi từ (bag-of-words): mỗi từ là một thuộc tính, giá trị là số lần từ đó xuất hiện trong email.
- Các tham số được thiết lập:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| outputWordCounts | True | Lưu số lần xuất hiện của từ (phù hợp Multinomial Naive Bayes) thay vì chỉ có/không |
| lowerCaseTokens | True | Chuyển về chữ thường để "Free" và "free" là một từ |
| minTermFreq | 2 | Bỏ các từ xuất hiện quá ít |
| wordsToKeep | 2000 | Số từ giữ lại cho mỗi lớp |
| stopwordsHandler | Rainbow | Loại các từ dừng tiếng Anh (the, and, of…) |
| attributeNamePrefix | W_ | Thêm tiền tố cho tên thuộc tính từ |

- Tiền tố `W_` là bắt buộc: trong các email có xuất hiện từ "class", nếu không có tiền tố thì thuộc tính sinh ra từ từ này trùng tên với thuộc tính lớp `class` và Weka báo lỗi "Attribute names are not unique".

![Hình 7: Cấu hình bộ lọc StringToWordVector](../weka/figures/hinh-07.png)

➢ Ta thu được kết quả: dữ liệu có **3321 thuộc tính**, gồm 3320 từ và thuộc tính lớp `class` (được chuyển lên vị trí đầu tiên).

![Hình 8: Dữ liệu sau khi vector hóa văn bản](../weka/figures/hinh-08.png)

## 2. Phân tích dữ liệu

Phân tích dữ liệu nhằm hiểu rõ hơn về dữ liệu và mối quan hệ giữa các thuộc tính với nhãn lớp.

➢ Thống kê mô tả: dữ liệu đưa vào khai phá gồm **4993 email**, 2 thuộc tính gốc:

| STT | Thuộc tính | Ý nghĩa | Thống kê |
|---|---|---|---|
| 1 | text | Nội dung email | 4993 giá trị phân biệt, không thiếu |
| 2 | class | Nhãn của email | ham: 3531 (70,7%), spam: 1462 (29,3%) |

- Dữ liệu lệch lớp ở mức vừa phải (tỉ lệ khoảng 7 : 3). Vì vậy ngoài độ chính xác (Accuracy), cần quan tâm Precision, Recall và F-Measure của lớp spam.

➢ Mối tương quan giữa các từ và nhãn lớp: sau khi chọn thuộc tính theo Information Gain (mục 3), hai từ có khả năng phân biệt cao nhất là **`enron`** và **`2000`**, tiếp theo là `cc`, `hpl`, `daren`, `http`, `gas`. Biểu đồ của từ `enron` (màu xanh là ham, đỏ là spam) cho thấy:

![Hình 9: Phân bố thuộc tính W_enron theo lớp](../weka/figures/hinh-11.png)

➢ Kết luận:

- Các từ liên quan đến công việc nội bộ của Enron (`enron`, `hpl`, `gas`, `meter`, `daren`, `cc`, `forwarded`) xuất hiện chủ yếu trong thư ham.
- Thư spam thường chứa đường dẫn (`http`) và các từ quảng cáo, mời chào.
- Điều này cũng cho thấy một hạn chế: mô hình học được nhiều từ đặc thù của công ty Enron, nên có thể kém chính xác với thư của tổ chức khác (kiểm chứng ở mục III.6).

## 3. Biến đổi và lựa chọn thuộc tính

- Sau khi vector hóa có tới 3320 thuộc tính từ, phần lớn ít liên quan đến nhãn. Nhóm thu giảm dữ liệu bằng bộ lọc **AttributeSelection** (Filter → supervised → attribute) với:
  - `evaluator` = **InfoGainAttributeEval**: đánh giá mỗi từ theo độ lợi thông tin đối với lớp, $Gain(A) = Entropy(D) - Entropy_A(D)$;
  - `search` = **Ranker**, `numToSelect` = **500**: xếp hạng và giữ lại 500 từ tốt nhất.

![Hình 10: Cấu hình bộ lọc AttributeSelection](../weka/figures/hinh-09.png)

➢ Ta thu được kết quả: dữ liệu còn **501 thuộc tính** (500 từ và `class`), các từ được xếp theo Information Gain giảm dần.

![Hình 11: Dữ liệu sau khi chọn 500 thuộc tính tốt nhất](../weka/figures/hinh-10.png)

**Lưu ý quan trọng về rò rỉ dữ liệu.** Các bước vector hóa (Hình 7–8) và chọn thuộc tính (Hình 10–11) ở trên được áp dụng trên toàn bộ dữ liệu **chỉ để minh họa**. Khi xây dựng và đánh giá mô hình (chương III), hai bộ lọc này được đặt **bên trong** `FilteredClassifier`. Nhờ đó, từ vựng và 500 từ được chọn chỉ học trên tập huấn luyện (hoặc trên phần huấn luyện của từng fold), tập kiểm thử hoàn toàn không tham gia vào việc chọn đặc trưng.

## 4. Phân tách dữ liệu (70/30)

- Lý do:
  - Chia dữ liệu thành tập huấn luyện và tập kiểm thử giúp đánh giá hiệu suất của mô hình trên dữ liệu mới. Tập huấn luyện dùng để xây dựng mô hình, tập kiểm thử dùng để đánh giá mô hình trên dữ liệu mà mô hình chưa từng thấy.
  - Việc tách riêng tập kiểm thử cũng giúp phát hiện hiện tượng quá khớp (overfitting).
- Dữ liệu huấn luyện và dữ liệu kiểm thử:
  - Dữ liệu huấn luyện (training data): tập dữ liệu dùng để xây dựng thuật toán, mô hình học máy.
  - Dữ liệu kiểm thử (test data): tập dữ liệu đưa vào sau khi đã xây dựng xong mô hình, nhằm xác nhận tính hiệu quả của mô hình trong thực tế.
- Lựa chọn tỉ lệ 70/30, thực hiện trên dữ liệu đã làm sạch `enron_dedup.arff` (chỉ gồm `text` và `class`):
  - Để khách quan, trộn dữ liệu ngẫu nhiên: **Filter → unsupervised → instance → Randomize** với `randomSeed = 42` (cố định để có thể lặp lại kết quả).

![Hình 12: Trộn ngẫu nhiên dữ liệu với Randomize (seed = 42)](../weka/figures/hinh-12.png)

  - Dùng bộ lọc **RemovePercentage** với `percentage = 30`: loại 30% bản ghi, phần còn lại (70%) là tập huấn luyện, lưu thành `train.arff`.

![Hình 13: Phân chia 70% dữ liệu làm tập huấn luyện](../weka/figures/hinh-13.png)

  - Hoàn tác (Undo) về dữ liệu đã trộn, áp dụng lại RemovePercentage với `invertSelection = True` để lấy đúng 30% còn lại làm tập kiểm thử, lưu thành `test.arff`.

![Hình 14: Phân chia 30% dữ liệu làm tập kiểm thử](../weka/figures/hinh-14.png)

➢ Kết quả phân tách:

| Tập dữ liệu | Số email | ham | spam |
|---|---|---|---|
| Huấn luyện (70%) | 3495 | 2462 | 1033 |
| Kiểm thử (30%) | 1498 | 1069 | 429 |
| Tổng | 4993 | 3531 | 1462 |

Nhóm kiểm tra lại bằng chương trình: hai tập không có email nào trùng nhau và gộp lại đúng bằng 4993 email của tập dữ liệu sạch. Tỉ lệ ham/spam ở hai tập gần như nhau (khoảng 70/30).

