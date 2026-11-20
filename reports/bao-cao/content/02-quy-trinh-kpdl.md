# II. Quy trình khai phá dữ liệu

## 1. Tiền xử lý dữ liệu

Tiền xử lý là quá trình xử lý dữ liệu thô/gốc nhằm cải thiện chất lượng dữ liệu và chất lượng của kết quả khai phá dữ liệu.

### 1.1. Làm sạch dữ liệu

Làm sạch là quá trình xử lý dữ liệu bị thiếu, nhận diện phần tử trùng lặp hoặc gây nhiễu, và xử lý dữ liệu không nhất quán.

#### 1.1.1. Loại bỏ thuộc tính dư thừa

TODO

#### 1.1.2. Loại bỏ dữ liệu trùng lặp

TODO

#### 1.1.3. Xử lý dữ liệu bị thiếu

TODO

#### 1.1.4. Xử lý dữ liệu không nhất quán

TODO

### 1.2. Tích hợp dữ liệu

- Tích hợp là quá trình trộn dữ liệu từ các nguồn khác nhau vào một kho dữ liệu sẵn sàng cho quá trình khai phá dữ liệu.
- Quá trình này liên quan đến cấu trúc và tính không thuần nhất (heterogeneity) về ngữ nghĩa (semantics) của dữ liệu.
- Tích hợp giúp giảm và tránh dư thừa, không nhất quán về dữ liệu, từ đó cải thiện độ chính xác và tốc độ của quá trình khai phá.

➢ Dữ liệu huấn luyện chỉ lấy từ một nguồn (Enron) nên bỏ qua bước này. Bộ SpamAssassin [10] không trộn vào dữ liệu huấn luyện mà chỉ dùng để kiểm thử trên nguồn khác (mục III.6).

### 1.3. Biến đổi dữ liệu (vector hóa văn bản)

TODO

## 2. Phân tích dữ liệu

TODO

## 3. Biến đổi và lựa chọn thuộc tính

TODO

## 4. Phân tách dữ liệu (70/30)

- Lý do:
  - Chia dữ liệu thành tập huấn luyện và tập kiểm thử giúp đánh giá hiệu suất của mô hình trên dữ liệu mới. Tập huấn luyện dùng để xây dựng mô hình, tập kiểm thử dùng để đánh giá mô hình trên dữ liệu mà mô hình chưa từng thấy.
  - Việc tách riêng tập kiểm thử cũng giúp phát hiện hiện tượng quá khớp (overfitting).
- Dữ liệu huấn luyện và dữ liệu kiểm thử:
  - Dữ liệu huấn luyện (training data): tập dữ liệu dùng để xây dựng thuật toán, mô hình học máy.
  - Dữ liệu kiểm thử (test data): tập dữ liệu đưa vào sau khi đã xây dựng xong mô hình, nhằm xác nhận tính hiệu quả của mô hình trong thực tế.

TODO

