# Decision Tree: phần lý thuyết và cài đặt của Nguyễn Quốc Doanh

**Đã có thực nghiệm email thật:** xem [báo cáo Enron/SpamAssassin](enron-v1/REPORT.md). Phần số liệu giả lập bên dưới được giữ làm nghiệm thu thuật toán, không dùng thay kết quả email thật.

## Mục tiêu

Cây phân loại email thành ham (0) hoặc spam (1) thông qua chuỗi câu hỏi một từ có xuất hiện hay không. Mỗi cột đầu vào là một từ đã được chọn trên train; giá trị 0/1. Cây tự học từ dữ liệu, không đặt sẵn danh sách quy tắc spam bằng tay.

## Cơ sở lý thuyết

Với nút S có tỷ lệ spam p, độ không thuần nhất:

`Gini(S) = 1 - p² - (1-p)² = 2p(1-p)`.

Với một từ tạo hai tập con L và R:

`Gain = Gini(S) - |L|/|S| * Gini(L) - |R|/|S| * Gini(R)`.

Chọn phép chia giảm Gini nhiều nhất. Trọng số số mẫu ngăn một nhánh nhỏ thuần khiến phép chia bị đánh giá quá cao. Với dữ liệu nhị phân chỉ cần ngưỡng 0,5. Không cần sắp xếp và khảo sát mọi ngưỡng số thực.

Cây dừng khi nút thuần, đạt max_depth, không đủ min_samples_split, không tạo được hai nhánh đạt min_samples_leaf hoặc gain <= 1e-12. Lá chọn lớp đông hơn; hòa chọn ham. Xác suất spam tại lá bằng số mẫu spam chia tổng mẫu của lá; đây là ước lượng theo mẫu, chưa hiệu chuẩn.

## Ví dụ kiểm tra bằng tay

X gồm hai cột free/url và bốn hàng `(0,0), (0,1), (1,0), (1,1)`; y tương ứng `0,0,1,1`. Gini gốc là 0,5. Chia theo free tạo hai lá thuần, Gini có trọng số bằng 0 và gain=0,5. Chia theo url tạo hai nhánh vẫn có hai lớp, gain=0. Cây phải chọn free và trả đúng bốn nhãn. Kiểm thử tự động kiểm tra cả dự đoán, xác suất và luật xuất ra.

## Cách cài đặt

Mã học và dự đoán trong `src/spam_detector/models/decision_tree.py` chỉ dùng NumPy, không gọi sklearn.tree. Tại mỗi nút, phép cộng theo cột đếm số mẫu và số spam có từ; từ đó tính impurity cho mọi cột hợp lệ. Dựng cây bằng stack thay vì đệ quy Python. Danh sách Node lưu số mẫu, số spam, độ sâu, cột chia và chỉ số hai con.

Ước lượng thời gian xét phép chia ở một tầng là O(n*d) với n mẫu và d đặc trưng; toàn bộ cây thường O(n*d*h) theo độ sâu h khi phải xét đủ cột ở mỗi tầng. Bộ nhớ bao gồm ma trận đầu vào và các lát cắt ở nút đang xử lý. K=500/1.000 và depth giới hạn là cách thực tế để kiểm soát thời gian với văn bản nhiều chiều.

Nếu có nhiều cột cùng impurity chính xác, chọn cột có index nhỏ nhất để kết quả xác định. sklearn có thể chọn cột hòa khác và có chính sách zero-gain khác, nên không yêu cầu mọi cấu trúc cây trùng hoàn toàn. Cây này chủ động không tách ở gain=0; ví dụ XOR thuần ở gốc sẽ giữ lá. Đây là giới hạn đã được ghi trong kiểm thử và cần nêu khi phản biện.

## Thực nghiệm nghiệm thu thuật toán

Demo giả lập có 4 cột free/url/meeting/money, 120 mẫu train, 40 validation, 40 test; nhãn sinh từ một quy tắc số học cố định. Kết quả chạy ngày 29/10/2026 với cấu hình baseline: cây sâu 3, 7 nút, 4 lá. Validation có TN=25, FP=0, FN=0, TP=15; test có TN=27, FP=0, FN=0, TP=13. Benchmark sklearn trên cùng validation có dự đoán khớp hoàn toàn.

Các kết quả trên chỉ cho thấy thuật toán và luồng lưu/tải/đánh giá hoạt động trên bài toán giả lập. Chúng không ước lượng chất lượng lọc spam trên Enron, SpamAssassin hay email người dùng thực tế.

## Phần bổ sung khi có dữ liệu thật

- Ghi phiên bản nguồn, số mẫu sau làm sạch, split/fingerprint và bộ từ đã chọn.
- Thử K và giới hạn độ sâu/kích thước lá trên validation; ghi Precision/Recall/F1 cùng FP/FN và thời gian.
- Báo cáo CV theo fold với bộ biến đổi fit riêng, đối chiếu sklearn trên cùng X và cấu hình gần tương đương.
- Đánh giá test và nguồn khác sau khi chốt cấu hình; không chọn tham số bằng nhãn test.
- Chọn vài luật có số mẫu hỗ trợ rõ ràng; phân tích thư thật bị chặn nhầm và spam bị bỏ lọt nếu xuất hiện.

## Câu hỏi phản biện cần chuẩn bị

1. Vì sao dùng Gini và phải nhân trọng số số mẫu cho hai nhánh?
2. Vì sao dùng đặc trưng nhị phân, đánh đổi điều gì so với TF-IDF liên tục?
3. max_depth và min_samples_leaf hạn chế học thuộc dữ liệu như thế nào?
4. Tại sao cây đơn dễ thay đổi khi dữ liệu thay đổi? Random Forest có thể giảm phương sai ra sao?
5. Vì sao tỷ lệ spam ở lá không tự động là độ tin cậy đã hiệu chuẩn?
6. Vì sao phải fit Chi-square trên train và học lại trong từng fold?
7. Vì sao một cây greedy không nhận split gain=0 có thể không học được XOR?

Tham khảo đối chiếu interface/điều kiện dừng của thư viện: [scikit-learn DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Cài đặt của đề tài giới hạn ở đặc trưng nhị phân và có quy tắc hòa/zero-gain được công bố riêng.
