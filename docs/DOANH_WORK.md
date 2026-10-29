# Kế hoạch triển khai và bàn giao của Nguyễn Quốc Doanh

## Đã hoàn thành trong đợt triển khai này

- [x] Package cài editable, CLI, cấu hình YAML và kiểm tra tham số.
- [x] Interface fit/predict, xác suất/score tùy khả năng cho nhóm.
- [x] Decision Tree NumPy tự viết với Gini có trọng số, điều kiện dừng và kiểm soát overfitting.
- [x] Xuất luật, thống kê cây và xác suất thực nghiệm ở lá.
- [x] Lưu/tải state JSON cùng thứ tự từ và kiểm tra dữ liệu hỏng.
- [x] CLI học, thử độ sâu/kích thước lá, đánh giá và đối chiếu sklearn tùy chọn.
- [x] Kiểm tra ID/nhóm và bộ cột của archive bàn giao; train không đọc test.
- [x] Demo số học giả lập, kiểm thử tự động, tài liệu tích hợp và phần báo cáo cây.
- [x] Workflow CI cho Linux/Python 3.11 và Windows/Python 3.12; kết quả CI được xác nhận riêng sau khi push.

## Thứ tự công việc khi nhóm bắt đầu tích hợp

1. **Hiệu + Dương:** Cung cấp records, nhãn, nhóm trùng lặp và ID split đã chốt.
2. **Minh:** Cung cấp bộ biến đổi binary K=500 và NPZ theo `docs/contracts.md`, chỉ học selector trên train.
3. **Doanh:** Chạy train-tree, xem thời gian và kiểm tra luật với tên từ thật; dùng validation để sửa lỗi cài đặt.
4. **Doanh:** Chạy K=500/1.000 với archive/pipeline tương ứng; thử depth=5/10/15 và leaf=1/5/10. Lưu mọi cấu hình, tránh thử test để chọn.
5. **Doanh + Dương:** Chốt cấu hình cây, giao bundle và dự đoán validation theo ID để Dương tính trọng số ensemble. Hợp nhất evaluator đang dùng vào evaluator nhóm.
6. **Doanh + Minh:** Đóng gói cây với đúng bộ biến đổi văn bản cho release; kiểm tra dự đoán trước/sau tải.
7. **Cả nhóm:** Sau khi đóng băng cấu hình, chạy test và SpamAssassin; Doanh phân tích FP/FN của cây, bổ sung số liệu thật vào báo cáo.
8. **Doanh:** Review bản tích hợp, báo cáo, slide và README; yêu cầu mỗi người tái lập một lần trên máy mình.

## Mốc trong lịch 4 tuần

- Tuần 1: dựng khung và cây chạy bài toán nhỏ; phần này đã có thể nghiệm thu bằng demo hiện tại.
- Tuần 2: ghép đặc trưng thật, so sánh sklearn cùng đầu vào, chốt interface dùng trong demo.
- Tuần 3: thử cấu hình, giao dự đoán validation, chạy đánh giá cuối và phân tích lỗi.
- Tuần 4: hoàn thiện báo cáo/slide, rà soát tích hợp và đóng gói.

## Đang chờ dữ liệu thực tế

Workspace chưa có Enron/SpamAssassin hoặc bộ đặc trưng đã học trên train. Vì vậy chưa có metrics email thực tế, đánh giá ngoài nguồn, kết quả CV hoặc tích hợp đủ bốn mô hình. Các số liệu demo hiện tại chỉ nghiệm thu thuật toán/luồng chạy, không dùng làm kết quả thực nghiệm của đề tài.

## Tiêu chí nghiệm thu tiếp theo

- Nhận đủ feature names, X/y và ID/nhóm từ pipeline thật.
- Không có giao nhau của ID/nhóm giữa tập học/chọn cấu hình và test.
- Chạy lại được từ cấu hình, dữ liệu và hướng dẫn; dự đoán giữ đúng thứ tự ID.
- Có luật từ cây đã học, FP/FN thực tế nếu phát sinh và giải thích giới hạn.
- Bundle cây được demo và ensemble sử dụng đúng release.
