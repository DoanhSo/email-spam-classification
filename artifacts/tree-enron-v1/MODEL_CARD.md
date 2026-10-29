# Decision Tree Enron v1

Người phụ trách: Nguyễn Quốc Doanh. Thuật toán NumPy tự viết, đặc trưng nhị phân unigram chọn bằng Chi-square trên train. Không có cây sklearn trong luồng dự đoán.

- Train: 2.936 email Enron sau canonical dedup.
- Validation: 979, dùng chọn K/tham số; không refit bằng train+validation.
- Tham số: K=1000, depth tối đa 15, min_samples_split=10, min_samples_leaf=1.
- Cây thực: 97 nút, 49 lá; hòa nhãn chọn ham, tỷ lệ spam lá >0,5 chọn spam.
- Test Enron 979 mẫu: Precision 82,16%, Recall 96,23%, F1 88,64%, FP=61, FN=11.
- External SpamAssassin 5.267 mẫu: Precision 29,04%, Recall 90,77%, F1 44,00%, FP=3006, FN=125.
- CV 5-fold trên train của cấu hình được chọn: F1 mean 88,84%, sample std 1,49 điểm phần trăm. CV là chẩn đoán sau lựa chọn, không phải nested CV.

Hạn chế chính: tổng quát ngoài nguồn thấp, chặn nhầm nhiều ham. Cây dùng dấu hiệu đặc thù Enron; không dùng release này để tự động chặn email thật. Tỷ lệ lớp ở lá chưa hiệu chuẩn. Chưa kiểm tra email hiện đại, tiếng Việt hoặc bảo đảm tách hoàn toàn các thư gần trùng/thread/sender.

Tải bằng `spam_detector.inference.TextPredictor('artifacts/tree-enron-v1')`; gọi predict_texts([...]). Loader kiểm tra checksum mô hình/đặc trưng, format và phiên bản preprocessing. Xem reports/enron-v1/REPORT.md và docs/contracts.md để tích hợp.
