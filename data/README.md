# Dữ liệu email

- Enron: https://www.kaggle.com/datasets/venky73/spam-mails-dataset, file spam_ham_dataset.csv.
- External: https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset, file completeSpamAssassin.csv.

Chạy `python -m spam_detector.cli train-enron --output runs/<new-name> --raw-dir data/raw` để tải CSV hoặc kiểm tra cache đã có. Cài extra `[training]` trước. Raw data nằm trong data/raw/<source>, có manifest SHA-256. Không sửa raw hoặc thay bằng file khác dưới cùng manifest.

Snapshot nguồn và checksum của lần chạy đã nghiệm thu nằm ở reports/enron-v1/provenance.json. API tải bản latest tại thời điểm chạy; nếu Kaggle đổi phiên bản, muốn tái lập đúng phải giữ CSV khớp checksum đã ghi. Bản nguồn này hiển thị license CC0: Public Domain khi kiểm tra.

Enron sau canonical dedup: 4.894 email. Tập train 2.936, validation 979, test 979. SpamAssassin sau làm sạch còn 5.267 email: loại 5 nội dung không có token, 241 bản trùng cùng nhãn và 533 bản thuộc một nhóm có nhãn mâu thuẫn. Kiểm tra thêm cho thấy nhóm mâu thuẫn có nội dung placeholder `empty`, gồm 335 nhãn spam và 198 ham; toàn bộ nhóm đã bị loại. Không có nhóm canonical trùng Enron trong external còn lại.

Đây là exact canonical dedup (NFKC, casefold và khoảng trắng); chưa xử lý mọi thư gần trùng, thread hoặc sender. Bản ghi và lỗi FP/FN có nội dung email chỉ ở runs cục bộ. Snapshot Git chỉ chứa thống kê, checksum, ID băm và dự đoán validation cần cho ensemble.
