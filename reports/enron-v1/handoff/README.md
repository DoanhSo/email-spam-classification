# Bàn giao nhánh Decision Tree

- Release dùng chung: `artifacts/tree-enron-v1/`; `release.json` có SHA-256 của cây và bộ biến đổi.
- Cấu hình đã chọn: `configs/tree-enron-selected.yaml`; nguồn/phiên bản dữ liệu đối chiếu provenance.json và data_quality.json của snapshot này.
- `tree.validation.csv`: 979 dòng dự đoán validation, 0=ham, 1=spam. Ghép theo sample_id, không theo thứ tự hàng của mô hình khác. Cột spam_fraction chưa hiệu chuẩn; weighted hard voting dùng y_pred.
- `enron-splits.csv`: manifest ID/nhóm/split Enron đã dùng, không chứa nội dung email. Chạy lại prepare bằng cùng nguồn/checksum để ánh xạ vào records.
- Fold CV trên phần train được ghi trong cv.assignments.csv ở thư mục cha; mỗi fold phải học lại đặc trưng bằng fold-train.
- F1 validation của release cây: 0,8846153846; Dương dùng metric validation tương ứng của từng mô hình để tính trọng số. Không dùng F1 test hoặc external.
- Hiệu có thể gọi `TextPredictor(...).predict_texts(texts)` cho demo; không copy thuật toán/fit lại trong giao diện. Trả nhãn, tỷ lệ ở lá và đường đi để giải thích.

Mô hình tổng quát ngoài Enron kém (external F1=0,44, nhiều false positive), cần trình bày trong báo cáo. Không điều chỉnh lại tham số cây theo external để làm đẹp kết quả đã công bố.
