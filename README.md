# Email Spam Classification

Bài tập lớn môn Khai phá dữ liệu: phát hiện email spam tiếng Anh bằng bốn mô hình tự cài đặt và kết hợp bỏ phiếu có trọng số.

## Thành viên và phân công

- **Nguyễn Quốc Doanh:** Nhóm trưởng, Decision Tree, kiến trúc và nghiệm thu tích hợp.
- **Lê Hữu Hiệu:** Naive Bayes, dữ liệu, tiền xử lý, EDA và giao diện demo.
- **Thân Đức Minh:** Logistic Regression, biểu diễn và lựa chọn đặc trưng.
- **Nguyễn Thái Dương:** Linear SVM, đánh giá thực nghiệm và ensemble.

## Tài liệu dự án

- [Kế hoạch nhóm trong 4 tuần](KE_HOACH_NHOM.md): phạm vi, phân công, tiến độ và đầu ra từng giai đoạn.
- [Thiết kế khung toàn bộ dự án](THIET_KE_KHUNG_DU_AN.md): kiến trúc, cấu trúc thư mục dự kiến, hợp đồng dữ liệu/mô hình, huấn luyện, đánh giá, lưu pipeline, demo và kiểm thử.

## Trạng thái

Repository hiện chứa tài liệu lập kế hoạch và thiết kế. Mã nguồn, dữ liệu, môi trường chạy và mô hình sẽ được bổ sung theo các mốc nghiệm thu; các lệnh trong tài liệu thiết kế là giao diện dự kiến, chưa chạy được ở trạng thái hiện tại.

Lộ trình dự kiến gồm 4 tuần, tạm tính từ 29/10 đến 25/11/2026, dự phòng 26-28/11. Hạn nộp chính thức sẽ được cập nhật theo thông báo môn học.

## Phạm vi triển khai

- Dữ liệu chính: Enron-Spam; đánh giá trên nguồn khác bằng SpamAssassin.
- Mô hình tự viết: Multinomial Naive Bayes, Logistic Regression, Linear SVM và Decision Tree.
- Ensemble: bỏ phiếu theo nhãn, trọng số từ F1 validation.
- Dự kiến dùng Python, NumPy/SciPy, pandas và Streamlit; thư viện mô hình dựng sẵn chỉ dùng trong thực nghiệm đối chiếu.
- Random Forest và các mở rộng ngoài email tiếng Anh để ở hướng phát triển.

## Bắt đầu phối hợp

1. Đọc bản thiết kế và chốt interface, quy ước nhãn, schema dữ liệu.
2. Dựng package và môi trường dùng chung.
3. Hoàn thành một luồng học thật từ email đến metrics và lưu mô hình.
4. Tích hợp lần lượt bốn mô hình, sau đó ensemble và demo.

Dữ liệu đầy đủ, môi trường ảo, mô hình đã học và đầu ra thực nghiệm lớn được loại khỏi Git. Khi triển khai, bổ sung hướng dẫn tải dữ liệu và phiên bản dependencies để tái lập kết quả.
