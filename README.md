# Hệ thống phát hiện email spam/rác

Bài tập lớn môn **Khai phá dữ liệu**, Trường Đại học Thủy Lợi. Giảng viên: TS. Trần Mạnh Tuấn.

Phân loại email tiếng Anh thành **ham** (thư hợp lệ) và **spam** (thư rác) bằng **Multinomial Naive Bayes** (mô hình chính) và **cây quyết định J48** (mô hình so sánh). Quy trình khai phá dữ liệu làm trên **Weka 3.8.7**; thuật toán Naive Bayes được **tự cài đặt bằng Python** và đối chiếu với Weka.

## Nhóm 12 – Lớp 65CNTT

| STT | Họ tên | MSSV |
|---|---|---|
| 1 | Nguyễn Quốc Doanh (nhóm trưởng) | 2351060429 |
| 2 | Nguyễn Thái Dương | 2351060435 |
| 3 | Lê Hữu Hiệu | 2351060446 |
| 4 | Thân Đức Minh | 2351060467 |

## Sản phẩm

| Sản phẩm | Vị trí |
|---|---|
| Báo cáo Word (42 trang) | [reports/bao-cao/BaoCao_Nhom12_SpamEmail.docx](reports/bao-cao/BaoCao_Nhom12_SpamEmail.docx) |
| Báo cáo PDF | [reports/bao-cao/BaoCao_Nhom12_SpamEmail.pdf](reports/bao-cao/BaoCao_Nhom12_SpamEmail.pdf) |
| Đối chiếu với báo cáo mẫu | [reports/bao-cao/REVIEW.md](reports/bao-cao/REVIEW.md) |
| Hướng dẫn thao tác Weka (24 hình) | [reports/weka/CHECKLIST.md](reports/weka/CHECKLIST.md) |
| Số liệu Weka đã ghi lại | [reports/weka/results/so-lieu.md](reports/weka/results/so-lieu.md) |
| Thiết kế và kế hoạch | [docs/superpowers/specs/](docs/superpowers/specs/2026-11-20-bao-cao-btl-spam-design.md), [docs/superpowers/plans/](docs/superpowers/plans/2026-11-20-bao-cao-btl-spam.md) |
| Demo kiểm tra email | `app/streamlit_app.py` |

## Kết quả chính

Dữ liệu: bộ **Enron-Spam** (Kaggle, 5171 email). Sau khi loại 178 bản ghi trùng còn **4993 email** (3531 ham, 1462 spam), chia **70/30** thành 3495 email huấn luyện và 1498 email kiểm thử.

| Mô hình (Weka) | Use training set | Cross-validation 10 | Tập kiểm thử | F-Measure spam (kiểm thử) |
|---|---|---|---|---|
| Naive Bayes Multinomial | 94,71% | 94,25% | **93,06%** | **0,881** |
| J48 | 95,54% | 93,05% | 91,86% | 0,867 |

- **Đối chiếu:** Naive Bayes tự viết bằng Python cho **cùng dự đoán với Weka trên 1498/1498 email** kiểm thử.
- **Kiểm định giả thuyết** (Weka Experimenter, 10 lần × 10-fold CV, corrected paired t-test, α = 0,05): Naive Bayes có kết quả trung bình cao hơn J48 ở mọi độ đo. Precision của lớp spam khác biệt có ý nghĩa (p = 0,037), F-Measure chưa đủ ý nghĩa (p = 0,072), nên chưa bác bỏ H0.
- **Kiểm thử nguồn khác (SpamAssassin):** độ chính xác giảm còn khoảng 44% do lệch miền dữ liệu. Mô hình chỉ phù hợp với thư cùng nguồn với dữ liệu huấn luyện.

## Cấu trúc thư mục

```
app/streamlit_app.py            Demo kiểm tra email (Naive Bayes mặc định)
src/spam_detector/
  models/naive_bayes.py         Multinomial Naive Bayes tự viết (NumPy)
  models/decision_tree.py       Cây quyết định tự viết (NumPy)
  weka_io.py                    Ghi/đọc ARFF cho Weka
  weka_compare.py               Đối chiếu NB Python với dự đoán của Weka
  report_experiments.py         Thực nghiệm Python trên tập 70/30 của Weka và SpamAssassin
  nb_release.py, inference.py   Đóng gói mô hình và dự đoán từ văn bản
  web_service.py, cli.py        Dịch vụ cho demo và dòng lệnh
artifacts/                      Mô hình đã huấn luyện (nb-weka7030-v1, tree-weka7030-v1)
configs/web.yaml                Cấu hình mô hình cho demo
data/raw/                       CSV Enron và SpamAssassin
reports/weka/                   Dữ liệu ARFF, ảnh và kết quả Weka
reports/bao-cao/                Nội dung báo cáo (Markdown), hình, script dựng Word
tests/                          Kiểm thử tự động (pytest)
```

## Cài đặt

Yêu cầu Python ≥ 3.11 (đã kiểm tra với 3.12 trên Windows).

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\python -m pip install -e ".[dev,web,report]"
```

```powershell
.venv\Scripts\python -m pytest -q
```

Linux/macOS thay `.venv\Scripts\python` bằng `.venv/bin/python`.

## Chạy demo

```powershell
.venv\Scripts\python -m streamlit run app/streamlit_app.py --server.port 8501
```

Mở http://localhost:8501, nhập tiêu đề và nội dung email tiếng Anh, bấm **Kiểm tra email**. Kết quả gồm nhãn Spam/Thư thường, xác suất spam và các từ đóng góp nhiều nhất vào kết luận. Email không chứa từ nào trong bộ từ vựng sẽ báo **"Không dự đoán được"**. Có thể chọn cây quyết định hoặc so sánh hai mô hình ở thanh bên.

## Tái lập kết quả

**1. Xuất dữ liệu cho Weka**

```powershell
.venv\Scripts\python -m spam_detector.cli export-arff --csv data/raw/enron/spam_ham_dataset.csv --output reports/weka/data/enron_raw.arff
```

**2. Thao tác trên Weka** theo [CHECKLIST.md](reports/weka/CHECKLIST.md): làm sạch, vector hóa, chọn thuộc tính, chia 70/30, chạy NB/J48 và Experimenter. Kết quả lưu trong `reports/weka/data` và `reports/weka/results`.

**3. Đối chiếu Naive Bayes tự viết với Weka** (dùng vector do Weka batch filtering tạo ra):

```powershell
.venv\Scripts\python -m spam_detector.cli compare-weka --train-vec reports/weka/data/train_vec.arff --test-vec reports/weka/data/test_vec.arff --weka-pred reports/weka/results/nb_test_predictions.csv --output reports/bao-cao/results
```

**4. Thực nghiệm Python** (huấn luyện NB và cây trên tập 70/30, đánh giá trên tập kiểm thử và SpamAssassin, tạo mô hình cho demo):

```powershell
.venv\Scripts\python -m spam_detector.cli report-run --train reports/weka/data/train.arff --test reports/weka/data/test.arff --external data/raw/spamassassin/completeSpamAssassin.csv --output reports/bao-cao/results --artifacts artifacts
```

**5. Dựng báo cáo Word** từ các tệp `reports/bao-cao/content/*.md`:

```powershell
.venv\Scripts\python reports/bao-cao/build_docx.py
```

Sau khi mở file .docx trong Word, chuột phải vào mục lục → **Update Field** để cập nhật số trang.

## Ghi chú

- Các bước vector hóa và chọn thuộc tính được đặt trong `FilteredClassifier` khi đánh giá, nên từ vựng chỉ học trên tập huấn luyện (không rò rỉ dữ liệu kiểm thử).
- Bước xuất vector (batch filtering) và thí nghiệm Experimenter được chạy bằng `weka.jar` dòng lệnh với cùng cấu hình như giao diện, vì Experimenter GUI báo "Access is denied" khi ghi tệp kết quả.
- Mô hình tự viết chỉ dùng NumPy; scikit-learn chỉ dùng cho bước tách từ và chọn đặc trưng của demo.
- `KE_HOACH_NHOM.md`, `THIET_KE_KHUNG_DU_AN.md` và các tài liệu trong `docs/` (trừ `docs/superpowers/`) là kế hoạch phạm vi cũ (4 mô hình + ensemble), giữ lại làm lịch sử.
