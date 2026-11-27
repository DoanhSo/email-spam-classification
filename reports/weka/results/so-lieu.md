# Số liệu ghi từ Weka (đối chiếu với ảnh)

| Mục | Giá trị | Nguồn |
|---|---|---|
| Số bản ghi dữ liệu thô | 5171 (ham 3672, spam 1499) | hinh-01, hinh-03 |
| Thuộc tính bị loại | row_index, label_num | hinh-03, hinh-04 |
| Số bản ghi sau RemoveDuplicates | 4993 (loại 178 bản ghi trùng) | hinh-05 |
| Ham / spam sau khử trùng | 3531 / 1462 (loại 141 ham, 37 spam) | hinh-06 |
| Missing của text | 0 (0%), 4993 giá trị phân biệt | hinh-05 |
| Kiểm tra bằng Python trên enron_dedup.arff | 4993 = 3531 + 1462, khớp | read_text_arff |
| StringToWordVector (-P W_ -W 2000 -C -L -M 2, Rainbow) | 3321 thuộc tính = 3320 từ + class; class chuyển lên vị trí 1 | hinh-07, hinh-08 |
| AttributeSelection (InfoGain + Ranker -N 500) | 501 thuộc tính = 500 từ + class | hinh-09, hinh-10 |
| 2 từ Information Gain cao nhất | W_enron, W_2000 (tiếp theo: W_cc, W_hpl, W_daren, W_http, W_gas) | hinh-10, hinh-11 |
| W_enron | min 0, max 146, mean 1.29, std 4.503 | hinh-11 |
| Randomize | -S 42 | hinh-12 |
| Train (RemovePercentage -P 30) | 3495 (ham 2462, spam 1033) | hinh-13, train.arff |
| Test (RemovePercentage -P 30 -V) | 1498 (ham 1069, spam 429) | hinh-14, test.arff |
| Kiểm tra Python | train ∩ test = 0 email; train ∪ test = enron_dedup (4993) | read_text_arff |

## Naive Bayes Multinomial trên Weka (FilteredClassifier: StringToWordVector -P W_ -W 2000 -C -L -M 2 Rainbow → AttributeSelection InfoGain/Ranker -N 500)

| Cách kiểm thử | Đúng | Accuracy | Kappa | MAE | RMSE | RAE | RRSE | Spam P/R/F | Spam MCC/ROC/PRC | Ma trận (ham: TN FP; spam: FN TP) | Ảnh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Use training set (3495) | 3310 | 94.7067% | 0.8745 | 0.0537 | 0.2087 | 12.8953% | 45.7324% | 0.893/0.933/0.912 | 0.875/0.979/0.912 | 2346 116; 69 964 | hinh-16 |
| 10-fold CV (3495) | 3294 | 94.2489% | 0.8636 | 0.0578 | 0.2176 | 13.8839% | 47.6915% | 0.886/0.924/0.905 | 0.864/0.977/0.904 | 2339 123; 78 955 | hinh-17 |
| Supplied test set (1498) | 1394 | 93.0574% | 0.832 | 0.0674 | 0.2326 | 16.3216% | 51.4359% | 0.865/0.897/0.881 | 0.832/0.974/0.895 | 1009 60; 44 385 | hinh-18 |

Ham (test): P 0.958, R 0.944, F 0.951, FP rate 0.103. Weighted avg F (test) 0.931.

## J48 trên Weka (cùng filter, J48 -C 0.25 -M 2): cây 65 lá, kích thước 129 nút

| Cách kiểm thử | Đúng | Accuracy | Kappa | MAE | RMSE | RAE | RRSE | Spam P/R/F | Spam MCC/ROC/PRC | Ma trận (ham: TN FP; spam: FN TP) | Ảnh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Use training set (3495) | 3339 | 95.5365% | 0.8948 | 0.0654 | 0.1808 | 15.6937% | 39.6176% | 0.898/0.957/0.927 | 0.896/0.991/0.973 | 2350 112; 44 989 | hinh-19 |
| 10-fold CV (3495) | 3252 | 93.0472% | 0.8354 | 0.0838 | 0.2331 | 20.1311% | 51.0833% | 0.864/0.907/0.885 | 0.836/0.970/0.932 | 2315 147; 96 937 | hinh-20 |
| Supplied test set (1498) | 1376 | 91.8558% | 0.8085 | 0.0971 | 0.2428 | 23.5389% | 53.7053% | 0.815/0.925/0.867 | 0.812/0.968/0.912 | 979 90; 32 397 | hinh-21 |

Gốc cây (hinh-22): W_2004 > 0 → spam (86); W_prescription > 0 → spam (71); W_meds → spam (47); W_sex → spam (46); W_paliourg → spam (39); W_dealer → spam (36); W_php ...

## Đối chiếu NB tự viết (Python) với NaiveBayesMultinomial (Weka)

Vector do Weka batch filtering tạo (train_vec/test_vec, 500 từ). Python học trên train_vec, dự đoán test_vec.
Kết quả: trùng 1498/1498 dự đoán; chênh lệch xác suất lớn nhất 0.000498 (Weka in 3 chữ số thập phân); ma trận nhầm lẫn giống hệt (TN 1009, FP 60, FN 44, TP 385). Nguồn: reports/bao-cao/results/nb_python_vs_weka.json

## Kiểm định giả thuyết (Weka Experimenter, train.arff, 10 lần × 10-fold CV, PairedCorrectedTTester, α = 0.05 hai phía, IR stats cho lớp spam: -C 2)

| Độ đo | NB (base) | J48 | Weka (v/ /*) | p (tính lại, hai phía) |
|---|---|---|---|---|
| F_measure | 0.9052 ± 0.0173 | 0.8876 ± 0.0244 | (0/1/0) không có ý nghĩa | 0.072 |
| Percent_correct | 94.27 ± 1.09 | 93.16 ± 1.49 | (0/1/0) không có ý nghĩa | 0.058 |
| IR_precision | 0.8869 ± 0.0262 | 0.8634 ± 0.0303 | (0/0/1) * J48 kém hơn có ý nghĩa | 0.037 |
| Area_under_ROC | 0.9764 ± 0.0071 | 0.9681 ± 0.0124 | (chưa chạy trong GUI) | 0.079 |
| IR_recall | 0.9250 ± 0.0214 | 0.9140 ± 0.0322 | (chưa chạy trong GUI) | 0.426 |

Kết luận: không bác bỏ H0 về F-Measure ở α = 0.05 (p một phía ≈ 0.036). Precision của NB cao hơn J48 có ý nghĩa.
Ghi chú: Experimenter GUI báo "Access is denied" khi ghi experiment.arff; thí nghiệm được chạy bằng weka.jar CLI với cùng cấu hình (config_nb.txt, config_j48.txt, -C 2), kết quả gộp vào experiment.arff và phân tích trong tab Analyse (hinh-24).
