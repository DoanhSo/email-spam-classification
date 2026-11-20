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
