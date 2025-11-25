# Báo Cáo Phân Tích Mô Hình SARIMAX
## Chuỗi: S001 - P0001
---
### 1. Cấu hình Mô hình Tối ưu
| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| Chuỗi dài | 731 ngày | Yêu cầu $\ge 180$ ngày |
| SARIMA Order | (0, 0, 0) | (p, d, q) |
| Seasonal Order | (0, 0, 0, 7)_7 | (P, D, Q)_s với s=7 |
| Biến Ngoại sinh (Exog) | ['Weather Condition_Cloudy', 'Weather Condition_Rainy', 'Weather Condition_Snowy', 'Weather Condition_Sunny', 'Price', 'Competitor Pricing', 'Discount', 'Demand Forecast', 'Holiday/Promotion', 'dow', 'month'] | Đã được Scale/Encode |
| ADF p-value (d) | 0.0000 (d=0) | Kiểm tra Tính dừng |

### 2. Đánh giá (Validation Set)
| Metric | SARIMAX | Naive-7 Baseline | So sánh tương đối | Kết luận |
| :--- | :--- | :--- | :--- | :--- |
| RMSE | 9.215 | 148.948 | Tỷ lệ: 0.06 | Yêu cầu $\le 0.85 \times \text{RMSE Naive-7}$? Đạt |
| WAPE | 5.959% | 86.374% | Cải thiện: 93.1% | Kỳ vọng $\ge 10-20\%$ |

### 3. Kiểm tra Chẩn đoán (QC)
#### Ljung-Box Test (Kiểm tra White Noise của Residuals)
| Lags | p-value | Kết luận |
| :--- | :--- | :--- |
| 10 | 0.085 | OK |
| 20 | 0.247 | OK |

#### Đa cộng tuyến (VIF)
| feature                  |   VIF |
|:-------------------------|------:|
| Competitor Pricing       | 83.44 |
| Price                    | 83.39 |
| Weather Condition_Cloudy |  2.63 |
| Weather Condition_Snowy  |  2.57 |
| Weather Condition_Sunny  |  2.50 |
| Weather Condition_Rainy  |  2.32 |
| Holiday/Promotion        |  1.02 |
| Discount                 |  1.02 |
| Demand Forecast          |  1.01 |
| dow                      |  1.01 |
| month                    |  1.00 |
*Lưu ý: VIF $\ge 5$ cần được xem xét loại bỏ hoặc kết hợp biến.*
