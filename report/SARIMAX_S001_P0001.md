# 📊 Báo Cáo Kết Quả Mô Hình SARIMAX
**Store:** S001 | **Product:** P0001

---

## 1. Đánh giá Hiệu suất (Performance)

| Metric | SARIMAX (Model) | Naive-7 (Baseline) | Kết quả |
| :--- | :--- | :--- | :--- |
| **RMSE** | **37.590** | 152.076 | Ratio: **0.25** (Target ≤ 0.85) |
| **WAPE** | **22.93%** | - | Kỳ vọng: 10-20% |

**Kết luận:** ĐẠT YÊU CẦU

---

## 2. Thông số Kỹ thuật
* **Model Order:** (0, 0, 0) x (0, 0, 0, 7) (m=7)
* **Thông tin AIC:** 6162.15

---

## 3. Các Yếu tố Ảnh hưởng Chính (Key Drivers)
*Các biến số có trọng số lớn nhất trong việc dự báo:*

|                         |         0 |
|:------------------------|----------:|
| sigma2                  | 1692.8110 |
| Demand Forecast         |  110.2455 |
| Category_Electronics    |   64.1749 |
| Category_Toys           |   63.7173 |
| Category_Groceries      |   61.6333 |
| Category_Furniture      |   60.4188 |
| Weather Condition_Sunny |   55.0905 |
| Weather Condition_Snowy |   55.0590 |
| Region_West             |   53.4692 |
| Weather Condition_Rainy |   49.9020 |
| Region_South            |   48.9619 |
| Region_North            |   48.6014 |
| month                   |    1.2872 |
| is_month_end            |    0.2281 |
| Price_Competitor_Diff   |   -0.1193 |
| is_weekend              |   -0.4570 |
| Discount                |   -0.7563 |
| dow                     |   -1.0632 |
| Holiday/Promotion       |   -1.4858 |
| Price                   |   -2.4123 |

---

## 4. Kết quả Kiểm tra Chất lượng (QC Summary)

### A. Kiểm định Nhiễu trắng (Residuals)
* **Ljung-Box p-value:** `0.5432`
* **Đánh giá:** Phần dư ngẫu nhiên (Tốt)

### B. Kiểm tra Đa cộng tuyến (VIF)
*Bảng dưới đây liệt kê mức độ tương quan giữa các biến (Yêu cầu VIF < 10)*:

| Feature                 |   VIF |
|:------------------------|------:|
| is_weekend              |  2.78 |
| dow                     |  2.77 |
| Region_South            |  1.74 |
| Region_North            |  1.63 |
| Category_Furniture      |  1.62 |
| Category_Groceries      |  1.61 |
| Weather Condition_Snowy |  1.59 |
| Region_West             |  1.58 |
| Weather Condition_Rainy |  1.55 |
| Category_Toys           |  1.51 |
| Weather Condition_Sunny |  1.51 |
| Category_Electronics    |  1.45 |
| Price                   |  1.05 |
| Demand Forecast         |  1.04 |
| Holiday/Promotion       |  1.04 |
| Discount                |  1.03 |
| is_month_end            |  1.03 |
| Price_Competitor_Diff   |  1.02 |
| month                   |  1.01 |

---
*Báo cáo được tạo tự động vào ngày 2025-12-08 14:45:30*
