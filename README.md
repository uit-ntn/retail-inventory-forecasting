# Phân tích & Dự báo Số Lượng Bán Ra Theo Ngày cho Bán Lẻ
*(Retail Demand Forecasting – Time Series)*

> **Mục tiêu:** xây dựng hệ thống dự báo **Units Sold** theo *ngày* ở mức hạt mịn **(Store ID, Product ID)** để hỗ trợ hoạch định nhập hàng, giảm thiếu/dư tồn và tối ưu chi phí.

---

## 0. Nhóm thực hiện
| STT | Họ và Tên | Mã số sinh viên | GitHub
|---:|---|---|---|
| 1 | Lê Anh Thư | 23521536 | https://github.com/thule05 |
| 2 | Lê Nguyễn Minh Thư | 23521539 | https://github.com/MinThwMN |
| 3 | Đinh Nhật Thông | 23521522 | https://github.com/AkaDNT |
| 4 | Nguyễn Thanh Nhân | 21521219 | https://github.com/uit-ntn |
| 5 | Mai Lan Anh | 23520052 | https://github.com/23520052 |

---

## 1. Giới thiệu đề tài
Trong kinh doanh bán lẻ đa cửa hàng, dự báo nhu cầu chính xác giúp:
- Giảm rủi ro **thiếu hàng** và **tồn kho dư thừa**,
- Lập kế hoạch **nhập hàng/khuyến mãi/nhân sự** hiệu quả,
- Tối ưu **doanh thu** và **chi phí chuỗi cung ứng**.

Đề tài sử dụng dữ liệu lịch sử bán hàng kết hợp biến ngoại sinh (giá bán, giảm giá, thời tiết, ngày lễ, giá đối thủ, dự báo nội bộ…) để dự báo 7–28 ngày tiếp theo.

---

## 2. Định nghĩa bài toán
- **Đầu vào:** lịch sử `Units Sold` cùng các đặc trưng theo thời gian & ngữ cảnh: `Price`, `Discount`, `Holiday/Promotion`, `Weather Condition`, `Competitor Pricing`, `Demand Forecast`, `Seasonality`, `Region`, `Category`, và **calendar features** (`dow`, `month`, …).
- **Đầu ra:** chuỗi dự báo `Units Sold` cho **horizon = 7/14/28 ngày** tiếp theo.
- **Đánh giá:** so sánh với **baseline Naive-7** và báo cáo các thước đo: **RMSE, MAPE, sMAPE, WAPE**.
- **Ràng buộc dữ liệu:** chỉ sử dụng chuỗi có ≥ **180 ngày** quan sát liên tục.

---

## 3. Dữ liệu
- File mẫu: `retail_store_inventory.csv` (~73.1k dòng • 15 cột • 2022-01-01 → 2024-01-01).
- Các cột chính:  
  `Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Inventory Level`, `Units Sold`, `Units Ordered`, `Demand Forecast`, `Price`, `Discount`, `Weather Condition`, `Holiday/Promotion`, `Competitor Pricing`, `Seasonality`.
- **Tiền xử lý chính:**  
  - Chuẩn hoá thời gian (`Date → datetime`), **reindex theo ngày**, sắp xếp thời gian.  
  - `Units Sold` thiếu → 0; `Inventory Level` → *forward fill*; `Price/Competitor` → *median impute*.  
  - One-hot cho `Weather/Seasonality/Region/Category`; thêm `dow`, `month`.  
  - Với ML/DL: **MinMaxScaler** (fit trên *train*); tạo **window** `lookback=28`, `horizon=7/14`.

---

## 4. Phương pháp sử dụng
- **Baseline:** Naive-7, Holt–Winters (ETS), ARIMA.  
- **Có biến ngoại sinh:** **SARIMAX**.  
- **Học máy:** **XGBoost**, **Random Forest**, **SVR (RBF)** với *lag/rolling* & *calendar features*.  
- **Học sâu:** **LSTM**, **GRU** (multi-var, window-based).  
- **Thực dụng:** **Prophet** (trend/seasonality/holiday nhanh).  
- **Nâng cao (tuỳ chọn):** **Temporal Fusion Transformer (TFT)** cho đa chuỗi với *static/known-future covariates*.

> Khuyến nghị so sánh: **SARIMAX + XGBoost + LSTM/GRU**, sau đó cân nhắc **ensemble**.

---

## 5. Quy trình & Pipeline
1. **EDA & Tiền xử lý** → kiểm tra thiếu/trùng, mùa vụ, dịp lễ, ngoại lệ.  
2. **Tạo đặc trưng** → lag (1/7/14/28), rolling (mean/std/min/max), calendar, one-hot.  
3. **Huấn luyện & Backtesting** → rolling-origin/expanding-window cho horizon 7/14/28.  
4. **Đánh giá** → RMSE/sMAPE/WAPE + baseline Naive-7; diễn giải **SHAP** (ML) hoặc **components/attention** (Prophet/TFT).  
5. **Suy luận & Bàn giao** → lưu model (`.keras`/`.h5`), scaler (`pkl`), dự báo CSV, báo cáo.

### 6. Cấu trúc thư mục gợi ý
```
retail-demand-forecasting/
├─ data/ (raw, processed)
├─ notebooks/  # LSTM/GRU/SARIMAX/XGB/Prophet
├─ models/<algo>/
├─ preds/<algo>/
├─ plots/<algo>/
├─ runs/        # logs, tensorboard
├─ configs/     # grid/optuna yaml
└─ README.md
```

