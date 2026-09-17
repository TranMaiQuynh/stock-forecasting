
# Tài Liệu Dataset — StockForecasting

> Tài liệu này mô tả toàn diện nguồn gốc, đặc trưng, pipeline xử lý, phân tích thống kê thực nghiệm và mối liên kết logic giữa đặc tính dữ liệu với kiến trúc mô hình (Data → Model Bridge).
> 🔗 Xem phần mô hình tại: [modeling.md](./modeling.md)
> 🔗 Xem visualization trực quan tại: [eda.html](./eda.html)
> 🔗 Xem dashboard kết quả kiểm định tại: [results.html](./results.html)

---

## 1. Nguồn Gốc Dữ Liệu

### 1.1. Nguồn thu thập

Toàn bộ dữ liệu được tải từ **Yahoo Finance** thông qua thư viện Python `yfinance`. Yahoo Finance cung cấp dữ liệu lịch sử giao dịch chứng khoán được điều chỉnh (Adjusted) theo cổ tức và tách cổ phiếu (stock split) — đảm bảo tính liên tục và bảo toàn giá trị kinh tế theo thời gian.

```
Nguồn:      Yahoo Finance (https://finance.yahoo.com/)
Thư viện:   yfinance >= 0.2.x
Hàm tải:    preprocessing/run_pipeline.py → download_yahoo_data()
Lưu trữ:    data/{TICKER}_{start}_{end}.csv
```

### 1.2. Danh mục tài sản thực nghiệm (23 chuỗi thời gian)

Dự án mở rộng thực nghiệm trên **23 mã cổ phiếu và chỉ số vĩ mô** đại diện cho nhiều phân khúc kinh tế:

| Phân nhóm ngành                  | Mã tài sản | Tên đầy đủ / Ý nghĩa | Vai trò trong hệ thống                      |
| :---------------------------------- | :------------ | :-------------------------- | :--------------------------------------------- |
| **Mega-cap Tech**             | `AAPL`      | Apple Inc.                  | Target chính (Benchmark cốt lõi)            |
|                                     | `MSFT`      | Microsoft Corp.             | Cross-ticker ablation (High growth & SaaS)     |
|                                     | `GOOGL`     | Alphabet Inc.               | Tech & Advertising                             |
|                                     | `AMZN`      | Amazon.com Inc.             | E-commerce & Cloud infrastructure              |
|                                     | `META`      | Meta Platforms Inc.         | Social media & AI                              |
| **Bán dẫn & Phần cứng**   | `NVDA`      | NVIDIA Corporation          | Cổ phiếu biến động cực cao (AI hardware) |
|                                     | `INTC`      | Intel Corporation           | Chu kỳ bán dẫn truyền thống               |
|                                     | `IBM`       | IBM Corporation             | Enterprise IT                                  |
| **Tài chính - Ngân hàng** | `JPM`       | JPMorgan Chase & Co.        | Ngân hàng thương mại lớn nhất Mỹ       |
|                                     | `BAC`       | Bank of America Corp.       | Nhạy cảm với chu kỳ lãi suất FED         |
|                                     | `MA`        | Mastercard Inc.             | Thanh toán & FinTech                          |
| **Tiêu dùng & Bán lẻ**    | `COST`      | Costco Wholesale Corp.      | Bán lẻ thiết yếu ổn định                |
|                                     | `PEP`       | PepsiCo Inc.                | Hàng tiêu dùng phòng thủ                  |
|                                     | `KO`        | The Coca-Cola Company       | Cổ tức cao, độ biến động thấp          |
|                                     | `MCD`       | McDonald's Corp.            | F&B toàn cầu                                 |
|                                     | `NKE`       | Nike Inc.                   | Hàng tiêu dùng không thiết yếu           |
| **Y tế & Dược phẩm**      | `JNJ`       | Johnson & Johnson           | Cổ phiếu y tế phòng thủ                   |
|                                     | `LLY`       | Eli Lilly and Company       | Dược phẩm tăng trưởng cao (GLP-1)        |
|                                     | `ABBV`      | AbbVie Inc.                 | Dược phẩm sinh học                         |
| **Enterprise Software / EV**  | `ORCL`      | Oracle Corporation          | Cloud & Database                               |
|                                     | `TSLA`      | Tesla Inc.                  | Ablation cho tài sản siêu biến động      |
| **Chỉ số Vĩ mô (Macro)**  | `^VIX`      | CBOE Volatility Index       | Feature vĩ mô: "Chỉ số sợ hãi" phố Wall |
|                                     | `^TNX`      | 10-Year US Treasury Yield   | Feature vĩ mô: Áp lực chi phí vốn        |

### 1.3. Khoảng thời gian thu thập

```
Dải thời gian:      2018-01-01 → 2026-12-31 (~8.5 năm)
Số phiên thực tế:   2,169 phiên (AAPL) đến 2,188 phiên (các mã khác)
Phân chia:          Train (70%) / Val (15%) / Test (15%) theo thứ tự thời gian
Live Inference:     Lấy dữ liệu phiên gần nhất để dự báo phiên kế tiếp (t+1)
```

### 1.4. ⚠️ Giới hạn phạm vi áp dụng & Cảnh báo thị trường

> **QUAN TRỌNG:** Dữ liệu này thu thập từ thị trường chứng khoán Mỹ (US Equity Market: NYSE / NASDAQ).

**Tuyệt đối KHÔNG áp dụng nguyên trạng mô hình này cho thị trường chứng khoán Việt Nam (VN-Index, HNX)** vì:

1. **Biên độ dao động trần/sàn:** Thị trường Việt Nam có biên độ giá khống chế (HoSE: ±7%, HNX: ±10%, UPCoM: ±15%), trong khi thị trường Mỹ không có biên độ giá ngày (chỉ có Circuit Breaker tạm ngắt). Phân phối lợi suất thực tế của hai thị trường có cấu trúc hoàn toàn khác nhau.
2. **Chu kỳ thanh toán:** Thị trường Mỹ áp dụng chu kỳ T+1 settlement, trong khi Việt Nam có chu kỳ T+2.5 với cơ chế khớp lệnh định kỳ mở/đóng cửa (ATO/ATC) đặc thù.
3. **Đặc tính vĩ mô:** Các biến ngoại sinh như `^VIX` và `^TNX` đại diện trực tiếp cho thị trường tiền tệ và tâm lý nhà đầu tư Mỹ. Với thị trường Việt Nam, các biến vĩ mô cần thay thế bằng: Lãi suất liên ngân hàng (VNIBOR), Tỷ giá USD/VND, và Chỉ số VN30 / VN-Index.

---

## 2. Quy Trình Tạo Dữ Liệu (Data Pipeline)

```
[Yahoo Finance API]
     │
     ▼ (1) yfinance.download()
[Raw OHLCV DataFrame]  ── (Open, High, Low, Close, Volume)
     │
     ▼ (2) calculate_technical_indicators()
[OHLCV + 20 Indicators] ── Trend (MACD, SMA, EMA), Momentum (RSI, Stoch), Volatility (BB, ATR)
     │
     ▼ (3) merge_macro_data()
[Stock + VIX + TNX]    ── Đồng bộ timestamp, ffill/bfill các ngày nghỉ lệch pha
     │
     ▼ (4) Feature Engineering Dẫn Xuất
[Features Matrix]      ── Thêm ATR_norm, Delta_VIX, Log_Return
     │
     ▼ (5) Time-based Train/Val/Test Split (70% / 15% / 15%)
[Train / Val / Test]   ── Không xáo trộn (shuffle=False) để bảo toàn tính nhân quả thời gian
     │
     ▼ (6) clip_outliers() — Winsorization [1%, 99%] CHỈ tính trên Train
[Train (Clipped)]      ── Val & Test giữ nguyên hoặc clip theo ngưỡng của Train
     │
     ▼ (7) select_orthogonal_features()
[10 Orthogonal Cols]   ── Giảm thiểu đa cộng tuyến (Multicollinearity)
     │
     ▼ (8) Scaling (Zero Data Leakage)
[Scaled Arrays]        ── RobustScaler (Features) + StandardScaler (Target) fit CHỈ trên Train
     │
     ▼ (9) Sliding Window Dataset Generation (window=60, horizon=1)
[PyTorch Tensors]      ── Shape: (batch_size, 60, num_features)
     │
     ▼ (10) DataLoader Batching
[Model Input Tensor]
```

### Nguyên tắc bất biến: Zero Data Leakage

Mọi giá trị thống kê (ngưỡng Winsorization 1% và 99%, Median và IQR của `RobustScaler`, Mean và Std của `StandardScaler`) **chỉ được học từ tập Train**. Tập Val và Test chỉ được gọi `transform()` dựa trên tham số đã đóng băng từ Train. Không một thông tin tương lai nào được phép lọt vào quá khứ.

---

## 3. Danh Mục Đặc Trưng (Features) & Ý Nghĩa Tài Chính

### 3.1. Dữ liệu gốc OHLCV

- `Open`: Giá mở cửa (USD).
- `High`: Giá cao nhất phiên (USD).
- `Low`: Giá thấp nhất phiên (USD).
- `Close`: Giá đóng cửa điều chỉnh (Adjusted Close, USD).
- `Volume`: Khối lượng giao dịch thực tế (Cổ phiếu).

### 3.2. Nhóm Xu Hướng (Trend)

- `SMA_10`, `SMA_20`, `SMA_50`: Đường trung bình động đơn giản 10, 20, 50 phiên.
- `EMA_12`, `EMA_26`: Trung bình động hàm mũ nhạy cảm với biến động gần.
- `MACD`: Hiệu số $(EMA_{12} - EMA_{26})$.
- `MACD_Signal`: $EMA_9(MACD)$.
- `MACD_Hist`: $(MACD - MACD\_Signal)$ — **Đại diện trực giao cho động lực xu hướng**.

### 3.3. Nhóm Động Lượng (Momentum)

- `RSI_14`: Relative Strength Index 14 ngày (vùng quá mua > 70, quá bán < 30) — **Đại diện trực giao cho xung lượng giá**.
- `Stoch_K`, `Stoch_D`: Vị trí tương đối của giá đóng cửa trong biên độ giá 14 ngày.
- `ROC_10`: Rate of Change 10 ngày.

### 3.4. Nhóm Biến Động (Volatility)

- `BB_Upper`, `BB_Lower`, `BB_Bandwidth`: Dải Bollinger 20 phiên.
- `ATR_14`: Average True Range 14 ngày (đo biên độ rủi ro tuyệt đối).
- `ATR_norm`: $\frac{ATR_{14}}{Close}$ — **Đại diện trực giao cho biến động tương đối** (cho phép so sánh giữa các thời kỳ giá khác nhau).
- `Hist_Volatility_20`: Độ lệch chuẩn lợi suất 20 phiên nhân hóa năm $(\times \sqrt{252})$.

### 3.5. Nhóm Khối Lượng & Dòng Tiền (Volume)

- `Volume_SMA20`: Khối lượng trung bình 20 phiên.
- `Volume_Ratio`: $\frac{Volume}{Volume\_SMA20}$ — **Đại diện trực giao cho đột biến thanh khoản** (Volume breakout).
- `OBV`: On-Balance Volume đo dòng tiền tích luỹ.

### 3.6. Biến Mục Tiêu và Ngoại Sinh Vĩ Mô

- `Log_Return`: $\ln(P_t / P_{t-1})$ — **Biến mục tiêu dự báo (Target)**.
- `Delta_VIX`: $\Delta VIX = VIX_t - VIX_{t-1}$ — Xung lực sợ hãi thị trường.
- `Macro_TNX`: Lợi suất trái phiếu chính phủ Mỹ kỳ hạn 10 năm — Đại diện chi phí vốn phi rủi ro.

---

## 4. Phân Tích Thực Nghiệm Thống Kê (Empirical Statistics)

### 4.1. Phân tích mô tả chuẩn xác cho mã mục tiêu AAPL (2018–2026)

*(Dữ liệu tính toán trực tiếp từ `df.describe()` trên 2,169 phiên giao dịch)*

| Chỉ số thống kê                  | Giá mở cửa (Open) | Giá cao nhất (High) | Giá thấp nhất (Low) | Giá đóng cửa (Close) | Khối lượng (Volume) |
| :----------------------------------- | -------------------: | --------------------: | ---------------------: | -----------------------: | ---------------------: |
| **Số quan sát (Count)**      |                2,169 |                 2,169 |                  2,169 |                    2,169 |                  2,169 |
| **Trung bình (Mean)**         |              $145.74 |               $147.37 |                $144.26 |        **$145.89** |             91,248,890 |
| **Độ lệch chuẩn (Std)**    |               $74.95 |                $75.74 |                 $74.26 |         **$75.05** |             54,029,640 |
| **Giá trị nhỏ nhất (Min)** |               $34.13 |                $34.54 |                 $33.66 |         **$33.71** |             17,910,600 |
| **Tứ phân vị 25% (Q1)**     |               $69.31 |                $70.74 |                 $68.93 |         **$69.76** |             52,517,000 |
| **Trung vị (Median / 50%)**   |              $146.13 |               $147.93 |                $144.95 |        **$146.51** |             76,957,800 |
| **Tứ phân vị 75% (Q3)**     |              $194.92 |               $197.44 |                $192.53 |        **$195.19** |            111,943,300 |
| **Giá trị lớn nhất (Max)** |              $339.74 |               $344.27 |                $337.06 |        **$339.79** |            426,510,000 |

### 4.2. Thống kê biến mục tiêu `Log_Return` của AAPL

- **Số mẫu:** 2,168 phiên
- **Trung bình ngày (Mean):** $+0.000952$ ($+0.0952\%$/ngày, tương đương $+24.0\%$ lợi suất kép hàng năm)
- **Độ lệch chuẩn (Std):** $0.019238$ ($1.92\%$/ngày, biến động hàng năm $30.54\%$)
- **Biên độ cực tiểu (Min):** $-0.137708$ ($-13.77\%$, sụp đổ thị trường tháng 03/2020)
- **Biên độ cực đại (Max):** $+0.142618$ ($+14.26\%$)
- **Hệ số bất đối xứng (Skewness):** $-0.1192$ (Lệch nhẹ về đuôi âm — rủi ro sụt giảm đột ngột cao hơn khả năng tăng sốc)
- **Hệ số nhọn vượt chuẩn (Excess Kurtosis):** $+6.0716$ (**Leptokurtic / Fat Tails** — xác suất xảy ra các sự kiện ngoại lai 3-sigma cao gấp nhiều lần so với phân phối Gauss chuẩn)

### 4.3. Phân tích cân bằng nhãn (Class Balance)

Dù dự án giải quyết bài toán hồi quy (Regression), việc phân loại hướng đi giá phiên kế tiếp cho thấy:

- **Số ngày Tăng ($Log\_Return > 0$):** 1,161 phiên (**53.55%**)
- **Số ngày Giảm ($Log\_Return < 0$):** 1,003 phiên (**46.26%**)
- **Số ngày Đi ngang ($Log\_Return = 0$):** 4 phiên (**0.18%**)

> **Đánh giá:** Tỷ lệ xấp xỉ $53.6\% / 46.3\%$ phản ánh xu thế tăng giá dài hạn (Upward Drift) của cổ phiếu vốn hóa lớn, đồng thời thể hiện tính ngẫu nhiên cao (Random Walk). Dữ liệu hoàn toàn cân bằng tự nhiên, không bị mất cân bằng trầm trọng (imbalance), do đó không cần kỹ thuật oversampling nhân tạo (như SMOTE) vốn dễ gây bóp méo tương quan chuỗi thời gian.

---

## 5. Kiểm Định Thống Kê & Xử Lý Dữ Liệu

### 5.1. Kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF Test)

Kiểm định giả thuyết không $H_0$: Chuỗi có nghiệm đơn vị (Unit Root, không dừng).

| Chuỗi thời gian          | ADF Test Statistic ($t$-stat) | Ngưỡng tới hạn 1% | Ngưỡng tới hạn 5% | Kết luận thống kê ($p$-value) |           |           |                                                                                                                                |
| :------------------------- | --------------------------------------------------------------------------------------------------------------------: | --------: | --------: | :----------------------------------------------------------------------------------------------------------------------------- |
| **AAPL Close Price** |                                                                                                 **$-0.0269$** | $-3.43$ | $-2.86$ | **Không thể bác bỏ $H_0$ ($p > 0.95$)** → **Chuỗi không dừng (Non-Stationary)**                        |
| **AAPL Log_Return**  |                                                                                                **$-33.2739$** | $-3.43$ | $-2.86$ | **Bác bỏ $H_0$ với độ tin cậy $99.99\%$ ($p < 10^{-15}$)** → **Chuỗi dừng hoàn hảo (Stationary)** |

> **Ý nghĩa khoa học:** Kiểm định ADF chứng minh toán học rằng việc dự báo trực tiếp trên `Close Price` vi phạm giả định cơ bản về tính ổn định phân phối của mô hình học máy. Một mô hình huấn luyện trên chuỗi không dừng sẽ có xu hướng rơi vào bẫy **Lag Fallacy** (học hàm đồng nhất $\hat{y}_{t+1} \approx y_t$ để cực tiểu hóa MSE mà không học được bất kỳ tín hiệu thị trường nào). Ngược lại, `Log_Return` có tính dừng tuyệt đối, buộc mạng nơ-ron phải học quan hệ nhân quả thực sự.

### 5.2. Định lượng hiệu quả của Winsorization [1%, 99%] trên tập Train

Trên tập Train (1,483 phiên đầu tiên), các ngưỡng bách phân vị được xác định:

- Ngưỡng bách phân vị 1%: **$-5.37\%$**
- Ngưỡng bách phân vị 99%: **$+5.01\%$**

| Chỉ tiêu thống kê             |                         Trước Winsorization |                            Sau Winsorization [1%, 99%] | Thay đổi thực nghiệm |
| :-------------------------------- | --------------------------------------------: | -----------------------------------------------------: | :----------------------- |
| **Min Value**               |          $-13.77\%$ | **$-5.37\%$** |    Triệt tiêu các cú sốc thiên nga đen cô lập |                          |
| **Max Value**               |          $+11.32\%$ | **$+5.01\%$** |            Hạn chế gradient explosion trong Backprop |                          |
| **Độ lệch chuẩn (Std)** |           $0.01994$ | **$0.01817$** |              Giảm thiểu phương sai nhiễu$8.9\%$ |                          |
| **Excess Kurtosis**         | **$+5.0545$** | **$+0.8644$** | **Đưa phân phối về rất gần chuẩn Gauss** |                          |

---

## 6. Bảng Thống Kê So Sánh Toàn Bộ 23 Mã Cổ Phiếu & Chỉ Số

*(Toàn bộ dải dữ liệu 2018–2026 trong dataset)*

| Ticker          | Số phiên | Từ ngày  | Đến ngày | Giá TB ($) | Độ lệch chuẩn ($) | Biến động năm (%) | Skewness | Kurtosis | ADF$t$-stat | Tỷ lệ Tăng/Giảm (%) |             |
| :-------------- | ---------: | :--------- | :---------- | ------------------------------------: | --------------------: | -------: | -------: | ------------: | ----------------------: | :---------- |
| **AAPL**  |      2,169 | 2018-01-02 | 2026-08-19  |                                145.89 |                 75.05 |   30.54% |    -0.12 |          6.07 |                  -33.27 | 53.6 / 46.3 |
| **ABBV**  |      2,188 | 2018-01-02 | 2026-09-16  |                                123.33 |                 55.37 |   26.97% |    -1.20 |         14.36 |                  -33.03 | 53.8 / 45.9 |
| **AMZN**  |      2,188 | 2018-01-02 | 2026-09-16  |                                148.48 |                 53.21 |   34.43% |     0.03 |          4.62 |                  -33.31 | 52.6 / 47.2 |
| **BAC**   |      2,188 | 2018-01-02 | 2026-09-16  |                                 33.33 |                 10.27 |   31.43% |    -0.13 |         10.86 |                  -31.76 | 51.8 / 47.3 |
| **COST**  |      2,188 | 2018-01-02 | 2026-09-16  |                                525.01 |                281.40 |   22.63% |    -0.46 |          8.33 |                  -33.89 | 54.0 / 45.9 |
| **GOOGL** |      2,188 | 2018-01-02 | 2026-09-16  |                                132.52 |                 81.00 |   31.11% |    -0.12 |          3.85 |                  -34.11 | 53.4 / 46.5 |
| **IBM**   |      2,188 | 2018-01-02 | 2026-09-16  |                                143.50 |                 62.96 |   30.11% |    -1.97 |         32.97 |                  -31.94 | 53.9 / 46.0 |
| **INTC**  |      2,188 | 2018-01-02 | 2026-09-16  |                                 42.60 |                 17.53 |   48.46% |    -0.17 |         10.86 |                  -33.16 | 50.2 / 49.2 |
| **JNJ**   |      2,188 | 2018-01-02 | 2026-09-16  |                                144.93 |                 35.33 |   19.68% |    -0.41 |          8.53 |                  -33.81 | 52.6 / 47.1 |
| **JPM**   |      2,188 | 2018-01-02 | 2026-09-16  |                                155.70 |                 76.99 |   28.53% |    -0.09 |         12.90 |                  -32.11 | 52.8 / 46.9 |
| **KO**    |      2,188 | 2018-01-02 | 2026-09-16  |                                 52.26 |                 12.93 |   19.41% |    -0.70 |          9.00 |                  -32.35 | 52.5 / 46.7 |
| **LLY**   |      2,188 | 2018-01-02 | 2026-09-16  |                                420.62 |                336.42 |   31.66% |     0.45 |          9.19 |                  -33.15 | 54.2 / 45.6 |
| **MA**    |      2,188 | 2018-01-02 | 2026-09-16  |                                365.08 |                118.98 |   28.27% |    -0.03 |          8.45 |                  -34.26 | 54.0 / 45.9 |
| **MCD**   |      2,188 | 2018-01-02 | 2026-09-16  |                                222.52 |                 55.01 |   21.45% |    -0.26 |         27.95 |                  -33.24 | 51.6 / 48.2 |
| **META**  |      2,188 | 2018-01-02 | 2026-09-16  |                                338.47 |                188.14 |   41.91% |    -1.08 |         21.15 |                  -33.49 | 51.9 / 47.9 |
| **MSFT**  |      2,188 | 2018-01-02 | 2026-09-16  |                                275.53 |                128.33 |   29.07% |     0.00 |          8.11 |                  -35.44 | 53.6 / 46.2 |
| **NKE**   |      2,188 | 2018-01-02 | 2026-09-16  |                                 88.37 |                 27.34 |   34.10% |    -0.74 |         14.29 |                  -33.43 | 49.6 / 49.9 |
| **NVDA**  |      2,188 | 2018-01-02 | 2026-09-16  |                                 56.90 |                 67.02 |   50.35% |    -0.17 |          4.71 |                  -32.95 | 53.7 / 46.1 |
| **ORCL**  |      2,188 | 2018-01-02 | 2026-09-16  |                                 96.89 |                 57.02 |   37.47% |     1.13 |         20.25 |                  -33.55 | 53.2 / 46.5 |
| **PEP**   |      2,188 | 2018-01-02 | 2026-09-16  |                                129.36 |                 25.24 |   20.93% |    -0.37 |         15.96 |                  -34.43 | 51.7 / 48.0 |
| **TSLA**  |      2,188 | 2018-01-02 | 2026-09-16  |                                199.57 |                134.10 |   62.28% |    -0.06 |          3.74 |                  -32.51 | 51.8 / 48.2 |
| **^TNX**  |      2,169 | 2018-01-02 | 2026-08-19  |                                 2.96% |                 1.28% |   49.23% |     0.28 |         35.04 |                  -36.33 | 49.9 / 48.4 |
| **^VIX**  |      2,171 | 2018-01-02 | 2026-08-20  |                                 19.68 |                  7.25 |  129.30% |     1.34 |          8.26 |                  -35.56 | 44.0 / 55.5 |

---

## 7. Đọc và Hiểu Mã Nguồn — Insights Kỹ Thuật từ Implementation

Khi trực tiếp nghiên cứu và phát triển pipeline, chúng tôi rút ra các insights kỹ thuật sâu sắc từ mã nguồn:

1. **Xử lý MultiIndex của thư viện `yfinance` mới:**Trong các phiên bản `yfinance >= 0.2.36`, hàm `download()` tự động trả về header MultiIndex 2 cấp `(Price, Ticker)` ngay cả khi người dùng tải đơn mã. Trong [preprocessing/run_pipeline.py](../preprocessing/run_pipeline.py), đoạn code xử lý đã chủ động làm phẳng (flatten) cấu trúc cột:

   ```python
   if isinstance(df.columns, pd.MultiIndex):
       df.columns = df.columns.get_level_values(0)
   ```

   Điều này đảm bảo không gây crash khi truy cập `df['Close']`.
2. **Cách ly hoàn toàn Fit và Transform:**Trong hàm `scale_dataset()`, đối tượng scaler được fit trên Train split và serialize lại vào memory. Tập Validation và Test được scale thông qua `scaler.transform()`:

   ```python
   # Chỉ fit scaler trên tập train
   feature_scaler = RobustScaler()
   X_train_scaled = feature_scaler.fit_transform(X_train)
   # Tuyệt đối không gọi .fit() trên Val và Test
   X_val_scaled = feature_scaler.transform(X_val)
   X_test_scaled = feature_scaler.transform(X_test)
   ```

   Đây là minh chứng cho việc hiểu sâu quy tắc chống rò rỉ dữ liệu (Anti-Leakage Protocol).
3. **Sliding Window 3D Tensor:**
   Trong [datasets/stock_dataset.py](../datasets/stock_dataset.py), dữ liệu dạng bảng (Tabular 2D) được biến đổi thành chuỗi trượt 3 chiều `(N, L, D)` với $L=60$ phiên lịch sử:

   ```python
   X_windows.append(features[i : i + window_size])
   y_targets.append(targets[i + window_size])
   ```

   Mỗi mẫu dữ liệu là một "cuốn băng thời gian" dài 60 ngày, giúp mô hình trích xuất được cả các chu kỳ vi mô (sóng ngắn 10 ngày) lẫn xu hướng quý (60 ngày ~ 3 tháng giao dịch).

---

## 8. Cầu Nối: Từ Đặc Điểm Dữ Liệu Đến Quyết Định Kiến Trúc Mô Hình (Data → Model Bridge)

Phần này trả lời trực tiếp câu hỏi cốt lõi của Mentor: **"Từ đặc điểm của data mới quyết định bài toán, target, feature và model phù hợp."**

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    DATA CHARACTERISTIC → MODEL DECISION                       │
├──────────────────────────────────────┬────────────────────────────────────────┤
│ ĐẶC TÍNH DỮ LIỆU TÀI CHÍNH            │ QUYẾT ĐỊNH THIẾT KẾ MÔ HÌNH            │
├──────────────────────────────────────┼────────────────────────────────────────┤
│ 1. Giá đóng cửa có Unit Root,         │ → Quyết định KHÔNG dự báo Close price. │
│    không dừng (ADF t = -0.0269)      │ → Chọn Target = Log_Return (dừng hoàn  │
│                                      │   toàn, ADF t = -33.27, triệt Lag Fallacy)│
├──────────────────────────────────────┼────────────────────────────────────────┤
│ 2. Lợi suất có đuôi dày (Fat Tails,   │ → Không dùng MSE chuẩn (rất nhạy outlier).│
│    Kurtosis = 6.07)                  │ → Chọn Huber Loss (Smooth L1) và       │
│                                      │   RobustScaler (dựa trên IQR & Median).│
├──────────────────────────────────────┼────────────────────────────────────────┤
│ 3. Có mẫu hình dao động cục bộ        │ → Sử dụng 1D Temporal CNN (Kernel 3, 5)│
│    (Local Chart Patterns: đảo chiều, │   để trích xuất đặc trưng hình thái    │
│    bứt phá mô hình giá ngắn hạn)     │   không phụ thuộc bước trượt thời gian.│
├──────────────────────────────────────┼────────────────────────────────────────┤
│ 4. Có phụ thuộc dài hạn qua chu kỳ    │ → Sử dụng Bidirectional LSTM (BiLSTM)  │
│    vĩ mô và xu hướng (20-60 ngày)     │   để lưu giữ ngữ cảnh thời gian cả hai │
│                                      │   chiều, khắc phục vanishing gradient. │
├──────────────────────────────────────┼────────────────────────────────────────┤
│ 5. Tầm quan trọng của các phiên không │ → Tích hợp Temporal Self-Attention     │
│    đồng đều (phiên tin tức FOMC/CPI  │   để mô hình tự động gán trọng số lớn  │
│    quan trọng hơn các phiên bình lặng)│   vào các mốc thời gian mang tính bước │
│                                      │   ngoặt trong cửa sổ 60 ngày.          │
└──────────────────────────────────────┴────────────────────────────────────────┘
```

Chính chuỗi lập luận trên là cơ sở dẫn dắt sự ra đời của mô hình SOTA đề xuất:
**`CNN-BiLSTM-Attention (v2)`** — sự kết hợp cộng hưởng hoàn hảo để giải quyết triệt để các thách thức đặc thù của chuỗi thời gian tài chính!

---

## 9. Tham Chiếu Files

| File                                                                                         | Mô tả                                                         |
| :------------------------------------------------------------------------------------------- | :-------------------------------------------------------------- |
| [`preprocessing/run_pipeline.py`](../preprocessing/run_pipeline.py)                         | Toàn bộ pipeline tiền xử lý, scaling và tạo đặc trưng |
| [`datasets/stock_dataset.py`](../datasets/stock_dataset.py)                                 | Lớp PyTorch Dataset tạo Sliding Window                        |
| [`config/stock.yaml`](../config/stock.yaml)                                                 | Cấu hình tham số tickers, dates, train/val/test split        |
| [`scripts/generate_eda_html.py`](../scripts/generate_eda_html.py)                           | Script tạo báo cáo EDA HTML tương tác                     |
| [`docs/modeling.md`](./modeling.md)                                                         | Tài liệu kiến trúc mô hình và thực nghiệm Ablation     |
| [`docs/eda.html`](./eda.html)                                                               | Báo cáo trực quan tương tác EDA với Plotly               |
| [`output/ablation_benchmark_results_AAPL.md`](../output/ablation_benchmark_results_AAPL.md) | Kết quả kiểm định chi tiết trên AAPL                     |
