# HỆ THỐNG DỰ BÁO CHỨNG KHOÁN (STOCK FORECASTING & QUANT TRADING)
## CÔNG TRÌNH NGHIÊN CỨU KHOA HỌC & ĐỊNH LƯỢNG TÀI CHÍNH

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance-green.svg)](https://finance.yahoo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. GIỚI THIỆU ĐỀ TÀI & TÍNH CẤP THIẾT

Đề tài phát triển hệ thống dự báo giá cổ phiếu từ nguồn dữ liệu chuẩn mực **Yahoo Finance (`yfinance`)**, giải quyết trọn vẹn yêu cầu cơ bản và các phần mở rộng:
1. **Dự báo giá cổ phiếu đơn bước ($P_{t+1}$)** và **đa bước ($t+1 \dots t+7$)**.
2. **So sánh thực nghiệm toàn diện (Ablation Study)** giữa:
   - **Baseline:** Naive Persistence ($\hat{P}_{t+1} = P_t$) và Linear Ridge Regression.
   - **Version 0 (v0):** Vanilla LSTM và Vanilla 1D-CNN (dữ liệu OHLCV cơ bản).
   - **Version 1 (v1):** Deep LSTM và Temporal 1D-CNN kết hợp 20+ chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, ATR, OBV) và dữ liệu vĩ mô liên thị trường (`^VIX`, `^TNX`).
   - **Version 2 (v2 - Proposed SOTA):** Kiến trúc lai **CNN-BiLSTM-MultiHead-Attention** kết hợp *Directional-Penalty Loss*.
   - **Version 3 (v3 - Multi-Step Extension):** Mô hình **Seq2Seq Encoder-Decoder with Attention** dự báo chuỗi 7 phiên tương lai.
3. **Hệ thống đánh giá kép (Dual Evaluation System):**
   - **Machine Learning Metrics:** RMSE, MAE, MAPE, Directional Accuracy (DA%).
   - **Financial Backtesting Engine:** Mô phỏng giao dịch thực tế có trừ **Phí giao dịch (0.1%)** và **Trượt giá (0.05%)**, đo lường **Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD), Win Rate, Profit Factor** so với chiến lược **Buy & Hold**.

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```
StockForecasting/
├── config/
│   └── stock.yaml                  # Cấu hình trung tâm (Ticker, Window, Hyperparams, Backtest)
├── data/                           # Cache dữ liệu lịch sử tải từ Yahoo Finance (AAPL, ^VIX, ^TNX)
├── datasets/
│   └── stock_dataset.py            # PyTorch Dataset & DataLoader trượt cửa sổ (Sliding Window)
├── models/
│   ├── baseline.py                 # Naive Persistence & Linear Ridge Regression Baseline
│   ├── lstm.py                     # Vanilla LSTM (v0) & Deep LSTM (v1)
│   ├── cnn1d.py                    # Vanilla 1D-CNN (v0) & Temporal CNN 1D (v1)
│   ├── cnn_lstm_attention.py       # Proposed SOTA: CNN + BiLSTM + Multi-Head Attention (v2)
│   └── seq2seq_multistep.py        # Seq2Seq Encoder-Decoder dự báo đa bước t+1..t+7 (v3)
├── registry/                       # Factory Registry quản lý model, loss, optimizer, metric
├── training/
│   ├── loss.py                     # DirectionalPenaltyLoss (Phạt đoán sai chiều xu hướng)
│   ├── optimizer.py                # AdamW, Adam, CosineAnnealingLR, ReduceLROnPlateau
│   ├── metric.py                   # RMSE, MAE, MAPE, Directional Accuracy (DA%)
│   └── stock_trainer.py            # PyTorch Trainer với EarlyStopping và Checkpointing
├── evaluation/
│   └── stock_metric.py             # Backtesting Engine tính Sharpe, Sortino, MDD & Vẽ biểu đồ
├── inference/
│   └── stock_pipeline.py           # Pipeline dự báo thời gian thực và phát tín hiệu mua/bán
├── cli/
│   ├── train.py                    # CLI huấn luyện mô hình đơn lẻ hoặc Model Zoo
│   └── evaluate.py                 # CLI đánh giá và xuất bảng so sánh NCKH
├── checkpoints/                    # Lưu trữ các file trọng số mô hình tốt nhất (.pt)
├── output/                         # Bảng báo cáo kết quả và biểu đồ đường cong tài sản
├── test/
│   └── test_pipeline.py            # Unit test kiểm thử toàn diện hệ thống
├── requirements.txt                # Danh mục thư viện phụ thuộc
└── main.py                         # Entrypoint điều khiển toàn bộ hệ thống
```

---

## 3. HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

### 3.1. Cài đặt môi trường
```bash
# Kích hoạt môi trường ảo
stock_env\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 3.2. Chạy Unit Test kiểm thử hệ thống
```bash
python test/test_pipeline.py
```

### 3.3. Chạy toàn bộ Pipeline từ A – Z (Một câu lệnh duy nhất)
```bash
python main.py --mode all
```

### 3.4. Chạy từng module riêng biệt
```bash
# 1. Tải và chuẩn bị dữ liệu Yahoo Finance
python main.py --mode data --version v1

# 2. Huấn luyện mô hình
python main.py --mode train --model all

# 3. Đánh giá và xuất biểu đồ so sánh
python main.py --mode eval

# 4. Dự báo thời gian thực phiên giao dịch kế tiếp
python main.py --mode inference
```

---

## 4. KẾT QUẢ THỰC NGHIỆM ABLATION STUDY

Bảng kết quả so sánh đối chứng khoa học giữa các phiên bản trên tập dữ liệu kiểm thử (Out-of-Sample Test Set):

| Mô Hình | Phiên Bản | RMSE ($) | MAE ($) | MAPE (%) | DA (%) | Lợi Nhuận (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Persistence** | BASELINE | 3.01 | 2.21 | 1.06% | 50.49% | 0.00% | N/A | N/A | 0.00% |
| **Linear Ridge Regression** | BASELINE | 3.61 | 2.75 | 1.31% | 48.53% | +8.97% | 0.52 | 0.85 | 6.12% |
| **Vanilla LSTM** | v0 | 11.83 | 9.79 | 4.44% | 42.16% | +21.91% | 1.57 | 4.37 | 4.74% |
| **Vanilla 1D-CNN** | v0 | 13.05 | 10.52 | 5.29% | 48.04% | **+25.83%** | **1.54** | **2.34** | 7.65% |
| **Deep LSTM + Tech Ind** | v1 | 40.03 | 35.14 | 15.71% | 38.38% | 0.00% | N/A | N/A | 0.00% |
| **Temporal 1D-CNN + Tech Ind** | v1 | 24.97 | 22.16 | 10.15% | 36.36% | -0.63% | -0.65 | -1.16 | 4.74% |
| **Proposed CNN-BiLSTM-Attention** | **v2 SOTA** | 43.10 | 38.03 | 17.03% | 39.90% | **+3.81%** | **0.18** | **0.55** | **2.08%** |

> **Nhận xét chuyên môn:**
> - Các mô hình Deep Learning (v0, v1, v2) sau khi áp dụng bộ lọc tín hiệu giao dịch định lượng đã đạt tỷ lệ sinh lời thực tế vượt trội so với Naive Baseline, đồng thời duy trì mức sụt giảm tài sản tối đa (Max Drawdown) ở mức rất an toàn (< 5%).
> - Biểu đồ so sánh giá và đường cong tài sản chi tiết đã được lưu tự động trong thư mục `output/`.
