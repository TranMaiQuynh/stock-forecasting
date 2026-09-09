"""
Pipeline thu thập, làm sạch và trích xuất đặc trưng từ Yahoo Finance (yfinance).
Tuân thủ nghiêm ngặt nguyên tắc Không Rò Rỉ Dữ Liệu (Zero Data Leakage) cho NCKH.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def load_config(config_path: str = "config/stock.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_yahoo_data(ticker: str, start_date: str, end_date: str, cache_dir: str = "data") -> pd.DataFrame:
    """
    Tải dữ liệu OHLCV từ Yahoo Finance và lưu cache cục bộ.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start_date}_{end_date}.csv")
    
    if os.path.exists(cache_file):
        print(f"[Data] Đọc dữ liệu cache từ: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    print(f"[Data] Đang tải dữ liệu {ticker} từ Yahoo Finance ({start_date} đến {end_date})...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Xử lý MultiIndex columns nếu yfinance trả về dạng MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho mã {ticker} trên Yahoo Finance!")

    df.dropna(inplace=True)
    df.to_csv(cache_file)
    print(f"[Data] Đã lưu {len(df)} phiên giao dịch vào: {cache_file}")
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán hơn 20 chỉ báo kỹ thuật chuyên sâu từ dữ liệu OHLCV của Yahoo Finance.
    Bao gồm các nhóm: Trend, Momentum, Volatility, Volume.
    """
    df = df.copy()
    
    # 1. Nhóm Xu hướng (Trend)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 2. Nhóm Động lượng (Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
    
    # 3. Nhóm Biến động (Volatility)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9)
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
    low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # 4. Nhóm Khối lượng & Dòng tiền (Volume)
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA20'] + 1e-9)
    
    # On-Balance Volume (OBV)
    obv_direction = np.where(df['Close'] > df['Close'].shift(1), 1, np.where(df['Close'] < df['Close'].shift(1), -1, 0))
    df['OBV'] = (obv_direction * df['Volume']).cumsum()
    
    # 5. Log-return & Biến động lịch sử (Historical Volatility)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Hist_Volatility_20'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)

    df.dropna(inplace=True)
    return df


def merge_macro_data(main_df: pd.DataFrame, macro_tickers: list, start_date: str, end_date: str, cache_dir: str) -> pd.DataFrame:
    """
    Tải và ghép các chỉ số vĩ mô từ Yahoo Finance (VIX, Lợi suất trái phiếu TNX) vào DataFrame chính.
    """
    merged_df = main_df.copy()
    for macro in macro_tickers:
        clean_name = macro.replace("^", "").replace("=", "")
        try:
            macro_data = download_yahoo_data(macro, start_date, end_date, cache_dir)
            macro_series = macro_data['Close'].rename(f"Macro_{clean_name}")
            merged_df = merged_df.join(macro_series, how='left')
            merged_df[f"Macro_{clean_name}"] = merged_df[f"Macro_{clean_name}"].ffill().bfill()
        except Exception as e:
            print(f"[Warning] Không thể tải dữ liệu vĩ mô {macro}: {e}")
            
    merged_df.dropna(inplace=True)
    return merged_df


def prepare_dataset_pipeline(config: dict, version: str = "v0"):
    """
    Hàm thực thi pipeline chuẩn hóa dữ liệu theo phiên bản (v0, v1, v2, v3).
    - v0: Chỉ dùng giá OHLCV gốc (5 features)
    - v1, v2, v3: Dùng 20+ chỉ báo kỹ thuật + dữ liệu vĩ mô (25+ features)
    """
    cfg_data = config['data']
    cfg_split = config['split']
    
    # 1. Tải dữ liệu Yahoo Finance
    raw_df = download_yahoo_data(
        ticker=cfg_data['ticker'],
        start_date=cfg_data['start_date'],
        end_date=cfg_data['end_date'],
        cache_dir=cfg_data['cache_dir']
    )
    
    if version == "v0":
        print(f"\n[Pipeline] Đang chuẩn bị dữ liệu cho phiên bản v0 (Vanilla - OHLCV cơ bản)...")
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        processed_df = raw_df[feature_cols].copy()
    else:
        print(f"\n[Pipeline] Đang chuẩn bị dữ liệu cho phiên bản {version} (Feature Engineering + Macro)...")
        df_ti = calculate_technical_indicators(raw_df)
        macro_tickers = cfg_data.get('macro_tickers', [])
        processed_df = merge_macro_data(df_ti, macro_tickers, cfg_data['start_date'], cfg_data['end_date'], cfg_data['cache_dir'])
        feature_cols = [col for col in processed_df.columns]

    # 2. Phân chia Train/Val/Test theo thứ tự thời gian (Không rò rỉ)
    n_total = len(processed_df)
    n_train = int(n_total * cfg_split['train_ratio'])
    n_val = int(n_total * cfg_split['val_ratio'])
    
    train_df = processed_df.iloc[:n_train].copy()
    val_df = processed_df.iloc[n_train:n_train + n_val].copy()
    test_df = processed_df.iloc[n_train + n_val:].copy()
    
    print(f"[Pipeline] Phân chia dữ liệu: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} | Tổng={n_total}")
    print(f"[Pipeline] Số lượng đặc trưng đầu vào (Input Features): {len(feature_cols)}")

    # 3. Chuẩn hóa dữ liệu (CHỈ FIT TRÊN TRAIN)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    train_features = feature_scaler.fit_transform(train_df[feature_cols])
    val_features = feature_scaler.transform(val_df[feature_cols])
    test_features = feature_scaler.transform(test_df[feature_cols])
    
    target_col = cfg_data['target_col']
    train_target = target_scaler.fit_transform(train_df[[target_col]])
    val_target = target_scaler.transform(val_df[[target_col]])
    test_target = target_scaler.transform(test_df[[target_col]])
    
    return {
        'train_features': train_features,
        'val_features': val_features,
        'test_features': test_features,
        'train_target': train_target,
        'val_target': val_target,
        'test_target': test_target,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'target_col': target_col
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/stock.yaml")
    parser.add_argument("--version", type=str, default="v0", choices=["v0", "v1", "v2", "v3"])
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    data_bundle = prepare_dataset_pipeline(cfg, version=args.version)
    print(f"[Hoàn tất] Pipeline tiền xử lý dữ liệu {args.version} đã chạy thành công!")
