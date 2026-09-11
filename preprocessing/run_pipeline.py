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


def clip_outliers(df: pd.DataFrame, feature_cols: list,
                  lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    [v2-FIX] Winsorization: Cắt giá trị cực trị tại ngưỡng 1% và 99% phân vị.
    Phân vị ĐƯỢC TÍNH CHỈ TRÊN TẬP TRAIN để tránh data leakage.
    Không áp dụng cho cột Volume (đột biến khối lượng là tín hiệu quan trọng).
    """
    df = df.copy()
    skip_cols = {'Volume'}  # Giữ nguyên Volume spike (tín hiệu dòng tiền quan trọng)
    for col in feature_cols:
        if col in df.columns and col not in skip_cols:
            lower_bound = df[col].quantile(lower_pct)
            upper_bound = df[col].quantile(upper_pct)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def select_orthogonal_features(df: pd.DataFrame) -> list:
    """
    [v2-FIX] Chọn lọc đặc trưng trực giao — giảm đa cộng tuyến.
    Thay vì dùng tất cả 27+ cột (SMA5, SMA10, SMA20... gần giống nhau),
    chỉ chọn 1 đại diện cho mỗi nhóm tín hiệu:
    - Trend: MACD_Hist
    - Momentum: RSI_14
    - Volatility: ATR_norm (= ATR_14 / Close)
    - Volume: Volume_Ratio
    - Return: Log_Return
    - Macro: Delta_VIX, Macro_TNX

    LƯU Ý: Các cột dẫn xuất (ATR_norm, Delta_VIX) phải được tạo trên
    processed_df TRƯỚC khi chia train/val/test (trong prepare_dataset_pipeline).
    Hàm này chỉ CHỌN, KHÔNG TẠO cột mới.
    """
    # Bộ đặc trưng cốt lõi (luôn có trong mọi phiên bản)
    core = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Một đại diện cho mỗi nhóm kỹ thuật (ưu tiên theo thứ tự)
    tech_candidates = [
        'MACD_Hist',    # Trend
        'RSI_14',       # Momentum
        'ATR_norm',     # Volatility (normalized)
        'Volume_Ratio', # Volume flow
        'Log_Return',   # Return (stationary)
    ]
    macro_candidates = ['Delta_VIX', 'Macro_TNX']

    selected = list(core)
    for col in tech_candidates + macro_candidates:
        if col in df.columns:
            selected.append(col)

    print(f"[Feature] Lọc trực giao: {len(selected)} đặc trưng được giữ lại ← [{', '.join(selected)}]")
    return selected


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

    [v2-FIX] Cải tiến:
    - Tự động thêm cột Log_Return khi use_log_return = true
    - Winsorization: cắt outlier tại [1%, 99%] phân vị của tập Train
    - Chọn lọc đặc trưng trực giao (10 cột) thay vì dùng tất cả 27+ cột
    - Thay MinMaxScaler bằng StandardScaler: không giới hạn [0,1], tránh lỗi ngoại suy
    """
    cfg_data = config['data']
    cfg_split = config['split']
    use_log_return = cfg_data.get('use_log_return', False)

    # 1. Tải dữ liệu Yahoo Finance
    raw_df = download_yahoo_data(
        ticker=cfg_data['ticker'],
        start_date=cfg_data['start_date'],
        end_date=cfg_data['end_date'],
        cache_dir=cfg_data['cache_dir']
    )

    if version == "v0":
        print(f"\n[Pipeline] Đang chuẩn bị dữ liệu cho phiên bản v0 (Vanilla - OHLCV cơ bản)...")
        processed_df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    else:
        print(f"\n[Pipeline] Đang chuẩn bị dữ liệu cho phiên bản {version} (Feature Engineering + Macro)...")
        df_ti = calculate_technical_indicators(raw_df)
        macro_tickers = cfg_data.get('macro_tickers', [])
        processed_df = merge_macro_data(
            df_ti, macro_tickers, cfg_data['start_date'], cfg_data['end_date'], cfg_data['cache_dir']
        )

    # [v2-FIX] Tạo các cột dẫn xuất TRÊN processed_df TRƯỚC KHI chia train/val/test
    # Để đảm bảo val_df và test_df cũng có các cột này
    if 'ATR_14' in processed_df.columns:
        processed_df['ATR_norm'] = processed_df['ATR_14'] / processed_df['Close']
    if 'Macro_VIX' in processed_df.columns:
        processed_df['Delta_VIX'] = processed_df['Macro_VIX'].diff().fillna(0)

    # [v2-FIX] Thêm cột Log_Return vào processed_df
    if use_log_return and 'Close' in processed_df.columns:
        processed_df['Log_Return'] = np.log(
            processed_df['Close'] / processed_df['Close'].shift(1)
        )
        processed_df.dropna(inplace=True)
        print(f"[Pipeline][v2] Đã tính cột Log_Return = ln(P_t / P_{{t-1}})")

    # 2. Phân chia Train/Val/Test theo thứ tự thời gian (Không rò rỉ)
    n_total = len(processed_df)
    n_train = int(n_total * cfg_split['train_ratio'])
    n_val   = int(n_total * cfg_split['val_ratio'])

    train_df = processed_df.iloc[:n_train].copy()
    val_df   = processed_df.iloc[n_train:n_train + n_val].copy()
    test_df  = processed_df.iloc[n_train + n_val:].copy()

    print(f"[Pipeline] Phân chia dữ liệu: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} | Tổng={n_total}")

    # [v2-FIX] Winsorization: cắt outlier dựa trên phân vị của tập Train
    if version != "v0":  # v0 có ít cột, OHLCV thường không cần Winsorize
        all_cols = list(processed_df.columns)
        train_df = clip_outliers(train_df, all_cols)
        print(f"[Pipeline][v2] Winsorization hoàn tất trên tập Train (ngưỡng [1%, 99%])")

    # [v2-FIX] Chọn lọc đặc trưng trực giao (giảm đa cộng tuyến)
    if version in ["v1", "v2", "v3"]:
        feature_cols = select_orthogonal_features(train_df)
    else:
        # v0: giữ OHLCV + Log_Return (nếu có)
        feature_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return']
                        if c in processed_df.columns]

    print(f"[Pipeline] Số lượng đặc trưng đầu vào (Input Features): {len(feature_cols)}")

    # 3. Xác định target
    target_col = cfg_data['target_col']
    if target_col not in processed_df.columns:
        raise ValueError(
            f"Cột target '{target_col}' không tồn tại trong DataFrame. "
            f"Nếu dùng Log_Return, hãy bật use_log_return: true trong config."
        )

    # 4. [v2-FIX] Chuẩn hóa dữ liệu:
    #    - Feature scaler: RobustScaler (chống outlier, không bị giới hạn [0,1])
    #    - Target scaler : StandardScaler (chuẩn hóa Gauss μ=0, σ=1)
    #    Cả hai chỉ fit trên Train, transform Val/Test → Zero Data Leakage
    feature_scaler = RobustScaler()    # [v2-FIX] MinMaxScaler → RobustScaler
    target_scaler  = StandardScaler()  # [v2-FIX] MinMaxScaler → StandardScaler

    train_features = feature_scaler.fit_transform(train_df[feature_cols])
    val_features   = feature_scaler.transform(val_df[feature_cols])
    test_features  = feature_scaler.transform(test_df[feature_cols])

    train_target = target_scaler.fit_transform(train_df[[target_col]])
    val_target   = target_scaler.transform(val_df[[target_col]])
    test_target  = target_scaler.transform(test_df[[target_col]])

    print(f"[Pipeline][v2] Scaler: RobustScaler (features) + StandardScaler (target='{target_col}')")
    print(f"[Pipeline] Chỉ fit scaler trên Train → Zero Data Leakage (chuẩn NCKH)")

    return {
        'train_features': train_features,
        'val_features':   val_features,
        'test_features':  test_features,
        'train_target':   train_target,
        'val_target':     val_target,
        'test_target':    test_target,
        'feature_scaler': feature_scaler,
        'target_scaler':  target_scaler,
        'feature_cols':   feature_cols,
        'train_df':       train_df,
        'val_df':         val_df,
        'test_df':        test_df,
        'target_col':     target_col,
        'use_log_return': use_log_return,   # Truyền xuống cho evaluation
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/stock.yaml")
    parser.add_argument("--version", type=str, default="v0", choices=["v0", "v1", "v2", "v3"])
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    data_bundle = prepare_dataset_pipeline(cfg, version=args.version)
    print(f"[Hoàn tất] Pipeline tiền xử lý dữ liệu {args.version} đã chạy thành công!")
