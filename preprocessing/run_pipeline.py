"""
Pipeline thu thập, làm sạch và trích xuất đặc trưng từ Yahoo Finance (yfinance).
Tuân thủ nghiêm ngặt nguyên tắc Không Rò Rỉ Dữ Liệu (Zero Data Leakage).
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
    Tự động tận dụng file tổng 2018-01-01_2026-12-31 nếu có để cắt lát (slice) không cần tải lại.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start_date}_{end_date}.csv")
    full_cache_file = os.path.join(cache_dir, f"{ticker}_2018-01-01_2026-12-31.csv")
    
    # 1. Ưu tiên kiểm tra file cache chính xác
    if os.path.exists(cache_file):
        print(f"[Data] Đọc dữ liệu cache từ: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    # 2. Kiểm tra file cache toàn dải 2018-2026 và cắt lát theo ngày
    if os.path.exists(full_cache_file):
        print(f"[Data] Đọc và trích xuất dữ liệu từ file tổng: {full_cache_file} ({start_date} -> {end_date})")
        df_full = pd.read_csv(full_cache_file, index_col=0, parse_dates=True)
        df_slice = df_full.loc[start_date:end_date].copy()
        if len(df_slice) > 0:
            return df_slice

    print(f"[Data] Đang tải dữ liệu {ticker} từ Yahoo Finance ({start_date} đến {end_date})...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
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

    # 6. Price-structure features (stationary — tỷ lệ, không phụ thuộc mức giá tuyệt đối)
    # Đây là bộ feature thay thế OHLC thô: bất kể giá là $40 hay $400, các tỷ lệ này
    # luôn dao động trong khoảng ổn định → không gây out-of-distribution khi Test > Train range.
    df['Open_Return']    = df['Open'] / df['Close'].shift(1) - 1   # Overnight gap (gap qua đêm)
    df['High_Ratio']     = df['High'] / df['Close'] - 1            # Upper wick ratio (>= 0)
    df['Low_Ratio']      = df['Low']  / df['Close'] - 1            # Lower wick ratio (<= 0)
    df['MACD_Hist_norm'] = df['MACD_Hist'] / df['Close']            # MACD/Close: đơn vị % thay vì USD

    df.dropna(inplace=True)
    return df


def clip_outliers(df: pd.DataFrame, feature_cols: list, target_col: str = None,
                  lower_pct: float = 0.01, upper_pct: float = 0.99) -> tuple:
    """
    [v2-FIX] Winsorization: Cắt giá trị cực trị tại ngưỡng 1% và 99% phân vị.
    Phân vị ĐƯỢC TÍNH CHỈ TRÊN TẬP TRAIN để tránh data leakage.
    Không áp dụng cho cột Volume (đột biến khối lượng là tín hiệu quan trọng).
    Không áp dụng cho target_col — scaler phải thấy phân phối thật của target.

    Returns:
        (df_clipped, bounds_dict): bounds_dict lưu {col: (lower, upper)} để
        apply_clip_bounds() áp lên val/test với CÙNG ngưỡng từ train.
    """
    df = df.copy()
    skip_cols = {'Volume'}  # Giữ nguyên Volume spike (tín hiệu dòng tiền quan trọng)
    if target_col:
        skip_cols.add(target_col)  # Target không bị clip — scaler phải thấy phân phối thật
    bounds = {}
    for col in feature_cols:
        if col in df.columns and col not in skip_cols:
            lower_bound = df[col].quantile(lower_pct)
            upper_bound = df[col].quantile(upper_pct)
            bounds[col] = (lower_bound, upper_bound)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df, bounds


def apply_clip_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Áp dụng bounds Winsorization (đã tính từ tập Train) lên val/test.
    Không tính lại quantile — đảm bảo Zero Data Leakage.
    """
    df = df.copy()
    for col, (lower_bound, upper_bound) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def select_orthogonal_features(df: pd.DataFrame) -> list:
    """
    Chọn lọc đặc trưng trực giao — giảm đa cộng tuyến, đảm bảo tính dừng (stationarity).

    Tất cả features đều là tỷ lệ/delta — không có đơn vị USD tuyệt đối:
    - Price structure : Log_Return, Open_Return, High_Ratio, Low_Ratio (tỷ lệ so Close)
    - Volume          : Volume_Ratio (Volume / SMA20Volume)
    - Trend           : MACD_Hist_norm (MACD_Hist / Close → đơn vị %)
    - Momentum        : RSI_14 (bounded [0, 100])
    - Volatility      : ATR_norm (ATR_14 / Close → đơn vị %)
    - Macro           : Delta_VIX, Delta_TNX (first-difference → stationary)

    LƯU Ý: Hàm này chỉ CHỌN, KHÔNG TẠO cột mới.
    Các cột dẫn xuất phải được tạo trước trong prepare_dataset_pipeline().
    """
    # Core: 5 stationary price-structure + volume features (thay thế OHLCV thô)
    core = [
        'Log_Return',    # ln(Close_t/Close_{t-1}) — return đóng cửa
        'Open_Return',   # Open_t/Close_{t-1} - 1  — overnight gap
        'High_Ratio',    # High_t/Close_t - 1       — upper wick (>= 0)
        'Low_Ratio',     # Low_t/Close_t - 1        — lower wick (<= 0)
        'Volume_Ratio',  # Volume / SMA20Volume     — volume deviation
    ]

    # 1 đại diện cho mỗi nhóm kỹ thuật (đã stationary)
    tech_candidates = [
        'MACD_Hist_norm',  # Trend: MACD_Hist / Close → đơn vị %, không phụ thuộc mức giá
        'RSI_14',          # Momentum: bounded [0, 100]
        'ATR_norm',        # Volatility: ATR_14 / Close → đơn vị %
    ]

    # Macro: first-difference (Delta) thay vì mức tuyệt đối
    macro_candidates = ['Delta_VIX', 'Delta_TNX']

    selected = [c for c in core if c in df.columns]
    for col in tech_candidates + macro_candidates:
        if col in df.columns:
            selected.append(col)

    print(f"[Feature] Lọc trực giao: {len(selected)} đặc trưng stationary ← [{', '.join(selected)}]")
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
        print(f"\n[Pipeline] Đang chuẩn bị dữ liệu cho phiên bản v0 (Stationary OHLCV ratios)...")
        # Gọi calculate_technical_indicators để có Open_Return, High_Ratio, Low_Ratio,
        # Volume_Ratio, Log_Return — tất cả đều stationary, không phụ thuộc mức giá tuyệt đối.
        # v0 vẫn là "ablation dùng thông tin OHLCV" nhưng biểu diễn dạng tỷ lệ đúng chuẩn DL.
        # Macro data KHÔNG được tải ở đây — đặc điểm phân biệt v0 vs v1/v2/v3.
        processed_df = calculate_technical_indicators(raw_df)
    else:
        print(f"\n[Pipeline] Đang chuẩn bị dữ liệu cho phiên bản {version} (Feature Engineering + Macro)...")
        df_ti = calculate_technical_indicators(raw_df)
        macro_tickers = cfg_data.get('macro_tickers', [])
        processed_df = merge_macro_data(
            df_ti, macro_tickers, cfg_data['start_date'], cfg_data['end_date'], cfg_data['cache_dir']
        )

    # Tạo các cột dẫn xuất TRÊN processed_df TRƯỚC KHI chia train/val/test
    # (ATR_norm, Delta_VIX, Delta_TNX cần được tính trên toàn bộ chuỗi để val/test cũng có)
    if 'ATR_14' in processed_df.columns:
        processed_df['ATR_norm'] = processed_df['ATR_14'] / processed_df['Close']
    if 'Macro_VIX' in processed_df.columns:
        processed_df['Delta_VIX'] = processed_df['Macro_VIX'].diff().fillna(0)
    if 'Macro_TNX' in processed_df.columns:
        # Delta_TNX = first difference của lợi suất trái phiếu — stationary, thay Macro_TNX tuyệt đối
        processed_df['Delta_TNX'] = processed_df['Macro_TNX'].diff().fillna(0)

    # Log_Return đã được tính sẵn trong calculate_technical_indicators() cho cả v0 lẫn v1+.
    # Khối này chỉ còn là safety-net cho trường hợp processed_df không qua calculate_technical_indicators.
    if use_log_return and 'Log_Return' not in processed_df.columns and 'Close' in processed_df.columns:
        processed_df['Log_Return'] = np.log(
            processed_df['Close'] / processed_df['Close'].shift(1)
        )
        processed_df.dropna(inplace=True)
        print(f"[Pipeline] Đã tính cột Log_Return = ln(P_t / P_{{t-1}}) (fallback)")

    # 2. Phân chia Train/Val/Test theo thứ tự thời gian (Không rò rỉ)
    n_total = len(processed_df)
    n_train = int(n_total * cfg_split['train_ratio'])
    n_val   = int(n_total * cfg_split['val_ratio'])
    # test_df = phần còn lại — test_ratio trong config chỉ mang tính tài liệu, không điều khiển logic
    n_test  = n_total - n_train - n_val

    # Kiểm tra tập test đủ lớn cho cả nhánh multi-step (input_window + max_horizon)
    cfg_win = config.get('window', {})
    max_horizon = max(cfg_win.get('forecast_horizon', 1), cfg_win.get('multi_step_horizon', 1))
    min_test_rows = cfg_win.get('input_window', 60) + max_horizon
    if n_test < min_test_rows:
        raise ValueError(
            f"Tập test chỉ có {n_test} dòng, cần ít nhất {min_test_rows} "
            f"(input_window={cfg_win.get('input_window', 60)} + max_horizon={max_horizon}). "
            f"Kiểm tra lại train_ratio/val_ratio trong config."
        )

    train_df = processed_df.iloc[:n_train].copy()
    val_df   = processed_df.iloc[n_train:n_train + n_val].copy()
    test_df  = processed_df.iloc[n_train + n_val:].copy()

    print(f"[Pipeline] Phân chia dữ liệu: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} | Tổng={n_total}")

    # Winsorization: tính bounds từ Train, nhưng CHỈ áp lên bản sao *_feat_df dùng cho feature_scaler.
    # train_df / val_df / test_df GIỮ NGUYÊN giá trị gốc — cli/evaluate.py đọc test_df['Close']
    # làm actual_prices (ground-truth backtest). Nếu mutate test_df ở đây, giá Close sẽ bị flat-cap
    # tại ngưỡng 99th-percentile của Train → sai lệch toàn bộ RMSE, Sharpe, Total_Return, MDD.
    if version != "v0":
        target_col_for_clip = cfg_data.get('target_col')
        all_cols = list(processed_df.columns)
        # Bản sao chỉ dùng cho feature_scaler — không trả ra ngoài
        train_feat_df, clip_bounds = clip_outliers(train_df.copy(), all_cols, target_col=target_col_for_clip)
        val_feat_df  = apply_clip_bounds(val_df.copy(),  clip_bounds)
        test_feat_df = apply_clip_bounds(test_df.copy(), clip_bounds)
        print(f"[Pipeline][v2] Winsorization hoàn tất: bounds từ Train → áp lên bản sao feat_df (ngưỡng [1%, 99%])")
        print(f"[Pipeline][v2] train_df/val_df/test_df gốc GIỮ NGUYÊN → ground-truth backtest không bị ảnh hưởng")
    else:
        # v0: không Winsorize — dùng trực tiếp
        train_feat_df, val_feat_df, test_feat_df = train_df, val_df, test_df

    # [v2-FIX] Chọn lọc đặc trưng trực giao (giảm đa cộng tuyến)
    # Dùng train_feat_df (đã clip) để select_orthogonal_features — nhất quán với fit scaler bên dưới
    if version in ["v1", "v2", "v3"]:
        feature_cols = select_orthogonal_features(train_feat_df)
    else:
        # v0: dùng bộ stationary price-structure features (5 cột)
        # Không dùng MACD, RSI, macro — giữ đặc trưng ablation study chỉ từ OHLCV
        feature_cols = [c for c in [
            'Log_Return',    # Return đóng cửa
            'Open_Return',   # Overnight gap
            'High_Ratio',    # Upper wick ratio
            'Low_Ratio',     # Lower wick ratio
            'Volume_Ratio',  # Volume deviation
        ] if c in processed_df.columns]

    print(f"[Pipeline] Số lượng đặc trưng đầu vào (Input Features): {len(feature_cols)}")

    # 3. Xác định target
    target_col = cfg_data['target_col']
    if target_col not in processed_df.columns:
        raise ValueError(
            f"Cột target '{target_col}' không tồn tại trong DataFrame. "
            f"Nếu dùng Log_Return, hãy bật use_log_return: true trong config."
        )

    # 4. [v2-FIX] Chuẩn hóa dữ liệu:
    #    - feature_scaler fit/transform trên *_feat_df (đã Winsorize) → đặc trưng sạch cho model
    #    - target_scaler  fit/transform trên train_df gốc (target không bị clip) → phân phối thật
    #    Cả hai chỉ fit trên Train → Zero Data Leakage
    feature_scaler = RobustScaler()    # [v2-FIX] MinMaxScaler → RobustScaler
    target_scaler  = StandardScaler()  # [v2-FIX] MinMaxScaler → StandardScaler

    train_features = feature_scaler.fit_transform(train_feat_df[feature_cols])
    val_features   = feature_scaler.transform(val_feat_df[feature_cols])
    test_features  = feature_scaler.transform(test_feat_df[feature_cols])

    # target_scaler dùng train_df gốc (target_col đã được loại khỏi clip từ trước)
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
        # train_df/val_df/test_df trả về GIỮ NGUYÊN giá gốc (chưa clip)
        # → cli/evaluate.py đọc test_df['Close'] làm ground-truth backtest vẫn là giá thật
        'train_df':       train_df,
        'val_df':         val_df,
        'test_df':        test_df,
        'target_col':     target_col,
        'use_log_return': use_log_return,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/stock.yaml")
    parser.add_argument("--version", type=str, default="v0", choices=["v0", "v1", "v2", "v3"])
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    data_bundle = prepare_dataset_pipeline(cfg, version=args.version)
    print(f"[Hoàn tất] Pipeline tiền xử lý dữ liệu {args.version} đã chạy thành công!")
