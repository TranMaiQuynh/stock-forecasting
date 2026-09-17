"""
Inference Pipeline: Dự báo giá thời gian thực từ dữ liệu mới nhất trên Yahoo Finance.
Tạo khuyến nghị đầu tư (Actionable Signal) và dự báo xu hướng 1 ngày & 7 ngày tới.

Lưu ý quan trọng: Sử dụng cùng scaler từ training data để đảm bảo nhất quán.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import numpy as np
import pandas as pd
from preprocessing.run_pipeline import (
    load_config, download_yahoo_data, calculate_technical_indicators,
    merge_macro_data, prepare_dataset_pipeline
)
from registry.model_registry import build_model_by_name


def run_live_inference(config_path: str = "config/stock.yaml", model_name: str = "v2_sota", version: str = "v2"):
    config = load_config(config_path)
    cfg_data = config['data']
    cfg_win = config['window']
    
    ticker = cfg_data['ticker']
    window_size = cfg_win['input_window']
    
    # 1. Lấy scaler nhất quán từ training pipeline
    # Sử dụng cùng prepare_dataset_pipeline để có scaler fit trên training data
    print(f"\n[Inference] Đang chuẩn bị scaler từ training data...")
    data_version = "v0" if version == "v0" else "v1"
    training_bundle = prepare_dataset_pipeline(config, version=data_version)
    feature_scaler = training_bundle['feature_scaler']
    target_scaler = training_bundle['target_scaler']
    feature_cols = training_bundle['feature_cols']
    
    # 2. Tải dữ liệu thời gian thực
    print(f"[Inference] Đang tải dữ liệu thời gian thực của {ticker} từ Yahoo Finance...")
    raw_df = download_yahoo_data(
        ticker=ticker,
        start_date="2024-01-01",
        end_date="2026-12-31",
        cache_dir=cfg_data['cache_dir']
    )
    
    if version == "v0":
        processed_df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    else:
        df_ti = calculate_technical_indicators(raw_df)
        macro_tickers = cfg_data.get('macro_tickers', [])
        processed_df = merge_macro_data(df_ti, macro_tickers, "2024-01-01", "2026-12-31", cfg_data['cache_dir'])

    if 'ATR_14' in processed_df.columns and 'Close' in processed_df.columns:
        processed_df['ATR_norm'] = processed_df['ATR_14'] / processed_df['Close']
    if 'Macro_VIX' in processed_df.columns:
        processed_df['Delta_VIX'] = processed_df['Macro_VIX'].diff().fillna(0)
    if 'Close' in processed_df.columns:
        processed_df['Log_Return'] = np.log(processed_df['Close'] / processed_df['Close'].shift(1))
    processed_df.dropna(inplace=True)
        
    recent_features = processed_df.iloc[-window_size:][feature_cols].values
    last_close = processed_df['Close'].iloc[-1]
    last_date = processed_df.index[-1].strftime("%Y-%m-%d")
    
    # 3. Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(feature_cols)
    actual_model_name = "cnn_bilstm_attention" if model_name in ["v2_sota", "cnn_bilstm_attention_v2"] else model_name
    # forecast_horizon được giải quyết động — giống logic trong cli/train.py:61-65
    forecast_horizon = (
        cfg_win['multi_step_horizon']
        if ("multistep" in actual_model_name or version == "v3")
        else cfg_win['forecast_horizon']
    )
    model = build_model_by_name(actual_model_name, input_dim=input_dim, config=config, forecast_horizon=forecast_horizon)
    
    ckpt_candidates = [
        os.path.join(config['training']['checkpoint_dir'], ticker, f"{actual_model_name}_{version}_seed42_best.pt"),
        os.path.join(config['training']['checkpoint_dir'], ticker, f"{actual_model_name}_seed42_best.pt"),
        os.path.join(config['training']['checkpoint_dir'], f"{actual_model_name}_{version}_best.pt"),
        os.path.join(config['training']['checkpoint_dir'], f"{model_name}_best.pt"),
    ]
    checkpoint_path = None
    for cand in ckpt_candidates:
        if os.path.exists(cand):
            checkpoint_path = cand
            break
        
    if checkpoint_path is None:
        print(f"[Warning] Chưa tìm thấy checkpoint phù hợp trong {ckpt_candidates}, sử dụng trọng số khởi tạo để mô phỏng.")
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"[Inference] Đã tải thành công checkpoint: {checkpoint_path}")
        
    model.to(device)
    model.eval()
    
    # 4. Chuẩn hóa bằng CÙNG scaler từ training data (QUAN TRỌNG)
    scaled_input = feature_scaler.transform(recent_features)
    
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()
        
    # 5. Denormalize bằng target_scaler từ training data
    # pred_scaled có shape (forecast_horizon,) — chỉ lấy bước đầu tiên (t+1) để hiển thị
    # Nếu muốn hiển thị đa bước, mở rộng phần này trong tương lai
    target_val = target_scaler.inverse_transform(pred_scaled[0:1].reshape(-1, 1)).flatten()[0]
    use_log_return = config.get('data', {}).get('use_log_return', False)
    
    if use_log_return:
        pred_log_return = target_val
        pred_price = float(last_close * np.exp(pred_log_return))
        pct_change = float((np.exp(pred_log_return) - 1.0) * 100.0)
    else:
        pred_price = float(target_val)
        pct_change = float(((pred_price - last_close) / last_close) * 100.0)
    
    # 6. Tính toán khuyến nghị
    if pct_change > 1.0:
        action = "🟢 MUA MẠNH (STRONG BUY)"
    elif pct_change > 0.05:
        action = "🟢 MUA (BUY / ACCUMULATE)"
    elif pct_change < -1.0:
        action = "🔴 BÁN MẠNH (STRONG SELL)"
    elif pct_change < -0.05:
        action = "🔴 BÁN (SELL / TAKE PROFIT)"
    else:
        action = "🟡 THEO DÕI (HOLD / CASH)"
        
    print("\n" + "="*60)
    print(f"       KẾT QUẢ DỰ BÁO THỜI GIAN THỰC - MÃ: {ticker}")
    print("="*60)
    print(f"Phiên giao dịch gần nhất ({last_date}): ${last_close:.2f}")
    print(f"Giá dự báo phiên kế tiếp (Next Day):     ${pred_price:.2f}")
    print(f"Biến động kỳ vọng:                      {pct_change:+.2f}%")
    print(f"Khuyến nghị định lượng:                 {action}")
    print("="*60 + "\n")
    
    return {
        'ticker': ticker,
        'last_date': last_date,
        'last_close': last_close,
        'pred_price': pred_price,
        'expected_change%': pct_change,
        'action': action
    }


if __name__ == "__main__":
    run_live_inference()

