"""
CLI Đánh giá & So sánh Toàn diện (Comprehensive Benchmark & Ablation Evaluation):
- So sánh Base Model vs. v0 vs. v1 vs. v2 vs. v3
- Xuất bảng ML Metrics (RMSE, MAE, MAPE, DA%)
- Xuất bảng Financial & Risk-adjusted Metrics (Sharpe, Sortino, MDD, CAGR, Profit Factor)
- Lưu đồ thị so sánh vào thư mục output/
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.run_pipeline import load_config, prepare_dataset_pipeline
from datasets.stock_dataset import build_dataloaders
from registry.model_registry import build_model_by_name
from evaluation.stock_metric import calculate_financial_metrics, plot_and_save_results
from training.metric import evaluate_ml_metrics


def evaluate_all_models(config_path: str = "config/stock.yaml"):
    config = load_config(config_path)
    cfg_win = config['window']
    cfg_train = config['training']
    cfg_backtest = config['backtest']
    
    models_to_eval = [
        ("naive_baseline", "baseline", "Naive Persistence", 1),
        ("linear_baseline", "baseline", "Linear Ridge Regression", 1),
        ("vanilla_lstm", "v0", "Vanilla LSTM (v0)", 1),
        ("vanilla_cnn1d", "v0", "Vanilla 1D-CNN (v0)", 1),
        ("deep_lstm", "v1", "Deep LSTM + Technical Indicators (v1)", 1),
        ("temporal_cnn1d", "v1", "Temporal 1D-CNN + Technical Indicators (v1)", 1),
        ("cnn_bilstm_attention", "v2", "Proposed CNN-BiLSTM-Attention (v2 SOTA)", 1),
        ("seq2seq_multistep", "v3", "Seq2Seq Attention Multi-Step (v3)", cfg_win['multi_step_horizon']),
    ]
    
    summary_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*80)
    print("      BẮT ĐẦU ĐÁNH GIÁ VÀ SO SÁNH CÁC PHIÊN BẢN (ABLATION STUDY)")
    print("="*80)
    
    for model_name, version, display_name, forecast_horizon in models_to_eval:
        print(f"\n--- Đang đánh giá: {display_name} ---")
        
        data_bundle = prepare_dataset_pipeline(config, version="v0" if version in ["baseline", "v0"] else "v1")
        input_dim = len(data_bundle['feature_cols'])
        
        train_loader, val_loader, test_loader = build_dataloaders(
            data_bundle=data_bundle,
            input_window=cfg_win['input_window'],
            forecast_horizon=forecast_horizon,
            batch_size=cfg_train['batch_size']
        )
        
        # Lấy nhãn thực tế
        target_scaler = data_bundle['target_scaler']
        y_test_scaled = test_loader.dataset.y.numpy()
        X_test = test_loader.dataset.X.numpy()
        y_prev_scaled = test_loader.dataset.y_prev.numpy()
        
        # Dự báo
        if model_name == "naive_baseline":
            model = build_model_by_name(model_name, input_dim=input_dim, config=config)
            y_pred_scaled = model.predict(X_test)[:, np.newaxis]
        elif model_name == "linear_baseline":
            model = build_model_by_name(model_name, input_dim=input_dim, config=config)
            X_train = train_loader.dataset.X.numpy()
            y_train = train_loader.dataset.y.numpy()
            model.fit(X_train, y_train)
            y_pred_scaled = model.predict(X_test)[:, np.newaxis]
        else:
            model = build_model_by_name(model_name, input_dim=input_dim, config=config, forecast_horizon=forecast_horizon)
            ckpt_path = os.path.join(cfg_train['checkpoint_dir'], f"{model_name}_{version}_best.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                print(f"  [Loaded] Checkpoint: {ckpt_path}")
            else:
                print(f"  [Warning] Checkpoint không tồn tại: {ckpt_path}, dùng trọng số ngẫu nhiên")
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                preds = []
                for batch_data in test_loader:
                    X_b = batch_data[0].to(device)
                    preds.append(model(X_b).cpu().numpy())
                y_pred_scaled = np.vstack(preds)
                if len(y_pred_scaled.shape) == 1:
                    y_pred_scaled = y_pred_scaled[:, np.newaxis]
        
        # Xử lý multi-step: chỉ lấy step đầu tiên (t+1) để so sánh fair
        if forecast_horizon > 1:
            y_pred_for_eval = y_pred_scaled[:, 0:1]  # Chỉ lấy dự báo t+1
            y_test_for_eval = y_test_scaled[:, 0:1] if len(y_test_scaled.shape) > 1 and y_test_scaled.shape[1] > 1 else y_test_scaled
        else:
            y_pred_for_eval = y_pred_scaled
            y_test_for_eval = y_test_scaled
        
        # Đảm bảo y_prev có đúng shape
        if len(y_prev_scaled.shape) == 1:
            y_prev_for_eval = y_prev_scaled[:, np.newaxis]
        else:
            y_prev_for_eval = y_prev_scaled
                    
        # Denormalize về giá trị USD thực tế
        y_true_real = target_scaler.inverse_transform(y_test_for_eval.reshape(-1, 1))
        y_pred_real = target_scaler.inverse_transform(y_pred_for_eval.reshape(-1, 1))
        y_prev_real = target_scaler.inverse_transform(y_prev_for_eval.reshape(-1, 1))
        
        # Tính toán ML Metrics
        ml_res = evaluate_ml_metrics(y_true_real, y_pred_real, y_prev_real)
        
        # Tính toán Financial Backtest Metrics
        fin_res = calculate_financial_metrics(
            actual_prices=y_true_real,
            pred_prices=y_pred_real,
            initial_capital=cfg_backtest['initial_capital'],
            commission=cfg_backtest['commission_fee'],
            slippage=cfg_backtest['slippage'],
            signal_threshold=cfg_backtest['signal_threshold'],
            risk_free_rate=cfg_backtest['risk_free_rate']
        )
        
        row = {
            'Model': display_name,
            'Version': version.upper(),
            'RMSE ($)': round(ml_res['RMSE'], 3),
            'MAE ($)': round(ml_res['MAE'], 3),
            'MAPE (%)': round(ml_res['MAPE'], 2),
            'DA (%)': round(ml_res.get('DA%', 50.0), 2),
            'Total Return (%)': round(fin_res.get('Total_Return%', 0.0), 2),
            'Sharpe Ratio': round(fin_res.get('Sharpe_Ratio', 0.0), 2),
            'Sortino Ratio': round(fin_res.get('Sortino_Ratio', 0.0), 2),
            'Max Drawdown (%)': round(fin_res.get('Max_Drawdown%', 0.0), 2),
            'Win Rate (%)': round(fin_res.get('Win_Rate%', 0.0), 2),
            'Num Trades': fin_res.get('Num_Trades', 0),
        }
        summary_rows.append(row)
        
        # Vẽ biểu đồ kết quả cho tất cả DL models
        if version in ["v0", "v1", "v2", "v3"]:
            plot_and_save_results(
                actual_prices=y_true_real,
                pred_prices=y_pred_real,
                equity_curve=fin_res['Equity_Curve'],
                benchmark_equity=fin_res['Benchmark_Equity'],
                model_name=f"{model_name}_{version}",
                output_dir="output"
            )

    # In bảng tổng hợp kết quả
    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "="*100)
    print("                      BẢNG KẾT QUẢ SO SÁNH CÁC PHIÊN BẢN (NCKH BENCHMARK)")
    print("="*100)
    print(df_summary.to_string(index=False))
    print("="*100)
    
    # Lưu bảng kết quả ra file CSV và Markdown
    os.makedirs("output", exist_ok=True)
    df_summary.to_csv("output/ablation_benchmark_results.csv", index=False)
    try:
        df_summary.to_markdown("output/ablation_benchmark_results.md", index=False)
    except Exception:
        with open("output/ablation_benchmark_results.md", "w", encoding="utf-8") as f:
            f.write(df_summary.to_string(index=False))
    print(f"\n[Lưu trữ] Đã xuất báo cáo và biểu đồ chi tiết vào thư mục: output/")
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/stock.yaml")
    args = parser.parse_args()
    
    evaluate_all_models(args.config)
