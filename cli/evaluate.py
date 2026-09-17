"""
CLI Đánh giá & So sánh Toàn diện (Comprehensive Benchmark & Ablation Evaluation):
- So sánh Base Model vs. v0 vs. v1 vs. v2 vs. v3
- Hệ thống đo lường kép (Dual-Metrics) chuẩn NCKH:
  + Return Metrics (Không gian Log Return): Ret RMSE, Ret MAE, DA (%)
  + Reconstructed Price Metrics (Không gian giá USD): Price RMSE ($), Price MAE ($), Price MAPE (%)
- Phân tích tài chính thực chiến (Backtesting Engine):
  + Total Return (%), Sharpe Ratio, Sortino Ratio, Max Drawdown (%), Win Rate (%), Num Trades
- Tự động nạp checkpoint chuẩn từ checkpoints/{TICKER}/{model}_{version}_seed{seed}_best.pt
- Hỗ trợ đánh giá đơn mã cổ phiếu hoặc toàn bộ danh mục (--ticker all)
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
from evaluation.stock_metric import (
    calculate_financial_metrics,
    plot_and_save_results,
    calculate_directional_accuracy_detailed,
    run_cross_ticker_summary
)
from training.metric import evaluate_ml_metrics


def evaluate_single_ticker_models(
    config: dict,
    ticker: str = "AAPL",
    seed: int = 42,
    threshold: float = None,
    output_dir: str = "output"
):
    """
    Đánh giá toàn bộ 8 mô hình (Ablation Study) cho một mã cổ phiếu và một random seed.
    Tự động tái cấu trúc giá USD từ Log Return và tính toán song song ML & Financial Metrics.
    """
    cfg_win = config['window']
    cfg_train = config['training']
    cfg_backtest = config['backtest']
    sig_thresh = threshold if threshold is not None else cfg_backtest.get('signal_threshold', 0.0005)
    
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
    
    print("\n" + "="*90)
    print(f"   📊 ĐÁNH GIÁ VÀ SO SÁNH CÁC PHIÊN BẢN (ABLATION BENCHMARK) — {ticker.upper()} (Seed={seed})")
    print(f"   🎯 Ngưỡng Backtest: {sig_thresh*100:.3f}% (Log Return signal)")
    print("="*90)
    
    cfg_ticker = dict(config)
    cfg_ticker['data'] = dict(config['data'])
    cfg_ticker['data']['ticker'] = ticker
    
    for model_name, version, display_name, forecast_horizon in models_to_eval:
        print(f"\n--- Đang đánh giá: {display_name} ({ticker} | Seed {seed}) ---")
        
        v_data = "v0" if version in ["baseline", "v0"] else "v1"
        data_bundle = prepare_dataset_pipeline(cfg_ticker, version=v_data)
        input_dim = len(data_bundle['feature_cols'])
        feature_cols = data_bundle['feature_cols']
        target_scaler = data_bundle['target_scaler']
        test_df = data_bundle['test_df']
        use_log_return = data_bundle.get('use_log_return', False)
        
        train_loader, val_loader, test_loader = build_dataloaders(
            data_bundle=data_bundle,
            input_window=cfg_win['input_window'],
            forecast_horizon=forecast_horizon,
            batch_size=cfg_train['batch_size']
        )
        
        N_test = len(test_loader.dataset)
        W = cfg_win['input_window']
        test_close_prev = test_df['Close'].iloc[W - 1 : W - 1 + N_test].values
        test_close_true = test_df['Close'].iloc[W : W + N_test].values
        
        y_test_scaled = test_loader.dataset.y.numpy()
        X_test = test_loader.dataset.X.numpy()
        
        # 1. Dự báo
        if model_name == "naive_baseline":
            target_idx = feature_cols.index(data_bundle['target_col']) if data_bundle['target_col'] in feature_cols else -1
            model = build_model_by_name(model_name, input_dim=input_dim, config=config, target_feature_idx=target_idx)
            y_pred_scaled = model.predict(X_test)[:, np.newaxis]
        elif model_name == "linear_baseline":
            model = build_model_by_name(model_name, input_dim=input_dim, config=config)
            X_train = train_loader.dataset.X.numpy()
            y_train = train_loader.dataset.y.numpy()
            model.fit(X_train, y_train)
            y_pred_scaled = model.predict(X_test)[:, np.newaxis]
        else:
            model = build_model_by_name(model_name, input_dim=input_dim, config=config, forecast_horizon=forecast_horizon)
            
            # Tìm checkpoint theo thứ tự ưu tiên
            ckpt_candidates = [
                os.path.join(cfg_train['checkpoint_dir'], ticker, f"{model_name}_{version}_seed{seed}_best.pt"),
                os.path.join(cfg_train['checkpoint_dir'], ticker, f"{model_name}_seed{seed}_best.pt"),
                os.path.join(cfg_train['checkpoint_dir'], f"{model_name}_{version}_best.pt"),
                os.path.join(cfg_train['checkpoint_dir'], f"{model_name}_best.pt"),
            ]
            ckpt_path = None
            for cand in ckpt_candidates:
                if os.path.exists(cand):
                    try:
                        state_dict = torch.load(cand, map_location=device, weights_only=True)
                        model.load_state_dict(state_dict)
                        ckpt_path = cand
                        print(f"  [Loaded] Checkpoint: {ckpt_path}")
                        break
                    except Exception as e:
                        print(f"  [Warning] Bỏ qua {cand} do không khớp kiến trúc: {e}")
                    
            if ckpt_path is None:
                print(f"  [Warning] Không tìm thấy checkpoint hợp lệ cho {model_name}_{version}, dùng trọng số khởi tạo!")
                
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
        
        # Xử lý multi-step: chỉ lấy step đầu tiên (t+1) để so sánh chuẩn hóa
        if forecast_horizon > 1:
            y_pred_for_eval = y_pred_scaled[:, 0:1]
            y_test_for_eval = y_test_scaled[:, 0:1] if len(y_test_scaled.shape) > 1 and y_test_scaled.shape[1] > 1 else y_test_scaled
        else:
            y_pred_for_eval = y_pred_scaled
            y_test_for_eval = y_test_scaled
            
        # Denormalize
        y_pred_nat = target_scaler.inverse_transform(y_pred_for_eval.reshape(-1, 1)).flatten()
        y_true_nat = target_scaler.inverse_transform(y_test_for_eval.reshape(-1, 1)).flatten()
        
        # 2. Tính toán Dual Metrics (Return & Reconstructed Price)
        if use_log_return:
            r_pred = y_pred_nat
            r_true = y_true_nat
            
            # Return metrics
            rmse_ret = float(np.sqrt(np.mean((r_true - r_pred)**2)))
            mae_ret = float(np.mean(np.abs(r_true - r_pred)))
            da_dict = calculate_directional_accuracy_detailed(r_true, r_pred, is_log_return=True)
            da = da_dict['DA%']
            
            # Reconstruct prices (USD): P_hat_t = P_{t-1} * exp(r_hat_t)
            pred_prices = test_close_prev * np.exp(r_pred)
            actual_prices = test_close_true
            rmse_price = float(np.sqrt(np.mean((actual_prices - pred_prices)**2)))
            mae_price = float(np.mean(np.abs(actual_prices - pred_prices)))
            mape_price = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100.0)
        else:
            pred_prices = y_pred_nat
            actual_prices = y_true_nat
            rmse_ret = 0.0
            mae_ret = 0.0
            rmse_price = float(np.sqrt(np.mean((actual_prices - pred_prices)**2)))
            mae_price = float(np.mean(np.abs(actual_prices - pred_prices)))
            mape_price = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100.0)
            da = float(np.mean(np.sign(actual_prices - test_close_prev) == np.sign(pred_prices - test_close_prev)) * 100.0)
        
        # 3. Financial Backtesting
        # Ghép thêm phiên t-1 khởi điểm để đánh giá trọn vẹn toàn bộ N_test phiên
        act_p_full = np.insert(actual_prices, 0, test_close_prev[0])
        prd_p_full = np.insert(pred_prices, 0, test_close_prev[0])
        
        fin_res = calculate_financial_metrics(
            actual_prices=act_p_full,
            pred_prices=prd_p_full,
            initial_capital=cfg_backtest['initial_capital'],
            commission=cfg_backtest['commission_fee'],
            slippage=cfg_backtest['slippage'],
            signal_threshold=sig_thresh,
            risk_free_rate=cfg_backtest['risk_free_rate']
        )
        
        row = {
            'Ticker': ticker,
            'Seed': seed,
            'Model': display_name,
            'Version': version.upper(),
            'Ret RMSE': round(rmse_ret, 4),
            'Ret MAE': round(mae_ret, 4),
            'DA (%)': round(da, 2),
            'Price RMSE ($)': round(rmse_price, 2),
            'Price MAE ($)': round(mae_price, 2),
            'Price MAPE (%)': round(mape_price, 2),
            'Total Return (%)': round(fin_res.get('Total_Return%', 0.0), 2),
            'Sharpe Ratio': round(fin_res.get('Sharpe_Ratio', 0.0), 2),
            'Sortino Ratio': round(fin_res.get('Sortino_Ratio', 0.0), 2),
            'Max Drawdown (%)': round(fin_res.get('Max_Drawdown%', 0.0), 2),
            'Win Rate (%)': round(fin_res.get('Win_Rate%', 0.0), 2),
            'Num Trades': fin_res.get('Num_Trades', 0),
        }
        summary_rows.append(row)
        
        # Vẽ biểu đồ giá thực tế vs dự báo và Equity Curve
        if version in ["v0", "v1", "v2", "v3"]:
            chart_prefix = f"{ticker.upper()}_{model_name}_{version}"
            plot_and_save_results(
                actual_prices=actual_prices,
                pred_prices=pred_prices,
                equity_curve=fin_res['Equity_Curve'],
                benchmark_equity=fin_res['Benchmark_Equity'],
                model_name=chart_prefix,
                output_dir=output_dir
            )
            # Lưu bản sao tên chung nếu là ticker mặc định và seed 42
            if ticker.upper() == "AAPL" and seed == 42:
                plot_and_save_results(
                    actual_prices=actual_prices,
                    pred_prices=pred_prices,
                    equity_curve=fin_res['Equity_Curve'],
                    benchmark_equity=fin_res['Benchmark_Equity'],
                    model_name=f"{model_name}_{version}",
                    output_dir=output_dir
                )

    df_summary = pd.DataFrame(summary_rows)
    display_cols = [c for c in df_summary.columns if c not in ['Ticker', 'Seed']]
    
    print("\n" + "="*110)
    print(f"      BẢNG KẾT QUẢ SO SÁNH CÁC PHIÊN BẢN (NCKH BENCHMARK) — {ticker.upper()} (Seed={seed})")
    print("="*110)
    print(df_summary[display_cols].to_string(index=False))
    print("="*110)
    
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"ablation_benchmark_results_{ticker}.csv")
    out_md = os.path.join(output_dir, f"ablation_benchmark_results_{ticker}.md")
    df_summary[display_cols].to_csv(out_csv, index=False)
    try:
        df_summary[display_cols].to_markdown(out_md, index=False)
    except Exception:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(df_summary[display_cols].to_string(index=False))
            
    if ticker.upper() == "AAPL" and seed == 42:
        df_summary[display_cols].to_csv(os.path.join(output_dir, "ablation_benchmark_results.csv"), index=False)
        try:
            df_summary[display_cols].to_markdown(os.path.join(output_dir, "ablation_benchmark_results.md"), index=False)
        except Exception:
            with open(os.path.join(output_dir, "ablation_benchmark_results.md"), "w", encoding="utf-8") as f:
                f.write(df_summary[display_cols].to_string(index=False))
                
    return df_summary, summary_rows


def evaluate_all_models(
    config_path: str = "config/stock.yaml",
    target_ticker: str = "AAPL",
    target_seed: int = 42,
    threshold: float = None,
    output_dir: str = "output"
):
    config = load_config(config_path)
    all_tickers = config.get('data', {}).get('training_tickers', ["AAPL", "TSLA", "MSFT"])
    all_seeds = config.get('data', {}).get('training_seeds', [42, 100])
    
    if target_ticker == "all":
        print("\n" + "="*80)
        print(f"   🌐 ĐÁNH GIÁ TOÀN BỘ DANH MỤC: {all_tickers} | SEEDS: {all_seeds}")
        print("="*80)
        all_results = []
        for t in all_tickers:
            for s in all_seeds:
                _, rows = evaluate_single_ticker_models(
                    config=config, ticker=t, seed=s, threshold=threshold, output_dir=output_dir
                )
                all_results.extend(rows)
                
        # Tổng hợp Mean +- Std NCKH
        run_cross_ticker_summary(all_results, output_dir=output_dir)
        print(f"\n[Hoàn tất] Đã tổng hợp toàn bộ kết quả vào thư mục: {output_dir}/")
    else:
        evaluate_single_ticker_models(
            config=config,
            ticker=target_ticker,
            seed=target_seed,
            threshold=threshold,
            output_dir=output_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá & So sánh Toàn diện mô hình dự báo Log Return và Backtesting"
    )
    parser.add_argument("--config", type=str, default="config/stock.yaml")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Mã cổ phiếu: AAPL | TSLA | MSFT | all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed: 42 | 100")
    parser.add_argument("--threshold", type=float, default=None, help="Ghi đè signal threshold backtest")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    
    evaluate_all_models(
        config_path=args.config,
        target_ticker=args.ticker,
        target_seed=args.seed,
        threshold=args.threshold,
        output_dir=args.output_dir
    )

