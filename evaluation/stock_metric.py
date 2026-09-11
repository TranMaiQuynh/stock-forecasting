"""
Hệ thống Đánh giá Kép (Dual Evaluation System) cho NCKH và Quỹ Đầu tư:
1. Đánh giá Thống kê / Machine Learning (RMSE, MAE, MAPE, R2, Directional Accuracy).
2. Đánh giá Tài chính & Mô phỏng Giao dịch Thực tế (Backtesting Engine):
   - Tính toán đầy đủ Phí giao dịch (Commission) và Trượt giá (Slippage).
   - Đo lường: Sharpe Ratio, Sortino Ratio, Max Drawdown (MDD), Calmar Ratio, Profit Factor, Win Rate.
   - So sánh trực tiếp với chiến lược Buy & Hold thị trường.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_financial_metrics(actual_prices: np.ndarray, pred_prices: np.ndarray, 
                                initial_capital: float = 100000.0,
                                commission: float = 0.001, 
                                slippage: float = 0.0005,
                                signal_threshold: float = 0.002,
                                risk_free_rate: float = 0.04) -> dict:
    """
    Mô phỏng chiến lược giao dịch định lượng dựa trên tín hiệu dự báo của mô hình.
    """
    actual_prices = actual_prices.flatten()
    pred_prices = pred_prices.flatten()
    n_days = len(actual_prices)
    
    if n_days < 2:
        return _empty_metrics(initial_capital, n_days)
        
    # Tính lợi nhuận thực tế theo phiên
    actual_returns = np.diff(actual_prices) / actual_prices[:-1]
    # Tỷ lệ dự báo tăng/giảm so với giá hiện tại
    pred_returns = (pred_prices[1:] - actual_prices[:-1]) / actual_prices[:-1]
    
    # Tạo tín hiệu giao dịch: 1 (Long/Mua), 0 (Cash/Giữ tiền mặt)
    positions = np.where(pred_returns > signal_threshold, 1.0, 0.0)
    
    # Kiểm tra edge case: nếu model không giao dịch lần nào
    num_active_positions = np.sum(positions > 0)
    
    # Tính toán chi phí giao dịch khi đổi vị thế (Turnover)
    trades = np.abs(np.diff(np.insert(positions, 0, 0.0)))
    total_cost_per_trade = commission + slippage
    trading_costs = trades * total_cost_per_trade
    
    # Lợi nhuận chiến lược sau khi trừ phí và trượt giá
    strategy_returns = (positions * actual_returns) - trading_costs
    
    # Đường cong tài sản (Equity Curve)
    equity_curve = initial_capital * np.cumprod(1.0 + strategy_returns)
    equity_curve = np.insert(equity_curve, 0, initial_capital)
    
    # Benchmark: Chiến lược Buy & Hold (Mua và nắm giữ)
    benchmark_returns = actual_returns
    benchmark_equity = initial_capital * np.cumprod(1.0 + benchmark_returns)
    benchmark_equity = np.insert(benchmark_equity, 0, initial_capital)
    
    # 1. Tổng lợi nhuận (Total Return)
    total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100.0
    benchmark_total_return = (benchmark_equity[-1] - initial_capital) / initial_capital * 100.0
    
    # 2. Lợi nhuận hàng năm (CAGR - 252 ngày giao dịch/năm)
    years = max(n_days / 252.0, 0.01)
    cagr = ((equity_curve[-1] / initial_capital) ** (1.0 / years) - 1.0) * 100.0
    
    # 3. Biến động hàng năm (Annualized Volatility)
    daily_vol = np.std(strategy_returns)
    annual_vol = daily_vol * np.sqrt(252) * 100.0
    
    # 4. Sharpe Ratio — Xử lý edge case khi không giao dịch
    daily_rf = risk_free_rate / 252.0
    excess_returns = strategy_returns - daily_rf
    
    if num_active_positions == 0 or daily_vol < 1e-10:
        # Model không giao dịch → Sharpe/Sortino = 0 (không có ý nghĩa)
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    else:
        sharpe_ratio = (np.mean(excess_returns) / daily_vol) * np.sqrt(252)
        
        # 5. Sortino Ratio (chỉ phạt biến động giảm)
        downside_returns = strategy_returns[strategy_returns < daily_rf] - daily_rf
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            sortino_ratio = (np.mean(excess_returns) / (downside_std + 1e-9)) * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio  # Không có phiên lỗ → tương đương Sharpe
    
    # 6. Maximum Drawdown (MDD)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) * 100.0
    
    # 7. Calmar Ratio
    calmar_ratio = cagr / (max_drawdown + 1e-9) if max_drawdown > 0 else 0.0
    
    # 8. Win Rate & Profit Factor — chỉ tính trên các phiên có giao dịch
    active_returns = strategy_returns[positions > 0] if num_active_positions > 0 else strategy_returns
    winning_trades = active_returns[active_returns > 0]
    losing_trades = active_returns[active_returns < 0]
    
    if num_active_positions > 0:
        win_rate = (len(winning_trades) / num_active_positions) * 100.0
    else:
        win_rate = 0.0
    
    gross_profits = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
    gross_losses = np.abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 1e-9
    profit_factor = gross_profits / gross_losses

    return {
        'Total_Return%': total_return,
        'Benchmark_Return%': benchmark_total_return,
        'CAGR%': cagr,
        'Annual_Vol%': annual_vol,
        'Sharpe_Ratio': float(sharpe_ratio),
        'Sortino_Ratio': float(sortino_ratio),
        'Max_Drawdown%': float(max_drawdown),
        'Calmar_Ratio': float(calmar_ratio),
        'Win_Rate%': float(win_rate),
        'Profit_Factor': float(profit_factor),
        'Num_Trades': int(num_active_positions),
        'Equity_Curve': equity_curve,
        'Benchmark_Equity': benchmark_equity
    }


def _empty_metrics(initial_capital, n_days):
    """Trả về dict metrics mặc định khi dữ liệu quá ít."""
    return {
        'Total_Return%': 0.0, 'Benchmark_Return%': 0.0, 'CAGR%': 0.0,
        'Annual_Vol%': 0.0, 'Sharpe_Ratio': 0.0, 'Sortino_Ratio': 0.0,
        'Max_Drawdown%': 0.0, 'Calmar_Ratio': 0.0, 'Win_Rate%': 0.0,
        'Profit_Factor': 0.0, 'Num_Trades': 0,
        'Equity_Curve': np.array([initial_capital]),
        'Benchmark_Equity': np.array([initial_capital])
    }


def plot_and_save_results(actual_prices: np.ndarray, pred_prices: np.ndarray, 
                          equity_curve: np.ndarray, benchmark_equity: np.ndarray, 
                          model_name: str, output_dir: str = "output"):
    """
    Vẽ biểu đồ giá thực tế vs dự báo và biểu đồ đường cong tài sản (Equity Curve).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Biểu đồ 1: Giá Thực tế vs Dự báo
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Giá Thực Tế (Actual Close)", color="#1f77b4", linewidth=1.5)
    plt.plot(pred_prices, label=f"Dự Báo ({model_name})", color="#ff7f0e", linestyle="--", linewidth=1.5)
    plt.title(f"So Sánh Giá Thực Tế vs Dự Báo - {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Phiên Giao Dịch (Test Days)", fontsize=12)
    plt.ylabel("Giá Cổ Phiếu ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    pred_chart_path = os.path.join(output_dir, f"{model_name}_prediction.png")
    plt.savefig(pred_chart_path, dpi=300)
    plt.close()
    
    # Biểu đồ 2: Equity Curve vs Buy & Hold
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label=f"Chiến Lược ({model_name})", color="#2ca02c", linewidth=2.0)
    plt.plot(benchmark_equity, label="Benchmark (Buy & Hold)", color="#7f7f7f", linestyle="--", linewidth=1.5)
    plt.title(f"Hiệu Quả Sinh Lời Thực Chiến (Equity Curve) - {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Phiên Giao Dịch (Test Days)", fontsize=12)
    plt.ylabel("Giá Trị Danh Mục ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    equity_chart_path = os.path.join(output_dir, f"{model_name}_equity_curve.png")
    plt.savefig(equity_chart_path, dpi=300)
    plt.close()
    
    return pred_chart_path, equity_chart_path


def calculate_directional_accuracy_detailed(y_true, y_pred, y_prev=None, is_log_return: bool = True):
    """[v2-FIX] DA chi tiet: loc bo ngay dung gia, phan tich rieng Tang/Giam.
    - is_log_return=True: huong tang/giam xac dinh so voi moc 0 (sign(y)).
    - is_log_return=False: so sanh voi y_prev (sign(y - y_prev)).
    Returns: dict voi DA%, DA_filtered%, Up_correct%, Down_correct%.
    """
    import numpy as _np
    y_true = _np.array(y_true).flatten()
    y_pred = _np.array(y_pred).flatten()
    if is_log_return or y_prev is None:
        true_dir = _np.sign(y_true)
        pred_dir = _np.sign(y_pred)
    else:
        y_prev = _np.array(y_prev).flatten()
        true_dir = _np.sign(y_true - y_prev)
        pred_dir = _np.sign(y_pred - y_prev)
    total = len(true_dir)
    da_all = float((true_dir == pred_dir).mean() * 100.0)
    valid_mask = true_dir != 0
    filtered = int(valid_mask.sum())
    da_filtered = (
        float((true_dir[valid_mask] == pred_dir[valid_mask]).mean() * 100.0)
        if filtered > 0 else 0.0
    )
    up_mask = true_dir == 1
    down_mask = true_dir == -1
    up_correct = float((pred_dir[up_mask] == 1).mean() * 100.0) if up_mask.sum() > 0 else 0.0
    down_correct = float((pred_dir[down_mask] == -1).mean() * 100.0) if down_mask.sum() > 0 else 0.0
    return {
        "DA%": round(da_all, 2),
        "DA_filtered%": round(da_filtered, 2),
        "Up_correct%": round(up_correct, 2),
        "Down_correct%": round(down_correct, 2),
        "Total_samples": total,
        "Filtered_samples": filtered,
    }


def run_cross_ticker_summary(results_list, output_dir="output"):
    """[v2-FIX] Tong hop ket qua Mean +- Std theo nhieu tickers x seeds (chuan NCKH).
    Moi dict trong results_list phai co: model, ticker, seed + cac metric columns.
    """
    import os as _os
    import pandas as _pd
    _os.makedirs(output_dir, exist_ok=True)
    df = _pd.DataFrame(results_list)
    exclude = {"ticker", "seed", "model", "version"}
    metric_cols = [c for c in df.columns if c not in exclude]
    grouped = df.groupby(["model", "ticker"])[metric_cols]
    mean_df = grouped.mean().round(3)
    std_df = grouped.std().round(3).fillna(0)
    combined = {}
    for col in metric_cols:
        combined[col] = mean_df[col].astype(str) + " +- " + std_df[col].astype(str)
    summary_df = _pd.DataFrame(combined)
    path = _os.path.join(output_dir, "cross_ticker_summary.md")
    try:
        overall = df.groupby("model")[metric_cols].agg(["mean", "std"]).round(3)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Cross-Ticker Results Summary (Mean +- Std)\n")
            f.write("> Chuan bao cao NCKH: moi o = Mean +- Std tren nhieu seeds\n\n")
            f.write("## Chi tiet theo (Model x Ticker)\n")
            f.write(summary_df.to_markdown())
            f.write("\n\n## Tong hop theo Model\n")
            f.write(overall.to_markdown())
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary_df.to_string())
    print(f"[CrossTicker] Da luu bang Mean+-Std -> {path}")
    return summary_df
