"""
Định nghĩa các chỉ số đo lường hiệu năng Machine Learning:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy (DA% - Tỷ lệ dự đoán đúng chiều tăng/giảm)
"""

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0)


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray = None, is_log_return: bool = False) -> float:
    """
    Tính tỷ lệ dự đoán đúng xu hướng (Tăng/Giảm):
    - Nếu is_log_return=True: hướng xác định so với mốc 0 (r > 0 tăng, r < 0 giảm)
      DA = Sum(sign(y_true) == sign(y_pred)) / N * 100%
    - Nếu is_log_return=False: so sánh với y_prev (giá phiên trước)
      DA = Sum(sign(y_true - y_prev) == sign(y_pred - y_prev)) / N * 100%

    LƯU Ý: y_true/y_pred PHẢI ở raw scale (đã inverse_transform nếu dùng target_scaler).
    Truyền giá trị đã StandardScaler-transform sẽ cho kết quả sai khi scaler.mean_ != 0
    (xem training/loss.py DirectionalPenaltyLoss để biết chi tiết).
    Hàm này hiện là dead code trong pipeline thực tế — cli/evaluate.py dùng
    calculate_directional_accuracy_detailed() trong evaluation/stock_metric.py.
    """
    if is_log_return or y_prev is None:
        true_dir = np.sign(y_true)
        pred_dir = np.sign(y_pred)
    else:
        true_dir = np.sign(y_true - y_prev)
        pred_dir = np.sign(y_pred - y_prev)
    correct = np.sum(true_dir == pred_dir)
    total = len(true_dir)
    return float((correct / (total + 1e-9)) * 100.0)


def evaluate_ml_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray = None, is_log_return: bool = False) -> dict:
    metrics = {
        'RMSE': compute_rmse(y_true, y_pred),
        'MAE': compute_mae(y_true, y_pred),
        'MAPE': compute_mape(y_true, y_pred)
    }
    if is_log_return or y_prev is not None:
        metrics['DA%'] = compute_directional_accuracy(y_true, y_pred, y_prev, is_log_return=is_log_return)
    return metrics
