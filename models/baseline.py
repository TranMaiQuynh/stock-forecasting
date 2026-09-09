"""
Các mô hình Base Model (Baseline) để làm mốc đối chứng khoa học tối thiểu (Sanity Check).
Bao gồm: Naive Persistence Model (Bước đi ngẫu nhiên) và Linear Regression Baseline.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge


class NaivePersistenceModel:
    """
    Naive Baseline: Dự báo giá ngày mai bằng đúng giá hôm nay.
    y_hat_{t+1} = y_t
    """
    def __init__(self, target_feature_idx: int = 3):
        # Mặc định index 3 là cột 'Close' trong OHLCV
        self.target_feature_idx = target_feature_idx

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, window_size, num_features)
        Trả về giá trị của phiên cuối cùng trong cửa sổ (phiên t)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        # Lấy timestep cuối cùng [-1] của feature mục tiêu
        return X[:, -1, self.target_feature_idx]


class LinearRegressionBaseline:
    """
    Statistical Baseline: Mô hình hồi quy Ridge trên toàn bộ flattened window.
    """
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
            
        N, W, F = X.shape
        X_flat = X.reshape(N, W * F)
        self.model.fit(X_flat, y.ravel())

    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        N, W, F = X.shape
        X_flat = X.reshape(N, W * F)
        return self.model.predict(X_flat)
