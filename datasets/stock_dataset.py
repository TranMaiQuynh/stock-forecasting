"""
PyTorch Dataset & DataLoader cho dữ liệu chuỗi thời gian chứng khoán.
Hỗ trợ cả dự báo đơn bước (Single-step: t+1) và đa bước (Multi-step: t+1 ... t+k).
Trả thêm y_prev (giá Close phiên trước) để hỗ trợ DirectionalPenaltyLoss.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class StockTimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, input_window: int = 60,
                 forecast_horizon: int = 1, close_feature_idx: int = None):
        """
        features: Mảng numpy (N, num_features)
        targets: Mảng numpy (N, 1) hoặc (N, target_dim)
        input_window: Số phiên quan sát trong quá khứ (ví dụ: 60 ngày)
        forecast_horizon: Số phiên cần dự báo trong tương lai (1 ngày hoặc 7 ngày)
        close_feature_idx: Index của cột Close trong features (để trích y_prev cho Directional Loss)
        """
        self.features = features
        self.targets = targets
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.close_feature_idx = close_feature_idx
        
        self.X, self.y, self.y_prev = self._create_sliding_windows()

    def _create_sliding_windows(self):
        X_list, y_list, y_prev_list = [], [], []
        total_samples = len(self.features) - self.input_window - self.forecast_horizon + 1
        
        if total_samples <= 0:
            raise ValueError(
                f"Kích thước dữ liệu ({len(self.features)}) quá nhỏ so với cửa sổ (window={self.input_window} + horizon={self.forecast_horizon})"
            )
            
        for i in range(total_samples):
            # Input window: từ i đến i + input_window
            x_window = self.features[i : i + self.input_window]
            
            # Target window: từ i + input_window đến i + input_window + forecast_horizon
            if self.forecast_horizon == 1:
                y_target = self.targets[i + self.input_window]
            else:
                y_target = self.targets[i + self.input_window : i + self.input_window + self.forecast_horizon]
            
            # y_prev: giá target tại bước cuối cùng của input window (phiên gần nhất trước dự báo)
            y_prev = self.targets[i + self.input_window - 1]
                
            X_list.append(x_window)
            y_list.append(y_target)
            y_prev_list.append(y_prev)
            
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        y_prev_arr = np.array(y_prev_list, dtype=np.float32)
        
        # Nếu target 1D thì định dạng lại
        if self.forecast_horizon == 1 and len(y_arr.shape) > 2:
            y_arr = y_arr.squeeze(-1)
        elif self.forecast_horizon > 1 and len(y_arr.shape) == 3:
            y_arr = y_arr.squeeze(-1)
            
        return (
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32),
            torch.tensor(y_prev_arr, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_prev[idx]


def build_dataloaders(data_bundle: dict, input_window: int = 60, forecast_horizon: int = 1, batch_size: int = 32):
    """
    Tạo DataLoader cho cả 3 tập Train, Val, Test.
    Mỗi batch trả về (X, y, y_prev) để hỗ trợ DirectionalPenaltyLoss.
    """
    # Xác định index cột Close trong feature_cols
    feature_cols = data_bundle.get('feature_cols', [])
    close_idx = feature_cols.index('Close') if 'Close' in feature_cols else 3
    
    train_dataset = StockTimeSeriesDataset(
        features=data_bundle['train_features'],
        targets=data_bundle['train_target'],
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        close_feature_idx=close_idx
    )
    val_dataset = StockTimeSeriesDataset(
        features=data_bundle['val_features'],
        targets=data_bundle['val_target'],
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        close_feature_idx=close_idx
    )
    test_dataset = StockTimeSeriesDataset(
        features=data_bundle['test_features'],
        targets=data_bundle['test_target'],
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        close_feature_idx=close_idx
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader
