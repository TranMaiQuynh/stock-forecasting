"""
PyTorch Dataset & DataLoader cho dữ liệu chuỗi thời gian chứng khoán.
Hỗ trợ cả dự báo đơn bước (Single-step: t+1) và đa bước (Multi-step: t+1 ... t+k).
Trả thêm y_prev (giá Close phiên trước) để hỗ trợ DirectionalPenaltyLoss.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def augment_time_series(X: np.ndarray, y: np.ndarray, y_prev: np.ndarray,
                         n_augmented: int = 2,
                         noise_std: float = 0.002) -> tuple:
    """
    Tăng cường dữ liệu chuỗi thời gian bằng Jittering.
    Mục đích: Mở rộng tập Train từ ~1170 mẫu lên ~3510 mẫu (n_aug=2)
               mà không làm sai lệch phân phối thị trường.

    Jittering: Thêm nhiễu Gauss cực nhỏ (0.2% độ lệch chuẩn) vào chuỗi dữ liệu,
               mô phỏng sự rung lắc ngẫu nhiên của các lệnh chờ trong phiên giao dịch.

    Args:
        X        : Ma trận đặc trưng (N, seq_len, num_features)
        y        : Nhãn mục tiêu (N, ...)
        y_prev   : Giá trước đó (N, ...) cho DirectionalPenaltyLoss
        n_augmented : Số bản sao augmented (default=2 → tổng gấp 3× tập gốc)
        noise_std   : Độ lệch chuẩn của nhiễu Gauss (default=0.002 = 0.2%)

    Returns:
        X_aug, y_aug, y_prev_aug : Arrays đã được mở rộng
    """
    X_list      = [X]
    y_list      = [y]
    y_prev_list = [y_prev]

    for _ in range(n_augmented):
        noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
        X_list.append(X + noise)
        y_list.append(y)
        y_prev_list.append(y_prev)

    print(f"[Augment] Jittering: {len(X)} → {len(X) * (n_augmented + 1)} mẫu train (noise_std={noise_std})")
    return (
        np.vstack(X_list),
        np.concatenate(y_list, axis=0),
        np.concatenate(y_prev_list, axis=0)
    )


class StockTimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, input_window: int = 60,
                 forecast_horizon: int = 1,
                 use_augmentation: bool = False, noise_std: float = 0.002,
                 n_augmented: int = 2):
        """
        features: Mảng numpy (N, num_features)
        targets: Mảng numpy (N, 1) hoặc (N, target_dim)
        input_window: Số phiên quan sát trong quá khứ (ví dụ: 60 ngày)
        forecast_horizon: Số phiên cần dự báo trong tương lai (1 ngày hoặc 7 ngày)
        use_augmentation: Bật Jittering augmentation (chỉ cho tập Train)
        noise_std: Độ lệch chuẩn nhiễu Gauss (default=0.002)
        n_augmented: Số bản sao tăng cường (default=2 → ×3 mẫu)
        """
        self.features = features
        self.targets = targets
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.use_augmentation = use_augmentation
        self.noise_std = noise_std
        self.n_augmented = n_augmented
        
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

        # [v2-FIX] Áp dụng Jittering Augmentation (chỉ nếu use_augmentation=True)
        if self.use_augmentation and self.n_augmented > 0:
            X_arr, y_arr, y_prev_arr = augment_time_series(
                X_arr, y_arr, y_prev_arr,
                n_augmented=self.n_augmented,
                noise_std=self.noise_std
            )
            
        return (
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32),
            torch.tensor(y_prev_arr, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_prev[idx]


def build_dataloaders(data_bundle: dict, input_window: int = 60, forecast_horizon: int = 1,
                      batch_size: int = 32, use_augmentation: bool = False,
                      noise_std: float = 0.002, n_augmented: int = 2):
    """
    Tạo DataLoader cho cả 3 tập Train, Val, Test.
    Mỗi batch trả về (X, y, y_prev) để hỗ trợ DirectionalPenaltyLoss.

    [v2-FIX] Tham số mới:
    - use_augmentation: Bật Jittering đa dạng hoá dữ liệu cho tập Train
    - noise_std       : Độ lệch chuẩn nhiễu Gauss (default=0.002 = 0.2%)
    - n_augmented     : Số bản sao (default=2 → tổng gấp 3×)
    Note: Augmentation chỉ áp dụng trên tập Train, Val/Test giữ nguyên.
    """
    feature_cols = data_bundle.get('feature_cols', [])
    # close_idx đã được xóa: Close không còn trong feature_cols sau khi stationary fix.
    
    train_dataset = StockTimeSeriesDataset(
        features=data_bundle['train_features'],
        targets=data_bundle['train_target'],
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        use_augmentation=use_augmentation,  # Chỉ train mới augment
        noise_std=noise_std,
        n_augmented=n_augmented,
    )
    val_dataset = StockTimeSeriesDataset(
        features=data_bundle['val_features'],
        targets=data_bundle['val_target'],
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        use_augmentation=False,             # Val không augment
    )
    test_dataset = StockTimeSeriesDataset(
        features=data_bundle['test_features'],
        targets=data_bundle['test_target'],
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        use_augmentation=False,             # Test không augment
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader
