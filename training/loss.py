"""
Các hàm mất mát (Loss functions) chuyên dụng cho chuỗi thời gian tài chính.
Bao gồm DirectionalPenaltyLoss (Hàm loss phạt đoán sai chiều xu hướng - Điểm sáng NCKH).

[v2-FIX] Hỗ trợ 2 chế độ:
- use_log_return=False (target=Close): hướng = sign(y_true − y_prev)
- use_log_return=True  (target=Log_Return): hướng = sign(y_true), mốc chuẩn = 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalPenaltyLoss(nn.Module):
    """
    Hàm Loss kết hợp giữa Huber/MSE Loss và hình phạt sai hướng (Directional Penalty).

    Chế độ Log_Return (use_log_return=True):
        y_pred/y_true nhận được trong forward() đã qua StandardScaler.fit_transform().
        sign(y_true_scaled) ≠ sign(y_true_raw) khi scaler.mean_ ≠ 0 (luôn đúng với Log_Return thật).
        Cần so sánh với zero_point = (0 - mean_) / scale_ — giá trị scaled của "raw = 0" (giá không đổi).
        zero_point được trao từ ngoài (StockTrainer.__init__ tính từ target_scaler) — không có magic number.

    Chế độ Close (use_log_return=False):
        Hướng = sign(y_true - y_prev). Cả hai cùng scale nên độ lệch mean bị triệt tiêu khi trừ nhau.
        zero_point không được dùng trong nhánh này.
    """
    def __init__(self, penalty_weight: float = 0.5, base_loss: str = "huber",
                 use_log_return: bool = False, zero_point: float = 0.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.use_log_return = use_log_return
        self.zero_point = zero_point   # scaled value of raw=0; computed from target_scaler outside
        if base_loss == "huber":
            self.base_criterion = nn.SmoothL1Loss()
        else:
            self.base_criterion = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_prev: torch.Tensor = None):
        base_loss = self.base_criterion(y_pred, y_true)
        
        if self.penalty_weight <= 0:
            return base_loss

        if self.use_log_return:
            # So sánh với zero_point (không phải 0 tuyệt đối) — sửa lỗi lệch hướng do StandardScaler
            zp = torch.tensor(self.zero_point, dtype=y_true.dtype, device=y_true.device)
            true_direction = torch.sign(y_true - zp)
            penalty = F.relu(-true_direction * (y_pred - zp))
        else:
            # Chế độ Close: hướng = sign(y_true - y_prev), cả hai cùng scale nên đúng
            if y_prev is None:
                return base_loss
            true_direction = torch.sign(y_true - y_prev)
            pred_diff = y_pred - y_prev
            penalty = F.relu(-true_direction * pred_diff)

        total_loss = base_loss + self.penalty_weight * torch.mean(penalty)
        return total_loss


def get_loss_function(loss_type: str = "huber", penalty_weight: float = 0.5,
                      use_log_return: bool = False, zero_point: float = 0.0):
    """
    Tạo hàm loss theo loại đã chọn.

    zero_point: chỉ có hiệu lực khi loss_type='directional' và use_log_return=True.
    Giá trị này là (0 - target_scaler.mean_[0]) / target_scaler.scale_[0]
    (giá trị 'raw = 0' quy đổi sang không gian đã scale — không phải magic number).
    """
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.SmoothL1Loss()
    elif loss_type == "directional":
        return DirectionalPenaltyLoss(
            penalty_weight=penalty_weight,
            base_loss="huber",
            use_log_return=use_log_return,
            zero_point=zero_point,
        )
    else:
        raise ValueError(f"Loại loss không hợp lệ: {loss_type}")

