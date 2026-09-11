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

    Chế độ Close (use_log_return=False):
        Loss = Base_Loss + penalty_weight * mean(max(0, -sign(y_true - y_prev) * (y_pred - y_prev)))

    Chế độ Log_Return (use_log_return=True):
        Loss = Base_Loss + penalty_weight * mean(max(0, -sign(y_true) * y_pred))
        → Giá tăng khi r_t > 0, giảm khi r_t < 0. Mốc chuẩn là 0, không cần y_prev.
    """
    def __init__(self, penalty_weight: float = 0.5, base_loss: str = "huber",
                 use_log_return: bool = False):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.use_log_return = use_log_return
        if base_loss == "huber":
            self.base_criterion = nn.SmoothL1Loss()
        else:
            self.base_criterion = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_prev: torch.Tensor = None):
        base_loss = self.base_criterion(y_pred, y_true)
        
        if self.penalty_weight <= 0:
            return base_loss

        if self.use_log_return:
            # [v2-FIX] Log_Return mode: hướng = sign(r_true), penalty khi đoán sai dấu
            # r_true > 0 → giá tăng, r_true < 0 → giá giảm
            # Nếu sign(r_true) * r_pred < 0 → đoán sai hướng → bị phạt
            true_direction = torch.sign(y_true)
            penalty = F.relu(-true_direction * y_pred)
        else:
            # Chế độ Close gốc: hướng = sign(y_true - y_prev)
            if y_prev is None:
                return base_loss
            true_direction = torch.sign(y_true - y_prev)
            pred_diff = y_pred - y_prev
            penalty = F.relu(-true_direction * pred_diff)

        total_loss = base_loss + self.penalty_weight * torch.mean(penalty)
        return total_loss


def get_loss_function(loss_type: str = "huber", penalty_weight: float = 0.5,
                      use_log_return: bool = False):
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
        )
    else:
        raise ValueError(f"Loại loss không hợp lệ: {loss_type}")

