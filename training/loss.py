"""
Các hàm mất mát (Loss functions) chuyên dụng cho chuỗi thời gian tài chính.
Bao gồm DirectionalPenaltyLoss (Hàm loss phạt đoán sai chiều xu hướng - Điểm sáng NCKH).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalPenaltyLoss(nn.Module):
    """
    Hàm Loss kết hợp giữa Huber/MSE Loss và hình phạt sai hướng (Directional Penalty).
    Loss = Base_Loss + penalty_weight * mean(max(0, - sign(y_true - y_prev) * (y_pred - y_prev)))
    """
    def __init__(self, penalty_weight: float = 0.5, base_loss: str = "huber"):
        super().__init__()
        self.penalty_weight = penalty_weight
        if base_loss == "huber":
            self.base_criterion = nn.SmoothL1Loss()
        else:
            self.base_criterion = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_prev: torch.Tensor = None):
        base_loss = self.base_criterion(y_pred, y_true)
        
        if y_prev is None or self.penalty_weight <= 0:
            return base_loss
            
        true_direction = torch.sign(y_true - y_prev)
        pred_diff = y_pred - y_prev
        
        # Nếu tích true_direction * pred_diff < 0 => đoán ngược hướng
        penalty = F.relu(-true_direction * pred_diff)
        total_loss = base_loss + self.penalty_weight * torch.mean(penalty)
        return total_loss


def get_loss_function(loss_type: str = "huber", penalty_weight: float = 0.5):
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return nn.SmoothL1Loss()
    elif loss_type == "directional":
        return DirectionalPenaltyLoss(penalty_weight=penalty_weight, base_loss="huber")
    else:
        raise ValueError(f"Loại loss không hợp lệ: {loss_type}")
