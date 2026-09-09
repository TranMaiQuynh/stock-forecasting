"""
Bộ khởi tạo Optimizer và Learning Rate Scheduler.
"""

import torch
import torch.optim as optim


def build_optimizer(model: torch.nn.Module, lr: float = 0.001, weight_decay: float = 0.0001, opt_type: str = "adamw"):
    opt_type = opt_type.lower()
    if opt_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer không hợp lệ: {opt_type}")


def build_scheduler(optimizer, scheduler_type: str = "plateau", patience: int = 5, factor: float = 0.5):
    if scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    else:
        return None
