"""
Kiến trúc 1D-CNN (Temporal Convolutional Network) để so sánh với LSTM theo yêu cầu đề bài:
- VanillaCNN1D: Mô hình cơ bản v0 (1 tầng Conv1D).
- TemporalCNN1D: Mô hình đa tầng v1 với BatchNorm, Dropout và Residual Connection.
"""

import torch
import torch.nn as nn


class VanillaCNN1D(nn.Module):
    """
    Version 0 (v0): 1D-CNN cơ bản nhất.
    """
    def __init__(self, input_dim: int, num_filters: int = 64, kernel_size: int = 3, output_dim: int = 1):
        super().__init__()
        # Conv1D nhận input shape: (batch_size, in_channels, seq_len)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) -> chuyển sang (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        c = self.relu(self.conv(x))
        p = self.pool(c).squeeze(-1) # (batch_size, num_filters)
        out = self.fc(p)
        return out


class TemporalCNN1D(nn.Module):
    """
    Version 1 (v1): Deep Temporal CNN với nhiều block tích chập và dilated convolutions.
    """
    def __init__(self, input_dim: int, num_filters: int = 64, kernel_size: int = 3, dropout: float = 0.2, output_dim: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters * 2, num_filters)
        self.fc2 = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.dropout(self.relu(self.bn1(self.conv1(x))))
        h = self.dropout(self.relu(self.bn2(self.conv2(h))))
        pooled = self.pool(h).squeeze(-1)
        out = self.fc2(self.relu(self.fc1(pooled)))
        return out
