"""
Kiến trúc LSTM:
- VanillaLSTM: Mô hình sơ khởi (v0) với 1 lớp LSTM + Linear.
- DeepLSTM: Mô hình nâng cao (v1) với nhiều tầng LSTM, Dropout và LayerNorm.
"""

import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):
    """
    Version 0 (v0): Vanilla LSTM cơ bản nhất để kiểm chứng Deep Learning so với Base Model.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Lấy hidden state của bước thời gian cuối cùng
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        out = self.fc(last_hidden)        # (batch_size, output_dim)
        return out


class DeepLSTM(nn.Module):
    """
    Version 1 (v1): Deep Multi-layer LSTM với Dropout và LayerNorm.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        normed = self.layer_norm(last_hidden)
        dropped = self.dropout(normed)
        h = self.relu(self.fc1(dropped))
        out = self.fc2(h)
        return out
