"""
Version 2 (v2 - Proposed SOTA): Hybrid CNN-BiLSTM with Multi-Head Self-Attention
Đột phá kiến trúc phục vụ bài báo Nghiên cứu Khoa học và Quỹ đầu tư:
1. 1D-CNN trích xuất đặc trưng không gian đa chỉ báo kỹ thuật.
2. Bi-LSTM học ngữ cảnh xu hướng 2 chiều quá khứ - tương lai cục bộ.
3. Multi-Head Self-Attention gán trọng số tập trung vào các phiên biến động mạnh / tin tức.
4. Residual Skip Connection chống suy giảm gradient.
"""

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim phải chia hết cho num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        B, S, D = x.shape
        
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, S, D_h)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, S, S)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V) # (B, H, S, D_h)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(context)
        return out, attn_weights


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, cnn_filters: int = 64, kernel_size: int = 3, 
                 lstm_hidden: int = 64, num_heads: int = 4, dropout: float = 0.2, output_dim: int = 1):
        super().__init__()
        
        # 1. Feature Extractor: 1D-CNN
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.conv_bn = nn.BatchNorm1d(cnn_filters)
        
        # 2. Sequence Modeler: Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 3. Attention Layer
        lstm_out_dim = lstm_hidden * 2 # Do dùng Bidirectional
        self.attention = MultiHeadSelfAttention(embed_dim=lstm_out_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        
        # 4. Regressor Head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_out_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # 1. CNN trích xuất đặc trưng không gian
        x_trans = x.transpose(1, 2) # (B, input_dim, S)
        conv_out = self.relu(self.conv_bn(self.conv(x_trans)))
        conv_out = conv_out.transpose(1, 2) # (B, S, cnn_filters)
        
        # 2. BiLSTM học ngữ cảnh thời gian 2 chiều
        lstm_out, _ = self.bilstm(conv_out) # (B, S, lstm_out_dim)
        
        # 3. Multi-Head Attention + Residual Connection
        attn_out, attn_weights = self.attention(lstm_out)
        normed = self.layer_norm(lstm_out + attn_out) # Residual add & norm
        
        # 4. Pooling & Projection
        # Lấy trọng số chú ý tổng hợp hoặc timestep cuối
        context_vector = torch.mean(normed, dim=1) # Global average pooling qua time
        h = self.relu(self.fc1(self.dropout(context_vector)))
        out = self.fc2(h)
        return out
