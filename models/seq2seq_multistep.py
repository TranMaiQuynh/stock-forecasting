"""
Version 3 (v3 - Multi-step Forecasting): Seq2Seq Encoder-Decoder with Bahdanau Attention
Thực hiện yêu cầu mở rộng của đề tài: Dự báo giá cổ phiếu đa bước (ví dụ: 7 ngày tiếp theo).

Kiến trúc:
1. Encoder: Multi-layer LSTM mã hóa chuỗi đầu vào thành context vectors.
2. Bahdanau Attention: Tính trọng số chú ý (attention weights) giữa decoder hidden state
   và toàn bộ encoder outputs, giúp decoder "nhìn lại" các phiên quan trọng.
3. Decoder: LSTM giải mã từng bước (autoregressive), kết hợp context vector từ attention
   để dự báo tuần tự t+1, t+2, ..., t+k.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder LSTM: Mã hóa chuỗi đầu vào thành hidden states."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (batch_size, seq_len, hidden_dim) — toàn bộ hidden states
        # hidden: (num_layers, batch_size, hidden_dim) — hidden state cuối
        return outputs, (hidden, cell)


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention):
    score(s_t, h_i) = V^T * tanh(W_s * s_t + W_h * h_i)
    Cho phép decoder "nhìn lại" tất cả encoder outputs và gán trọng số cho từng timestep.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (batch_size, hidden_dim) — hidden state hiện tại của decoder
        encoder_outputs: (batch_size, src_len, hidden_dim) — toàn bộ outputs của encoder
        """
        # Mở rộng decoder_hidden để so sánh với mọi encoder timestep
        # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim) -> broadcast
        decoder_expanded = decoder_hidden.unsqueeze(1)  # (B, 1, H)

        # Tính điểm attention: score = V * tanh(W_s * s + W_h * h)
        energy = torch.tanh(self.W_s(decoder_expanded) + self.W_h(encoder_outputs))  # (B, src_len, H)
        scores = self.V(energy).squeeze(-1)  # (B, src_len)

        # Chuẩn hóa thành xác suất attention
        attn_weights = F.softmax(scores, dim=1)  # (B, src_len)

        # Context vector = trung bình có trọng số của encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B, 1, H)
        context = context.squeeze(1)  # (B, H)

        return context, attn_weights


class Decoder(nn.Module):
    """Decoder LSTM với Bahdanau Attention: Giải mã từng bước dự báo."""
    def __init__(self, hidden_dim: int = 64, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = BahdanauAttention(hidden_dim)

        # Decoder LSTM: nhận [context_vector; previous_prediction] = hidden_dim + 1
        self.lstm_cell = nn.LSTMCell(
            input_size=hidden_dim + output_dim,
            hidden_size=hidden_dim
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)  # concat context + hidden

    def forward(self, encoder_outputs, hidden, cell, forecast_horizon, target=None, teacher_forcing_ratio=0.0):
        """
        encoder_outputs: (B, src_len, H)
        hidden: (B, H) — hidden state khởi tạo (từ encoder layer cuối)
        cell: (B, H) — cell state khởi tạo
        forecast_horizon: số bước dự báo
        target: (B, forecast_horizon) — nhãn thực tế (nếu dùng teacher forcing)
        teacher_forcing_ratio: xác suất dùng nhãn thực thay vì dự báo
        """
        batch_size = encoder_outputs.size(0)
        outputs = []

        # Khởi tạo input đầu tiên cho decoder (zero vector)
        decoder_input = torch.zeros(batch_size, 1, device=encoder_outputs.device)

        for t in range(forecast_horizon):
            # 1. Tính attention context từ encoder outputs
            context, attn_weights = self.attention(hidden, encoder_outputs)

            # 2. Ghép context vector với input hiện tại
            lstm_input = torch.cat([context, decoder_input], dim=1)  # (B, H+1)

            # 3. Bước LSTM
            hidden, cell = self.lstm_cell(lstm_input, (hidden, cell))

            # 4. Dự báo: concat hidden + context → FC
            combined = torch.cat([hidden, context], dim=1)  # (B, 2H)
            prediction = self.fc_out(self.dropout(combined))  # (B, 1)

            outputs.append(prediction)

            # 5. Teacher forcing: dùng nhãn thực hoặc dự báo làm input bước tiếp
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1]  # (B, 1)
            else:
                decoder_input = prediction.detach()  # (B, 1)

        # Stack tất cả dự báo: (B, forecast_horizon)
        outputs = torch.cat(outputs, dim=1)
        return outputs


class Seq2SeqAttentionMultiStep(nn.Module):
    """
    Seq2Seq Encoder-Decoder hoàn chỉnh với Bahdanau Attention.
    Encoder mã hóa 60 ngày quá khứ → Decoder giải mã tuần tự 7 ngày tương lai.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, forecast_horizon: int = 7, dropout: float = 0.2):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.encoder = Encoder(input_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.decoder = Decoder(hidden_dim=hidden_dim, output_dim=1, dropout=dropout)

    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        """
        x: (batch_size, seq_len, input_dim)
        target: (batch_size, forecast_horizon) — optional, for teacher forcing during training
        """
        # 1. Encode
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # 2. Lấy hidden/cell state từ tầng LSTM cuối cùng cho decoder
        decoder_hidden = hidden[-1]  # (batch_size, hidden_dim)
        decoder_cell = cell[-1]      # (batch_size, hidden_dim)

        # 3. Decode với attention
        predictions = self.decoder(
            encoder_outputs=encoder_outputs,
            hidden=decoder_hidden,
            cell=decoder_cell,
            forecast_horizon=self.forecast_horizon,
            target=target,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        return predictions  # (batch_size, forecast_horizon)
