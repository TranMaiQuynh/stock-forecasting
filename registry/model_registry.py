"""
Model Registry: Khởi tạo linh hoạt các mô hình dựa trên cấu hình (config/stock.yaml).
"""

import torch.nn as nn
from models.baseline import NaivePersistenceModel, LinearRegressionBaseline
from models.lstm import VanillaLSTM, DeepLSTM
from models.cnn1d import VanillaCNN1D, TemporalCNN1D
from models.cnn_lstm_attention import CNNBiLSTMAttention
from models.seq2seq_multistep import Seq2SeqAttentionMultiStep


def build_model_by_name(model_name: str, input_dim: int, config: dict, forecast_horizon: int = 1, target_feature_idx: int = -1):
    model_name = model_name.lower()
    cfg_model = config['model']
    
    # 1. Base Models
    if model_name == "naive_baseline":
        return NaivePersistenceModel(target_feature_idx=target_feature_idx)
    elif model_name == "linear_baseline":
        return LinearRegressionBaseline()
        
    # 2. Version 0 (v0 - Vanilla Models)
    elif model_name == "vanilla_lstm" or model_name == "lstm_v0":
        return VanillaLSTM(
            input_dim=input_dim,
            hidden_dim=cfg_model['lstm']['hidden_dim'],
            output_dim=forecast_horizon
        )
    elif model_name == "vanilla_cnn1d" or model_name == "cnn1d_v0":
        return VanillaCNN1D(
            input_dim=input_dim,
            num_filters=cfg_model['cnn1d']['num_filters'],
            kernel_size=cfg_model['cnn1d']['kernel_size'],
            output_dim=forecast_horizon
        )
        
    # 3. Version 1 (v1 - Feature-Enhanced Deep Models)
    elif model_name == "deep_lstm" or model_name == "lstm_v1":
        return DeepLSTM(
            input_dim=input_dim,
            hidden_dim=cfg_model['lstm']['hidden_dim'],
            num_layers=cfg_model['lstm']['num_layers'],
            dropout=cfg_model['lstm']['dropout'],
            output_dim=forecast_horizon
        )
    elif model_name == "temporal_cnn1d" or model_name == "cnn1d_v1":
        return TemporalCNN1D(
            input_dim=input_dim,
            num_filters=cfg_model['cnn1d']['num_filters'],
            kernel_size=cfg_model['cnn1d']['kernel_size'],
            dropout=cfg_model['cnn1d']['dropout'],
            output_dim=forecast_horizon
        )
        
    # 4. Version 2 (v2 - Proposed SOTA: CNN-BiLSTM-Attention)
    elif model_name == "cnn_bilstm_attention" or model_name == "v2_sota":
        cfg_sota = cfg_model['cnn_bilstm_attention']
        return CNNBiLSTMAttention(
            input_dim=input_dim,
            cnn_filters=cfg_sota['cnn_filters'],
            kernel_size=cfg_sota['kernel_size'],
            lstm_hidden=cfg_sota['lstm_hidden'],
            num_heads=cfg_sota['num_heads'],
            dropout=cfg_sota['dropout'],
            output_dim=forecast_horizon
        )
        
    # 5. Version 3 (v3 - Multi-step Seq2Seq)
    elif model_name == "seq2seq_multistep" or model_name == "v3_multistep":
        cfg_s2s = cfg_model['seq2seq']
        return Seq2SeqAttentionMultiStep(
            input_dim=input_dim,
            hidden_dim=cfg_s2s['hidden_dim'],
            forecast_horizon=forecast_horizon,
            dropout=cfg_s2s['dropout']
        )
    else:
        raise ValueError(f"Không tìm thấy mô hình: {model_name} trong Model Registry!")
