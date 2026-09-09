import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from models.baseline import NaivePersistenceModel, LinearRegressionBaseline
from models.lstm import VanillaLSTM, DeepLSTM
from models.cnn1d import VanillaCNN1D, TemporalCNN1D
from models.cnn_lstm_attention import CNNBiLSTMAttention
from models.seq2seq_multistep import Seq2SeqAttentionMultiStep
from training.loss import DirectionalPenaltyLoss, get_loss_function
from training.metric import evaluate_ml_metrics
from evaluation.stock_metric import calculate_financial_metrics


class TestStockForecastingPipeline(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.seq_len = 60
        self.input_dim_v0 = 5   # OHLCV
        self.input_dim_v1 = 25  # Features làm giàu
        
        self.dummy_x_v0 = torch.randn(self.batch_size, self.seq_len, self.input_dim_v0)
        self.dummy_x_v1 = torch.randn(self.batch_size, self.seq_len, self.input_dim_v1)
        self.dummy_y_1step = torch.randn(self.batch_size, 1)
        self.dummy_y_7step = torch.randn(self.batch_size, 7)
        self.dummy_y_prev = torch.randn(self.batch_size, 1)

    def test_baseline_models(self):
        naive = NaivePersistenceModel(target_feature_idx=3)
        preds = naive.predict(self.dummy_x_v0.numpy())
        self.assertEqual(preds.shape, (self.batch_size,))

        ridge = LinearRegressionBaseline()
        ridge.fit(self.dummy_x_v0.numpy(), self.dummy_y_1step.numpy())
        preds_ridge = ridge.predict(self.dummy_x_v0.numpy())
        self.assertEqual(preds_ridge.shape, (self.batch_size,))

    def test_v0_models(self):
        # Vanilla LSTM
        lstm_v0 = VanillaLSTM(input_dim=self.input_dim_v0, hidden_dim=32, output_dim=1)
        out_lstm = lstm_v0(self.dummy_x_v0)
        self.assertEqual(out_lstm.shape, (self.batch_size, 1))

        # Vanilla CNN1D
        cnn_v0 = VanillaCNN1D(input_dim=self.input_dim_v0, num_filters=32, output_dim=1)
        out_cnn = cnn_v0(self.dummy_x_v0)
        self.assertEqual(out_cnn.shape, (self.batch_size, 1))

    def test_v1_models(self):
        # Deep LSTM
        lstm_v1 = DeepLSTM(input_dim=self.input_dim_v1, hidden_dim=32, num_layers=2, output_dim=1)
        out_lstm = lstm_v1(self.dummy_x_v1)
        self.assertEqual(out_lstm.shape, (self.batch_size, 1))

        # Temporal CNN1D
        cnn_v1 = TemporalCNN1D(input_dim=self.input_dim_v1, num_filters=32, output_dim=1)
        out_cnn = cnn_v1(self.dummy_x_v1)
        self.assertEqual(out_cnn.shape, (self.batch_size, 1))

    def test_v2_sota_model(self):
        # CNN-BiLSTM-Attention
        sota_model = CNNBiLSTMAttention(input_dim=self.input_dim_v1, cnn_filters=32, lstm_hidden=32, num_heads=2, output_dim=1)
        out_sota = sota_model(self.dummy_x_v1)
        self.assertEqual(out_sota.shape, (self.batch_size, 1))

    def test_v3_multistep_model(self):
        # Seq2Seq Multi-step with Attention
        s2s_model = Seq2SeqAttentionMultiStep(input_dim=self.input_dim_v1, hidden_dim=32, forecast_horizon=7)
        out_s2s = s2s_model(self.dummy_x_v1)
        self.assertEqual(out_s2s.shape, (self.batch_size, 7))

    def test_loss_and_metrics(self):
        # Test DirectionalPenaltyLoss với y_prev
        loss_fn = get_loss_function("directional", penalty_weight=0.5)
        loss_val = loss_fn(self.dummy_y_1step, self.dummy_y_1step, self.dummy_y_prev)
        self.assertTrue(torch.is_tensor(loss_val))
        
        # Test Huber Loss (không cần y_prev)
        huber_fn = get_loss_function("huber")
        huber_val = huber_fn(self.dummy_y_1step, self.dummy_y_1step)
        self.assertTrue(torch.is_tensor(huber_val))

        y_true = np.array([100.0, 102.0, 101.0, 105.0])
        y_pred = np.array([100.5, 101.8, 101.2, 104.5])
        y_prev = np.array([99.0, 100.0, 102.0, 101.0])
        metrics = evaluate_ml_metrics(y_true, y_pred, y_prev)
        self.assertIn('RMSE', metrics)
        self.assertIn('DA%', metrics)

        fin_metrics = calculate_financial_metrics(y_true, y_pred)
        self.assertIn('Sharpe_Ratio', fin_metrics)
        self.assertIn('Total_Return%', fin_metrics)
        # Kiểm tra Sharpe Ratio không còn là giá trị cực đoan
        self.assertTrue(abs(fin_metrics['Sharpe_Ratio']) < 1000, 
                       f"Sharpe Ratio bất thường: {fin_metrics['Sharpe_Ratio']}")


if __name__ == '__main__':
    unittest.main()
