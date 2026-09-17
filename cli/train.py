"""
CLI Huấn luyện Mô hình — Multi-Ticker & Multi-Seed:
Hỗ trợ huấn luyện từng phiên bản đơn lẻ hoặc chạy toàn bộ Ablation Study
(Base → v0 → v1 → v2 → v3) trên nhiều mã cổ phiếu và nhiều random seeds.

Chiến lược thực nghiệm:
- Baseline & v0: Chạy với seed mặc định (42), chỉ AAPL → làm mốc đối chứng
- v1, v2, v3: Chạy với cả 2 seeds (42 & 100) × 3 tickers (AAPL, TSLA, MSFT)
  → Chứng minh tính tổng quát hoá và ổn định của kiến trúc

Kết quả:
- Mỗi epoch → 1 ảnh PNG trong logs/training_runs/{TICKER}/{model}/seed_{N}/
- Checkpoint tốt nhất → checkpoints/{TICKER}/{model}_seed{N}_best.pt
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
import yaml
import torch
from preprocessing.run_pipeline import load_config, prepare_dataset_pipeline
from datasets.stock_dataset import build_dataloaders
from registry.model_registry import build_model_by_name
from training.stock_trainer import StockTrainer, generate_training_summary_report, set_seed


def train_single_model(
    model_name: str,
    version: str,
    config: dict,
    ticker: str = "AAPL",
    seed: int = 42,
):
    """
    Huấn luyện một mô hình cho một mã cổ phiếu và một random seed cụ thể.
    Tất cả logs epoch ảnh được tổ chức vào:
      logs/training_runs/{TICKER}/{model_name}/seed_{N}/epoch_XXX.png
    """
    print(f"\n{'='*70}")
    print(f"    TRAIN: {model_name.upper()} | {ticker} | Seed={seed} | Version={version}")
    print(f"{'='*70}")

    # Đặt seed ngay tại đây — TRƯỜC mọi lời gọi có dùng RNG:
    # build_dataloaders (jittering dùng np.random) và build_model_by_name (weight init dùng torch)
    # 1 điểm gọi duy nhất — không re-seed trong Trainer.__init__ hoặc fit()
    set_seed(seed)

    # ── 1. Chuẩn bị dữ liệu theo ticker và phiên bản ─────────────────────────
    # Ghi đè ticker trong config để tải đúng cổ phiếu
    cfg = dict(config)
    cfg['data'] = dict(config['data'])
    cfg['data']['ticker'] = ticker

    data_bundle = prepare_dataset_pipeline(cfg, version=version)
    input_dim = len(data_bundle['feature_cols'])

    cfg_win = config['window']
    cfg_train = config['training']
    forecast_horizon = (
        cfg_win['multi_step_horizon']
        if ("multistep" in model_name or version == "v3")
        else cfg_win['forecast_horizon']
    )

    # ── 2. Xây dựng DataLoaders ───────────────────────────────────────────────
    cfg_data = cfg['data']
    train_loader, val_loader, test_loader = build_dataloaders(
        data_bundle=data_bundle,
        input_window=cfg_win['input_window'],
        forecast_horizon=forecast_horizon,
        batch_size=cfg_train['batch_size'],
        use_augmentation=cfg_data.get('use_augmentation', False),  # Jittering chỉ trên Train
        noise_std=cfg_data.get('noise_std', 0.002),
        n_augmented=cfg_data.get('n_augmented', 2),
    )

    # ── 3. Baseline: Fit trực tiếp, không cần Trainer ─────────────────────────────────────────
    if "baseline" in model_name:
        # Tính target_feature_idx tại caller (có sẵn feature_cols) thay vì dùng default -1
        # Đúng như cách cli/evaluate.py:107 đã làm
        feature_cols = data_bundle['feature_cols']
        target_col   = data_bundle['target_col']
        target_feature_idx = (
            feature_cols.index(target_col) if target_col in feature_cols else -1
        )
        model = build_model_by_name(
            model_name, input_dim=input_dim, config=config,
            target_feature_idx=target_feature_idx
        )
        if model_name == "linear_baseline":
            X_train = train_loader.dataset.X.numpy()
            y_train = train_loader.dataset.y.numpy()
            model.fit(X_train, y_train)
        print(f"  ✅ [Baseline] {model_name} ({ticker}) đã sẵn sàng đánh giá.")
        return model, data_bundle, test_loader

    # ── 4. Deep Learning: Khởi tạo model và Trainer ──────────────────────────
    model = build_model_by_name(
        model_name, input_dim=input_dim, config=config, forecast_horizon=forecast_horizon
    )
    trainer = StockTrainer(
        model=model,
        config=config,
        model_name=f"{model_name}_{version}",
        ticker=ticker,
        seed=seed,
        target_scaler=data_bundle['target_scaler'],  # Cần để tính zero_point cho DirectionalPenaltyLoss
    )
    trainer.fit(train_loader, val_loader)

    return trainer, data_bundle, test_loader


def run_training_pipeline(
    config_path: str = "config/stock.yaml",
    target_model: str = "all",
    version: str = "v0",
):
    """
    Chạy toàn bộ pipeline huấn luyện theo chiến lược thực nghiệm chuẩn:

    - Baseline (naive + linear): 1 lần, AAPL, seed=42
    - v0 (Vanilla LSTM, Vanilla CNN): 1 lần, AAPL, seed=42
    - v1 (Deep LSTM, Temporal CNN): 2 seeds × 3 tickers
    - v2 (CNN-BiLSTM-Attention):    2 seeds × 3 tickers
    - v3 (Seq2Seq Attention):        2 seeds × 3 tickers
    """
    config = load_config(config_path)

    # Đọc danh sách tickers và seeds từ config (hoặc dùng mặc định)
    all_tickers = config.get('data', {}).get('training_tickers', ["AAPL"])
    all_seeds   = config.get('data', {}).get('training_seeds', [42])

    if target_model == "all":
        print("\n" + "="*70)
        print("   📋 ABLATION STUDY — Toàn bộ pipeline (Base → v0 → v1 → v2 → v3)")
        print(f"   🏷️  Tickers  : {all_tickers}")
        print(f"   🎲  Seeds    : {all_seeds}")
        print("="*70)

        # ─── PHASE 1: Baseline & v0 ─ AAPL only, seed=42 ─────────────────────
        # (Baseline không cần test đa dạng, chỉ làm mốc đối chiếu cho AAPL)
        print("\n\n📌 PHASE 1: Baseline + v0  (AAPL · Seed 42 — Mốc đối chứng)\n")
        for m_name, v_name in [
            ("naive_baseline",  "baseline"),
            ("linear_baseline", "baseline"),
            ("vanilla_lstm",    "v0"),
            ("vanilla_cnn1d",   "v0"),
        ]:
            train_single_model(m_name, v_name, config, ticker="AAPL", seed=42)

        # ─── PHASE 2: v1, v2, v3 ─ Multi-ticker × Multi-seed ─────────────────
        print("\n\n📌 PHASE 2: v1 / v2 / v3  (Multi-Ticker × Multi-Seed — Chứng minh tính tổng quát)\n")
        advanced_models = [
            ("deep_lstm",              "v1"),
            ("temporal_cnn1d",         "v1"),
            ("cnn_bilstm_attention",   "v2"),
            ("seq2seq_multistep",      "v3"),
        ]
        for ticker in all_tickers:
            for seed in all_seeds:
                for m_name, v_name in advanced_models:
                    train_single_model(m_name, v_name, config, ticker=ticker, seed=seed)

    else:
        # ─── Train một model cụ thể theo CLI args ─────────────────────────────
        for ticker in all_tickers:
            for seed in all_seeds:
                train_single_model(target_model, version, config, ticker=ticker, seed=seed)

    print("\n\n" + "="*70)
    print("   ✅ TOÀN BỘ QUÁN TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print("="*70)

    # Tổng hợp báo cáo so sánh tất cả runs
    generate_training_summary_report(log_base_dir="logs", output_dir="output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình dự báo giá cổ phiếu — Multi-Ticker & Multi-Seed"
    )
    parser.add_argument("--config",  type=str, default="config/stock.yaml")
    parser.add_argument(
        "--model", type=str, default="all",
        help="all | vanilla_lstm | vanilla_cnn1d | deep_lstm | temporal_cnn1d | "
             "cnn_bilstm_attention | seq2seq_multistep"
    )
    parser.add_argument(
        "--version", type=str, default="v0",
        choices=["baseline", "v0", "v1", "v2", "v3"]
    )
    args = parser.parse_args()

    run_training_pipeline(
        config_path=args.config,
        target_model=args.model,
        version=args.version,
    )
