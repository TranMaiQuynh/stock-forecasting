"""
MAIN ENTRYPOINT: HỆ THỐNG DỰ BÁO CHỨNG KHOÁN (STOCK FORECASTING PIPELINE)
Tuân thủ chuẩn Yahoo Finance (yfinance) & Phương pháp luận NCKH (Base -> v0 -> v1 -> v2 -> v3).
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse
from preprocessing.run_pipeline import load_config, prepare_dataset_pipeline
from cli.train import run_training_pipeline
from cli.evaluate import evaluate_all_models
from inference.stock_pipeline import run_live_inference


def main():
    parser = argparse.ArgumentParser(description="Hệ thống Dự báo Giá Cổ phiếu từ Yahoo Finance")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all", 
        choices=["data", "train", "eval", "inference", "all"],
        help="Chế độ chạy: data (chuẩn bị dữ liệu), train (huấn luyện), eval (đánh giá), inference (dự báo realtime), all (chạy toàn bộ)"
    )
    parser.add_argument("--config", type=str, default="config/stock.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--model", type=str, default="all", help="Mô hình cần huấn luyện: all, vanilla_lstm, vanilla_cnn1d, cnn_bilstm_attention, seq2seq_multistep")
    parser.add_argument("--version", type=str, default="v0", choices=["baseline", "v0", "v1", "v2", "v3"], help="Phiên bản phát triển")
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    print("\n" + "="*80)
    print("      CHƯƠNG TRÌNH DỰ BÁO GIÁ CỔ PHIẾU - NGHIÊN CỨU KHOA HỌC & ĐỊNH LƯỢNG")
    print("="*80)
    print(f"Mã cổ phiếu:      {cfg['data']['ticker']}")
    print(f"Khoảng thời gian: {cfg['data']['start_date']} -> {cfg['data']['end_date']}")
    print(f"Chế độ thực thi:  {args.mode.upper()}")
    print("="*80 + "\n")
    
    if args.mode == "data":
        print("[1/1] Thực thi chuẩn bị dữ liệu Yahoo Finance...")
        prepare_dataset_pipeline(cfg, version=args.version)
        
    elif args.mode == "train":
        print("[1/1] Thực thi huấn luyện mô hình...")
        run_training_pipeline(config_path=args.config, target_model=args.model, version=args.version)
        
    elif args.mode == "eval":
        print("[1/1] Thực thi đánh giá và so sánh (Ablation Benchmark)...")
        evaluate_all_models(config_path=args.config)
        
    elif args.mode == "inference":
        print("[1/1] Thực thi dự báo thời gian thực...")
        run_live_inference(config_path=args.config, model_name="cnn_bilstm_attention", version="v2")
        
    elif args.mode == "all":
        print(">>> BƯỚC 1: Tải và chuẩn bị dữ liệu từ Yahoo Finance...")
        prepare_dataset_pipeline(cfg, version="v0")
        prepare_dataset_pipeline(cfg, version="v1")
        
        print("\n>>> BƯỚC 2: Huấn luyện toàn bộ chuỗi tiến hóa mô hình (Base -> v0 -> v1 -> v2 -> v3)...")
        run_training_pipeline(config_path=args.config, target_model="all")
        
        print("\n>>> BƯỚC 3: Đánh giá kép (ML + Quant Metrics Backtest)...")
        evaluate_all_models(config_path=args.config)
        
        print("\n>>> BƯỚC 4: Dự báo thời gian thực với mô hình tốt nhất (v2 SOTA)...")
        run_live_inference(config_path=args.config, model_name="cnn_bilstm_attention", version="v2")
        
    print("\n[Thành công] Chương trình đã hoàn tất toàn bộ tiến trình!")


if __name__ == "__main__":
    main()
