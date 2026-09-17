"""
Module Trainer chuẩn PyTorch:
- Hỗ trợ huấn luyện với EarlyStopping
- Tự động lưu checkpoint mô hình tốt nhất (.pt) theo cấu trúc checkpoints/{TICKER}/
- Lưu ảnh trực quan hoá sau TỪNG EPOCH vào logs/training_runs/{TICKER}/{model}/seed_{N}/
- Gradient Clipping chống bùng nổ đạo hàm (Exploding Gradients)
- Hỗ trợ DirectionalPenaltyLoss với y_prev từ DataLoader
- Hỗ trợ multi-ticker (AAPL, TSLA, MSFT) và multi-seed (42, 100)
"""

import os
import sys
import time
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import numpy as np
from training.loss import get_loss_function, DirectionalPenaltyLoss
from training.optimizer import build_optimizer, build_scheduler

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


def set_seed(seed: int = 42):
    """Đặt random seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StockTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        model_name: str = "model",
        ticker: str = "AAPL",
        seed: int = 42,
        log_base_dir: str = "logs",
        target_scaler=None,  # Cần để tính đúng zero_point cho DirectionalPenaltyLoss
    ):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.ticker = ticker.upper()
        self.seed = seed

        # ── Thiết lập thư mục log theo cấu trúc rõ ràng ──────────────────────
        # logs/training_runs/{TICKER}/{model_name}/seed_{N}/
        self.run_dir = os.path.join(
            log_base_dir, "training_runs", self.ticker, model_name, f"seed_{seed}"
        )
        os.makedirs(self.run_dir, exist_ok=True)

        cfg_train = config['training']
        self.epochs = cfg_train['epochs']
        self.lr = cfg_train['learning_rate']
        self.weight_decay = cfg_train['weight_decay']
        self.patience = cfg_train['early_stopping_patience']

        # ── Checkpoint theo ticker ────────────────────────────────────────────
        # checkpoints/{TICKER}/{model_name}_seed{N}_best.pt
        base_ckpt = cfg_train['checkpoint_dir']
        self.checkpoint_dir = os.path.join(base_ckpt, self.ticker)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ── Thiết bị ──────────────────────────────────────────────────────────
        if cfg_train['device'] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg_train['device'])
        self.model.to(self.device)

        # ── Hàm Loss ──────────────────────────────────────────────────────────
        self.loss_type = cfg_train.get('loss_type', 'huber')
        use_log_return = config.get('data', {}).get('use_log_return', False)

        # Tính zero_point tất định từ target_scaler:
        # zero_point = (0 - mean_) / scale_  — giá trị "raw Log_Return = 0" sau khi StandardScaler.
        # Cần thiết vì y_pred/y_true trong forward() đã ở không gian scaled:
        # sign(y_true_scaled) ≠ sign(y_true_raw) khi scaler.mean_ ≠ 0.
        zero_point = 0.0
        if use_log_return and target_scaler is not None:
            try:
                zero_point = float(-target_scaler.mean_[0] / target_scaler.scale_[0])
            except (AttributeError, IndexError, ZeroDivisionError):
                zero_point = 0.0  # Fallback an toàn nếu scaler chưa fit

        self.criterion = get_loss_function(
            loss_type=self.loss_type,
            penalty_weight=cfg_train.get('directional_penalty_weight', 0.5),
            use_log_return=use_log_return,
            zero_point=zero_point,
        )
        self.uses_directional_loss = isinstance(self.criterion, DirectionalPenaltyLoss)

        self.optimizer = build_optimizer(self.model, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = build_scheduler(self.optimizer)

        # ── Teacher Forcing (chỉ cho Seq2SeqAttentionMultiStep) ────────────────────────────
        # Phát hiện một lần tại __init__, không kiểm tra lại mỗi batch
        import inspect
        self._model_supports_tf = (
            'teacher_forcing_ratio' in inspect.signature(self.model.forward).parameters
        )
        self._tf_ratio = float(cfg_train.get('teacher_forcing_ratio', 0.0))
        self._tf_decay = float(cfg_train.get('teacher_forcing_decay', 1.0))

        self.history = {'train_loss': [], 'val_loss': [], 'epoch_logs': []}

    # ──────────────────────────────────────────────────────────────────────────
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for batch_data in train_loader:
            X_batch, y_batch, y_prev_batch = batch_data
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_prev_batch = y_prev_batch.to(self.device)

            self.optimizer.zero_grad()

            # Teacher forcing: chỉ truyền target vào MODEL trong train_epoch
            # validate() và predict() KHÔNG bao giờ nhận target — đảm bảo bằng cấu trúc code
            if self._model_supports_tf and self._tf_ratio > 0.0:
                y_pred = self.model(
                    X_batch,
                    target=y_batch,
                    teacher_forcing_ratio=self._tf_ratio
                )
            else:
                y_pred = self.model(X_batch)

            if self.uses_directional_loss:
                loss = self.criterion(y_pred, y_batch, y_prev_batch)
            else:
                loss = self.criterion(y_pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    # ──────────────────────────────────────────────────────────────────────────
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                X_batch, y_batch, y_prev_batch = batch_data
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_prev_batch = y_prev_batch.to(self.device)

                y_pred = self.model(X_batch)

                if self.uses_directional_loss:
                    loss = self.criterion(y_pred, y_batch, y_prev_batch)
                else:
                    loss = self.criterion(y_pred, y_batch)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    # ──────────────────────────────────────────────────────────────────────────
    def _save_epoch_visualization(self, current_epoch: int, patience_counter: int, current_lr: float, is_best: bool):
        """
        Lưu ảnh trực quan hoá phong phú sau TỪNG EPOCH.
        File: logs/training_runs/{TICKER}/{model_name}/seed_{N}/epoch_{N:03d}.png

        Layout 2 cột:
        - Trái: Đường cong Loss (Train vs Val) tích luỹ từ epoch 1 đến epoch hiện tại
        - Phải: Bảng trạng thái chi tiết epoch hiện tại (loss, LR, patience, ...)
        """
        n = len(self.history['train_loss'])
        epochs_range = list(range(1, n + 1))
        train_losses = self.history['train_loss']
        val_losses = self.history['val_loss']
        best_val_idx = int(np.argmin(val_losses))
        best_val = val_losses[best_val_idx]
        current_train = train_losses[-1]
        current_val = val_losses[-1]

        fig = plt.figure(figsize=(14, 5))
        fig.patch.set_facecolor('#0f1117')
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.35)

        # ── Panel trái: Loss Curve ─────────────────────────────────────────
        ax_loss = fig.add_subplot(gs[0])
        ax_loss.set_facecolor('#1a1d27')

        ax_loss.plot(epochs_range, train_losses,
                     color='#4a9eff', linewidth=2.0, label='Train Loss', marker='o',
                     markersize=3 if n <= 40 else 0)
        ax_loss.plot(epochs_range, val_losses,
                     color='#ff6b6b', linewidth=2.0, label='Val Loss', marker='s',
                     markersize=3 if n <= 40 else 0)

        # Đánh dấu epoch tốt nhất
        ax_loss.scatter([best_val_idx + 1], [best_val],
                        color='#ffd700', s=120, zorder=6,
                        label=f'Best Val (E{best_val_idx+1}: {best_val:.5f})',
                        marker='*')

        # Gạch đứng tại epoch hiện tại
        ax_loss.axvline(x=current_epoch, color='#888888', linestyle='--', linewidth=1.0, alpha=0.6)

        ax_loss.set_title(
            f'[LOSS] {self.ticker} | {self.model_name} | Seed {self.seed}',
            fontsize=11, fontweight='bold', color='white', pad=10
        )
        ax_loss.set_xlabel('Epoch', color='#aaaaaa', fontsize=10)
        ax_loss.set_ylabel('Loss', color='#aaaaaa', fontsize=10)
        ax_loss.tick_params(colors='#aaaaaa')
        ax_loss.spines['bottom'].set_color('#333344')
        ax_loss.spines['left'].set_color('#333344')
        ax_loss.spines['top'].set_visible(False)
        ax_loss.spines['right'].set_visible(False)
        ax_loss.legend(fontsize=9, facecolor='#1a1d27', labelcolor='white', framealpha=0.8)
        ax_loss.grid(True, linestyle=':', alpha=0.3, color='#555555')

        # ── Panel phải: Bảng trạng thái ──────────────────────────────────
        ax_info = fig.add_subplot(gs[1])
        ax_info.set_facecolor('#1a1d27')
        ax_info.axis('off')

        # Tính improvement so với epoch trước
        if n >= 2:
            delta_val = val_losses[-2] - current_val
            delta_str = f"+{delta_val:.6f} ↓" if delta_val > 0 else f"{delta_val:.6f} ↑"
            delta_color = '#4ade80' if delta_val > 0 else '#f87171'
        else:
            delta_str = "—"
            delta_color = '#aaaaaa'

        best_mark = " [BEST]" if is_best else ""
        status_color = '#4ade80' if patience_counter == 0 else ('#fbbf24' if patience_counter < self.patience // 2 else '#f87171')

        rows = [
            ("EPOCH",         f"{current_epoch:03d} / {self.epochs}{best_mark}",  '#e2e8f0'),
            ("",              "",                                                   '#333344'),
            ("Train Loss",    f"{current_train:.6f}",                              '#93c5fd'),
            ("Val Loss",      f"{current_val:.6f}",                                '#fca5a5'),
            ("Best Val Loss", f"{best_val:.6f}  (E{best_val_idx+1})",             '#fbbf24'),
            ("Δ Val vs prev", delta_str,                                           delta_color),
            ("",              "",                                                   '#333344'),
            ("Learning Rate", f"{current_lr:.7f}",                                 '#c4b5fd'),
            ("Patience",      f"{patience_counter} / {self.patience}",             status_color),
            ("",              "",                                                   '#333344'),
            ("Ticker",        self.ticker,                                          '#67e8f9'),
            ("Seed",          str(self.seed),                                       '#86efac'),
            ("Loss Type",     self.loss_type.upper(),                              '#fdba74'),
            ("Device",        str(self.device).upper(),                             '#e2e8f0'),
        ]

        y_pos = 0.97
        line_h = 0.065
        for label, value, color in rows:
            if label == "" and value == "":
                ax_info.axhline(y=y_pos - line_h * 0.4,
                                xmin=0.02, xmax=0.98,
                                color='#333344', linewidth=0.8)
                y_pos -= line_h * 0.6
                continue
            ax_info.text(0.04, y_pos, label,
                         transform=ax_info.transAxes,
                         fontsize=9, color='#888888',
                         verticalalignment='top', fontfamily='monospace')
            ax_info.text(0.48, y_pos, value,
                         transform=ax_info.transAxes,
                         fontsize=9, color=color, fontweight='bold',
                         verticalalignment='top', fontfamily='monospace')
            y_pos -= line_h

        ax_info.set_title('[STATUS] Training Details', fontsize=11, fontweight='bold',
                          color='white', pad=10)

        # ── Lưu file ──────────────────────────────────────────────────────
        plot_path = os.path.join(self.run_dir, f"epoch_{current_epoch:03d}.png")
        plt.savefig(plot_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, train_loader, val_loader, verbose: bool = True):
        # Seed đã được set tại train_single_model() trước mọi lời gọi RNG.
        # Không re-seed ở đây để tránh reset trạng thái RNG giữa chừng.

        # Tên file checkpoint: checkpoints/{TICKER}/{model_name}_seed{N}_best.pt
        best_model_path = os.path.join(
            self.checkpoint_dir, f"{self.model_name}_seed{self.seed}_best.pt"
        )
        history_csv_path = os.path.join(self.run_dir, "training_log.csv")

        print(f"\n{'='*80}")
        print(f" [Trainer] {self.ticker} | {self.model_name.upper()} | Seed {self.seed} | Loss: {self.loss_type} | Device: {self.device}")
        print(f"{'='*80}")
        print(f" 📁 Epoch logs  → {self.run_dir}/")
        print(f" 💾 Checkpoint  → {best_model_path}")
        print(f"{'='*80}")
        start_time = time.time()

        best_val_loss = float('inf')
        patience_counter = 0
        current_tf_ratio = self._tf_ratio  # Teacher forcing ratio cho epoch hiện tại

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            epoch_time = time.time() - epoch_start

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if self.scheduler:
                self.scheduler.step(val_loss)

            # Decay teacher forcing ratio sau mỗi epoch (hướng tới autoregressive thuần túy)
            if self._model_supports_tf and self._tf_ratio > 0.0:
                current_tf_ratio *= self._tf_decay
                self._tf_ratio = current_tf_ratio

            is_best = False
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss if best_val_loss != float('inf') else 0.0
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                improved_str = f" ⭐ [Best -Δ:{improvement:.5f}]"
                is_best = True
            else:
                patience_counter += 1
                improved_str = f" (Patience: {patience_counter}/{self.patience})"

            epoch_info = {
                'epoch': epoch,
                'train_loss': round(train_loss, 6),
                'val_loss': round(val_loss, 6),
                'best_val_loss': round(best_val_loss, 6),
                'lr': current_lr,
                'time_sec': round(epoch_time, 2),
                'is_best': is_best,
                'patience': patience_counter,
                'ticker': self.ticker,
                'seed': self.seed
            }
            self.history['epoch_logs'].append(epoch_info)

            # ── Lưu CSV mọi epoch ──────────────────────────────────────────
            pd.DataFrame(self.history['epoch_logs']).to_csv(history_csv_path, index=False)

            # ── Lưu ảnh trực quan hoá mọi epoch ───────────────────────────
            self._save_epoch_visualization(epoch, patience_counter, current_lr, is_best)

            if verbose and (epoch % 5 == 0 or epoch == 1 or is_best or patience_counter >= self.patience):
                print(f"  Epoch [{epoch:03d}/{self.epochs}] | "
                      f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                      f"LR: {current_lr:.6f} | {epoch_time:.1f}s{improved_str}", flush=True)

            if patience_counter >= self.patience:
                print(f"\n  [EarlyStopping] Dừng tại epoch {epoch} — Val Loss không cải thiện sau {self.patience} epochs.", flush=True)
                break

        elapsed = time.time() - start_time
        print(f"\n  ✅ Hoàn tất {self.model_name} ({self.ticker}, seed={self.seed}) | {elapsed:.1f}s | Best Val Loss: {best_val_loss:.6f}", flush=True)
        print(f"  📁 Đã lưu {len(self.history['epoch_logs'])} ảnh epoch vào: {self.run_dir}/", flush=True)

        # Lưu JSON đầy đủ
        with open(os.path.join(self.run_dir, "training_log.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # Phục hồi best weights
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))

        return self.history

    # ──────────────────────────────────────────────────────────────────────────
    def predict(self, test_loader):
        self.model.eval()
        preds_list, targets_list = [], []

        with torch.no_grad():
            for batch_data in test_loader:
                X_batch, y_batch, _ = batch_data
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                preds_list.append(y_pred.cpu().numpy())
                targets_list.append(y_batch.numpy())

        return np.vstack(preds_list), np.vstack(targets_list)


# ──────────────────────────────────────────────────────────────────────────────
def generate_training_summary_report(log_base_dir: str = "logs", output_dir: str = "output"):
    """
    Tổng hợp toàn bộ file training_log.csv từ cấu trúc:
      logs/training_runs/{TICKER}/{model}/seed_{N}/training_log.csv
    Tạo:
    - Bảng so sánh tất cả runs theo ticker và seed
    - Biểu đồ tổng hợp Loss Comparison
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)
    runs_dir = os.path.join(log_base_dir, "training_runs")

    csv_files = glob.glob(os.path.join(runs_dir, "**", "training_log.csv"), recursive=True)

    if not csv_files:
        print("[Report] Chưa tìm thấy file training_log.csv nào trong logs/training_runs/")
        return

    summary_rows = []
    ticker_model_data = {}  # để vẽ biểu đồ so sánh

    for csv_path in sorted(csv_files):
        # Parse path: .../training_runs/TICKER/model_name/seed_N/training_log.csv
        parts = csv_path.replace("\\", "/").split("/")
        try:
            seed_dir_idx = next(i for i, p in enumerate(parts) if p.startswith("seed_"))
            ticker = parts[seed_dir_idx - 2]
            model_name = parts[seed_dir_idx - 1]
            seed = parts[seed_dir_idx].replace("seed_", "")
        except Exception:
            ticker, model_name, seed = "?", "?", "?"

        df_log = pd.read_csv(csv_path)
        if df_log.empty:
            continue

        best_idx = df_log['val_loss'].idxmin()
        summary_rows.append({
            'Ticker': ticker,
            'Model': model_name,
            'Seed': seed,
            'Total Epochs': len(df_log),
            'Best Epoch': int(df_log.loc[best_idx, 'epoch']),
            'Best Val Loss': round(float(df_log['val_loss'].min()), 6),
            'Final Val Loss': round(float(df_log['val_loss'].iloc[-1]), 6),
            'Avg Time/Epoch (s)': round(float(df_log['time_sec'].mean()) if 'time_sec' in df_log else 0.0, 2)
        })

        key = f"{ticker}/{model_name}/seed_{seed}"
        ticker_model_data[key] = df_log

    # ── Bảng tổng hợp ────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "training_runs_summary.csv")
    summary_md = os.path.join(output_dir, "training_runs_summary.md")
    df_summary.to_csv(summary_csv, index=False)
    try:
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("# Tổng Kết Tất Cả Thực Nghiệm Huấn Luyện\n\n")
            f.write(df_summary.to_markdown(index=False))
    except Exception:
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write(df_summary.to_string(index=False))

    print(f"\n[Report] ✅ Đã xuất bảng tổng hợp {len(summary_rows)} runs → {summary_md}")
    return df_summary
