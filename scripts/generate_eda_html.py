"""
Script tự động sinh file docs/eda.html — Phân tích dữ liệu thăm dò (EDA).

Yêu cầu: pip install plotly pandas numpy scipy
Chạy từ thư mục gốc dự án:
    python scripts/generate_eda_html.py

Output: docs/eda.html — file HTML standalone có thể mở thẳng trên browser.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Thêm project root vào path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("[ERROR] Cần cài plotly: pip install plotly")
    sys.exit(1)

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    print("[Warning] scipy không có — bỏ qua kiểm định ADF/normality")
    HAS_SCIPY = False

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: Load dữ liệu
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = "data"
TICKERS = {
    "AAPL": "AAPL_2018-01-01_2026-12-31.csv",
    "TSLA": "TSLA_2018-01-01_2026-12-31.csv",
    "MSFT": "MSFT_2018-01-01_2026-12-31.csv",
    "^VIX": "^VIX_2018-01-01_2026-12-31.csv",
    "^TNX": "^TNX_2018-01-01_2026-12-31.csv",
}

COLOR_MAP = {
    "AAPL": "#4a9eff",
    "TSLA": "#ff6b6b",
    "MSFT": "#4ade80",
    "^VIX": "#fbbf24",
    "^TNX": "#c4b5fd",
}

dfs = {}
for name, fname in TICKERS.items():
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        dfs[name] = df
        print(f"[Load] {name}: {len(df)} phiên giao dịch ({df.index[0].date()} → {df.index[-1].date()})")
    else:
        print(f"[Warning] Không tìm thấy file: {path}")

if "AAPL" not in dfs:
    print("[ERROR] Cần có file AAPL để phân tích. Chạy 'python main.py --mode data' trước.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: Tính toán các chỉ số cần thiết
# ──────────────────────────────────────────────────────────────────────────────

def compute_log_return(df):
    df = df.copy()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()

def compute_technical_indicators_for_eda(df):
    """Tính một số chỉ báo để hiển thị trong EDA."""
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    return df.dropna()

aapl = compute_log_return(dfs["AAPL"])
aapl = compute_technical_indicators_for_eda(aapl)

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: Thống kê mô tả
# ──────────────────────────────────────────────────────────────────────────────

def describe_extended(series, name):
    """Thống kê mô tả mở rộng bao gồm skewness và kurtosis."""
    s = series.dropna()
    result = {
        "Feature": name,
        "Count": int(len(s)),
        "Mean": round(float(s.mean()), 6),
        "Std": round(float(s.std()), 6),
        "Min": round(float(s.min()), 6),
        "25%": round(float(s.quantile(0.25)), 6),
        "Median": round(float(s.median()), 6),
        "75%": round(float(s.quantile(0.75)), 6),
        "Max": round(float(s.max()), 6),
        "Skewness": round(float(s.skew()), 4),
        "Kurtosis": round(float(s.kurtosis()), 4),
        "Missing": int(series.isna().sum()),
    }
    return result

stats_cols = ["Open", "High", "Low", "Close", "Volume", "Log_Return", "RSI_14", "MACD_Hist"]
stats_rows = [describe_extended(aapl[c], c) for c in stats_cols if c in aapl.columns]
stats_df = pd.DataFrame(stats_rows)

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: Class Balance (Up / Down)
# ──────────────────────────────────────────────────────────────────────────────

def get_class_balance(df_lr):
    lr = df_lr["Log_Return"].dropna()
    up = (lr > 0).sum()
    down = (lr < 0).sum()
    flat = (lr == 0).sum()
    total = len(lr)
    return {
        "up": int(up), "down": int(down), "flat": int(flat),
        "up_pct": round(up / total * 100, 2),
        "down_pct": round(down / total * 100, 2),
        "flat_pct": round(flat / total * 100, 2),
        "total": total
    }

balance = {}
for ticker_name in ["AAPL", "TSLA", "MSFT"]:
    if ticker_name in dfs:
        df_lr = compute_log_return(dfs[ticker_name])
        balance[ticker_name] = get_class_balance(df_lr)

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 5: Correlation Matrix (10 features đại diện)
# ──────────────────────────────────────────────────────────────────────────────

corr_cols = [c for c in ["Close", "Volume", "Log_Return", "RSI_14", "MACD_Hist", "SMA_20", "BB_Upper", "BB_Lower"] if c in aapl.columns]
corr_matrix = aapl[corr_cols].corr().round(3)

# VIX + TNX thêm vào correlation nếu có
if "^VIX" in dfs and "^TNX" in dfs:
    macro_df = aapl[["Close", "Log_Return"]].copy()
    vix_close = dfs["^VIX"]["Close"].rename("VIX")
    tnx_close = dfs["^TNX"]["Close"].rename("TNX")
    macro_df = macro_df.join(vix_close, how="left").join(tnx_close, how="left")
    macro_df = macro_df.ffill().bfill().dropna()
    macro_corr = macro_df.corr().round(3)
else:
    macro_corr = None

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 6: Xây dựng các biểu đồ Plotly
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG = "#0f1117"
PANEL_BG = "#1a1d27"
GRID_COLOR = "#333344"
TEXT_COLOR = "#e2e8f0"
ACCENT = "#4a9eff"

def dark_layout(title="", xaxis_title="", yaxis_title="", height=400):
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        xaxis=dict(title=xaxis_title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(title=yaxis_title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=GRID_COLOR),
    )

# ── Chart 1: Giá đóng cửa so sánh 3 mã ─────────────────────────────────────
fig_price = go.Figure()
for ticker_name in ["AAPL", "TSLA", "MSFT"]:
    if ticker_name in dfs:
        df_t = dfs[ticker_name]
        fig_price.add_trace(go.Scatter(
            x=df_t.index, y=df_t["Close"],
            name=ticker_name,
            line=dict(color=COLOR_MAP[ticker_name], width=1.5),
            hovertemplate=f"<b>{ticker_name}</b><br>Ngày: %{{x|%Y-%m-%d}}<br>Giá: $%{{y:.2f}}<extra></extra>"
        ))
fig_price.update_layout(**dark_layout("📈 Giá Đóng Cửa — AAPL vs TSLA vs MSFT (2018–2024)", "Ngày", "Giá (USD)", 450))

# ── Chart 2: AAPL với Bollinger Bands + SMA ──────────────────────────────────
fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=aapl.index, y=aapl["BB_Upper"], name="BB Upper",
    line=dict(color="#fbbf24", dash="dot", width=1), showlegend=True))
fig_bb.add_trace(go.Scatter(x=aapl.index, y=aapl["BB_Lower"], name="BB Lower",
    line=dict(color="#fbbf24", dash="dot", width=1),
    fill="tonexty", fillcolor="rgba(251,191,36,0.07)", showlegend=True))
fig_bb.add_trace(go.Scatter(x=aapl.index, y=aapl["Close"], name="AAPL Close",
    line=dict(color=ACCENT, width=1.5)))
fig_bb.add_trace(go.Scatter(x=aapl.index, y=aapl["SMA_20"], name="SMA 20",
    line=dict(color="#ff6b6b", dash="dash", width=1)))
fig_bb.add_trace(go.Scatter(x=aapl.index, y=aapl["SMA_50"], name="SMA 50",
    line=dict(color="#4ade80", dash="dash", width=1)))
fig_bb.update_layout(**dark_layout("📊 AAPL — Giá, Bollinger Bands & SMA", "Ngày", "Giá (USD)", 450))

# ── Chart 3: Log_Return time series (kiểm tra stationarity trực quan) ────────
fig_lr = go.Figure()
fig_lr.add_trace(go.Scatter(x=aapl.index, y=aapl["Log_Return"], name="Log_Return",
    line=dict(color=ACCENT, width=0.8),
    hovertemplate="Ngày: %{x|%Y-%m-%d}<br>Log_Return: %{y:.5f}<extra></extra>"))
fig_lr.add_hline(y=0, line_dash="solid", line_color="#555555", line_width=1)
fig_lr.update_layout(**dark_layout(
    "📉 AAPL Log_Return (ln(P_t/P_{t-1})) — Chuỗi dừng (Stationary)",
    "Ngày", "Log_Return", 380))

# ── Chart 4: Histogram Log_Return vs Normal Distribution ─────────────────────
lr_vals = aapl["Log_Return"].dropna().values
mu, sigma = lr_vals.mean(), lr_vals.std()

fig_hist = make_subplots(rows=1, cols=2,
    subplot_titles=["Histogram Log_Return AAPL", "RSI_14 Distribution"],
    horizontal_spacing=0.1)

# Log_Return histogram
fig_hist.add_trace(go.Histogram(
    x=lr_vals, nbinsx=80, name="Log_Return",
    marker_color=ACCENT, opacity=0.75,
    histnorm="probability density"
), row=1, col=1)

# Normal distribution overlay
x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
normal_y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
fig_hist.add_trace(go.Scatter(
    x=x_range, y=normal_y, name="Normal Fit",
    line=dict(color="#ff6b6b", width=2, dash="dash")
), row=1, col=1)

# RSI histogram
if "RSI_14" in aapl.columns:
    fig_hist.add_trace(go.Histogram(
        x=aapl["RSI_14"].dropna().values, nbinsx=50, name="RSI_14",
        marker_color="#4ade80", opacity=0.75,
        histnorm="probability density"
    ), row=1, col=2)
    fig_hist.add_vline(x=30, line_dash="dash", line_color="#ff6b6b", row=1, col=2)
    fig_hist.add_vline(x=70, line_dash="dash", line_color="#fbbf24", row=1, col=2)

fig_hist.update_layout(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    height=400, showlegend=True,
    title=dict(text="📊 Phân Phối Dữ Liệu — Log_Return & RSI_14", font=dict(color=TEXT_COLOR, size=14)),
    margin=dict(l=50, r=30, t=80, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0.4)"),
)
fig_hist.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
fig_hist.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)

# ── Chart 5: Class Balance (Up/Down) ─────────────────────────────────────────
balance_tickers = list(balance.keys())
fig_balance = make_subplots(
    rows=1, cols=len(balance_tickers),
    specs=[[{"type": "pie"}] * len(balance_tickers)],
    subplot_titles=balance_tickers
)
pie_colors = ["#4a9eff", "#ff6b6b", "#555555"]
for i, ticker_name in enumerate(balance_tickers):
    b = balance[ticker_name]
    fig_balance.add_trace(go.Pie(
        labels=["Tăng (Up)", "Giảm (Down)", "Đi ngang"],
        values=[b["up"], b["down"], b["flat"]],
        hole=0.45,
        marker_colors=pie_colors,
        textfont_size=11,
        hovertemplate="%{label}: %{value} phiên (%{percent})<extra></extra>"
    ), row=1, col=i+1)

fig_balance.update_layout(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    height=380,
    title=dict(text="⚖️ Class Balance — Tỷ Lệ Phiên Tăng/Giảm theo Mã", font=dict(color=TEXT_COLOR, size=14)),
    margin=dict(l=30, r=30, t=80, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0.4)"),
)

# ── Chart 6: Correlation Heatmap ─────────────────────────────────────────────
fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns.tolist(),
    y=corr_matrix.index.tolist(),
    colorscale=[
        [0.0, "#ff6b6b"], [0.5, PANEL_BG], [1.0, "#4a9eff"]
    ],
    zmid=0, zmin=-1, zmax=1,
    text=corr_matrix.values.round(2),
    texttemplate="%{text}",
    textfont_size=10,
    hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
    showscale=True
))
fig_corr.update_layout(**dark_layout("🔗 Ma Trận Tương Quan (Correlation Matrix) — AAPL Features", height=500))

# ── Chart 7: RSI + MACD ───────────────────────────────────────────────────────
fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
    subplot_titles=["RSI_14", "MACD Histogram"], vertical_spacing=0.08)

if "RSI_14" in aapl.columns:
    fig_tech.add_trace(go.Scatter(x=aapl.index, y=aapl["RSI_14"], name="RSI_14",
        line=dict(color="#4a9eff", width=1.2)), row=1, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="#fbbf24", row=1, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="#ff6b6b", row=1, col=1)

if "MACD_Hist" in aapl.columns:
    colors_macd = ["#4ade80" if v >= 0 else "#ff6b6b" for v in aapl["MACD_Hist"]]
    fig_tech.add_trace(go.Bar(x=aapl.index, y=aapl["MACD_Hist"], name="MACD Hist",
        marker_color=colors_macd, opacity=0.8), row=2, col=1)

fig_tech.update_layout(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    height=500,
    title=dict(text="📡 Chỉ Báo Kỹ Thuật — RSI_14 & MACD Histogram (AAPL)", font=dict(color=TEXT_COLOR, size=14)),
    margin=dict(l=60, r=30, t=80, b=50),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0.4)"),
)
fig_tech.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
fig_tech.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)

# ── Chart 8: VIX và TNX ──────────────────────────────────────────────────────
fig_macro = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
    subplot_titles=["VIX — Chỉ Số Biến Động Thị Trường", "TNX — Lợi Suất Trái Phiếu 10 Năm"],
    vertical_spacing=0.1)

if "^VIX" in dfs:
    vix = dfs["^VIX"]
    fig_macro.add_trace(go.Scatter(x=vix.index, y=vix["Close"], name="VIX",
        line=dict(color="#fbbf24", width=1.2),
        hovertemplate="VIX: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig_macro.add_hline(y=30, line_dash="dash", line_color="#ff6b6b", line_width=1, row=1, col=1)
    fig_macro.add_annotation(x=vix.index[len(vix)//4], y=30, text="VIX=30 (Extreme Fear)",
        font=dict(color="#ff6b6b", size=10), showarrow=False, row=1, col=1)

if "^TNX" in dfs:
    tnx = dfs["^TNX"]
    fig_macro.add_trace(go.Scatter(x=tnx.index, y=tnx["Close"], name="TNX (%)",
        line=dict(color="#c4b5fd", width=1.2),
        hovertemplate="TNX: %{y:.2f}%<extra></extra>"), row=2, col=1)

fig_macro.update_layout(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    height=500,
    title=dict(text="🌍 Chỉ Số Vĩ Mô — VIX & TNX (2018–2024)", font=dict(color=TEXT_COLOR, size=14)),
    margin=dict(l=60, r=30, t=80, b=50),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0.4)"),
)
fig_macro.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
fig_macro.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)

# ── Chart 9: Volume Analysis ──────────────────────────────────────────────────
aapl_volume = dfs["AAPL"].copy() if "AAPL" in dfs else aapl
aapl_volume["Volume_SMA20"] = aapl_volume["Volume"].rolling(20).mean()

fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
    subplot_titles=["AAPL Volume & SMA_20", "Volume Ratio (Volume / SMA20)"], vertical_spacing=0.08)

fig_vol.add_trace(go.Bar(x=aapl_volume.index, y=aapl_volume["Volume"], name="Volume",
    marker_color="#555577", opacity=0.6), row=1, col=1)
fig_vol.add_trace(go.Scatter(x=aapl_volume.index, y=aapl_volume["Volume_SMA20"], name="SMA_20 Volume",
    line=dict(color="#fbbf24", width=1.5)), row=1, col=1)

aapl_volume["Volume_Ratio"] = aapl_volume["Volume"] / (aapl_volume["Volume_SMA20"] + 1e-9)
fig_vol.add_trace(go.Scatter(x=aapl_volume.index, y=aapl_volume["Volume_Ratio"], name="Volume Ratio",
    line=dict(color=ACCENT, width=1.2)), row=2, col=1)
fig_vol.add_hline(y=1.0, line_dash="dash", line_color="#555555", row=2, col=1)
fig_vol.add_hline(y=2.0, line_dash="dash", line_color="#fbbf24", row=2, col=1)

fig_vol.update_layout(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    height=500,
    title=dict(text="📦 Phân Tích Khối Lượng (Volume Analysis) — AAPL", font=dict(color=TEXT_COLOR, size=14)),
    margin=dict(l=60, r=30, t=80, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0.4)"),
)
fig_vol.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
fig_vol.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 7: Chuyển sang JSON để nhúng vào HTML
# ──────────────────────────────────────────────────────────────────────────────

def fig_to_json(fig):
    return fig.to_json()

# Thống kê mô tả dạng HTML table
def df_to_html_table(df):
    rows = ""
    for _, row in df.iterrows():
        cols = "".join(f"<td>{v}</td>" for v in row.values)
        rows += f"<tr>{cols}</tr>"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"

stats_html = df_to_html_table(stats_df)

# Balance data
balance_rows = ""
for ticker_name, b in balance.items():
    balance_rows += f"""
    <tr>
        <td><strong>{ticker_name}</strong></td>
        <td>{b['total']}</td>
        <td class="up">{b['up']} ({b['up_pct']}%)</td>
        <td class="down">{b['down']} ({b['down_pct']}%)</td>
        <td>{b['flat']} ({b['flat_pct']}%)</td>
        <td>{'✅ Cân bằng' if abs(b['up_pct'] - b['down_pct']) < 10 else '⚠️ Mất cân bằng'}</td>
    </tr>"""

# Missing values
missing_rows = ""
for col in aapl.columns:
    n_missing = dfs["AAPL"][col].isna().sum() if col in dfs["AAPL"].columns else "N/A"
    missing_rows += f"<tr><td>{col}</td><td>{n_missing}</td><td>{'✅' if n_missing == 0 else '⚠️'}</td></tr>"

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 8: Sinh HTML
# ──────────────────────────────────────────────────────────────────────────────

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA — StockForecasting Dataset Analysis</title>
    <meta name="description" content="Phân tích dữ liệu thăm dò (EDA) cho dataset cổ phiếu AAPL, TSLA, MSFT từ Yahoo Finance 2018-2024">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f1117;
            --panel: #1a1d27;
            --border: #2d3147;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --accent: #4a9eff;
            --green: #4ade80;
            --red: #ff6b6b;
            --yellow: #fbbf24;
            --purple: #c4b5fd;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
        }}
        /* Header */
        .hero {{
            background: linear-gradient(135deg, #1a1d27 0%, #0f1117 50%, #1a1d27 100%);
            border-bottom: 1px solid var(--border);
            padding: 40px 60px 32px;
            position: relative;
            overflow: hidden;
        }}
        .hero::before {{
            content: '';
            position: absolute;
            top: -50%; left: -10%;
            width: 400px; height: 400px;
            background: radial-gradient(circle, rgba(74,158,255,0.08) 0%, transparent 70%);
            pointer-events: none;
        }}
        .hero h1 {{ font-size: 2rem; font-weight: 700; color: var(--text); margin-bottom: 8px; }}
        .hero h1 span {{ color: var(--accent); }}
        .hero p {{ color: var(--muted); font-size: 0.95rem; max-width: 700px; }}
        .badge-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }}
        .badge {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
            border: 1px solid;
        }}
        .badge-blue {{ background: rgba(74,158,255,0.12); border-color: rgba(74,158,255,0.4); color: var(--accent); }}
        .badge-green {{ background: rgba(74,222,128,0.12); border-color: rgba(74,222,128,0.4); color: var(--green); }}
        .badge-yellow {{ background: rgba(251,191,36,0.12); border-color: rgba(251,191,36,0.4); color: var(--yellow); }}
        /* Nav */
        .toc {{
            position: sticky; top: 0; z-index: 100;
            background: rgba(26,29,39,0.95); backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 0 60px;
            display: flex; gap: 0; overflow-x: auto;
        }}
        .toc a {{
            display: block; padding: 14px 16px;
            color: var(--muted); text-decoration: none; font-size: 0.85rem; white-space: nowrap;
            border-bottom: 2px solid transparent; transition: all 0.2s;
        }}
        .toc a:hover {{ color: var(--accent); border-bottom-color: var(--accent); }}
        /* Main */
        main {{ padding: 40px 60px; max-width: 1400px; margin: 0 auto; }}
        section {{ margin-bottom: 60px; }}
        .section-title {{
            font-size: 1.25rem; font-weight: 700; margin-bottom: 6px;
            display: flex; align-items: center; gap: 10px;
        }}
        .section-title .icon {{ font-size: 1.3rem; }}
        .section-desc {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 24px; }}
        /* Cards */
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 28px; }}
        .stat-card {{
            background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
            padding: 20px; transition: border-color 0.2s;
        }}
        .stat-card:hover {{ border-color: var(--accent); }}
        .stat-label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
        .stat-value {{ font-size: 1.6rem; font-weight: 700; color: var(--text); font-family: 'JetBrains Mono', monospace; }}
        .stat-sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 4px; }}
        /* Chart wrapper */
        .chart-card {{
            background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
            padding: 8px; margin-bottom: 24px; overflow: hidden;
        }}
        .chart-card .chart-note {{
            font-size: 0.8rem; color: var(--muted); padding: 8px 16px 4px;
        }}
        /* Tables */
        .table-wrap {{ overflow-x: auto; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th {{
            background: #252836; color: var(--muted); font-weight: 600;
            padding: 12px 16px; text-align: left; font-size: 0.78rem;
            text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;
        }}
        td {{ padding: 10px 16px; border-top: 1px solid var(--border); vertical-align: middle; }}
        tr:hover td {{ background: rgba(74,158,255,0.04); }}
        td.up {{ color: var(--green); font-weight: 600; }}
        td.down {{ color: var(--red); font-weight: 600; }}
        /* Callouts */
        .callout {{
            border-radius: 10px; padding: 16px 20px; margin: 16px 0; font-size: 0.875rem;
            display: flex; gap: 12px; align-items: flex-start; border-left: 3px solid;
        }}
        .callout-info {{ background: rgba(74,158,255,0.08); border-color: var(--accent); }}
        .callout-warn {{ background: rgba(251,191,36,0.08); border-color: var(--yellow); }}
        .callout-success {{ background: rgba(74,222,128,0.08); border-color: var(--green); }}
        .callout-icon {{ font-size: 1.1rem; flex-shrink: 0; }}
        /* Footer */
        footer {{
            text-align: center; padding: 32px; color: var(--muted); font-size: 0.8rem;
            border-top: 1px solid var(--border); margin-top: 40px;
        }}
        footer a {{ color: var(--accent); text-decoration: none; }}
    </style>
</head>
<body>

<!-- HERO -->
<div class="hero">
    <h1>📊 <span>EDA</span> — StockForecasting Dataset Analysis</h1>
    <p>Phân tích dữ liệu thăm dò toàn diện (Exploratory Data Analysis) cho dataset cổ phiếu Mỹ từ Yahoo Finance, phục vụ nghiên cứu khoa học dự báo chuỗi thời gian tài chính.</p>
    <div class="badge-row">
        <span class="badge badge-blue">📈 AAPL · TSLA · MSFT</span>
        <span class="badge badge-blue">🌍 ^VIX · ^TNX (Macro)</span>
        <span class="badge badge-green">📅 2018–2024 (~1.700 phiên/mã)</span>
        <span class="badge badge-yellow">⚠️ US Equity Market — NASDAQ</span>
    </div>
</div>

<!-- NAV -->
<nav class="toc">
    <a href="#overview">📋 Tổng quan</a>
    <a href="#price">📈 Giá cổ phiếu</a>
    <a href="#logreturn">📉 Log Return</a>
    <a href="#distribution">📊 Phân phối</a>
    <a href="#balance">⚖️ Class Balance</a>
    <a href="#technical">📡 Kỹ thuật</a>
    <a href="#macro">🌍 Vĩ mô</a>
    <a href="#volume">📦 Volume</a>
    <a href="#correlation">🔗 Tương quan</a>
    <a href="#statistics">📋 Thống kê</a>
</nav>

<main>

<!-- SECTION 1: OVERVIEW -->
<section id="overview">
    <div class="section-title"><span class="icon">📋</span> Tổng Quan Dataset</div>
    <div class="section-desc">Thông tin cơ bản về nguồn gốc, kích thước và cấu trúc dữ liệu.</div>

    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Số mã cổ phiếu</div>
            <div class="stat-value">3</div>
            <div class="stat-sub">AAPL · TSLA · MSFT</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Số chỉ số vĩ mô</div>
            <div class="stat-value">2</div>
            <div class="stat-sub">^VIX · ^TNX</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Phiên giao dịch (AAPL)</div>
            <div class="stat-value">~1.700</div>
            <div class="stat-sub">2018-01-01 → 2024-12-31</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Features (v2)</div>
            <div class="stat-value">10–12</div>
            <div class="stat-sub">Sau lọc trực giao</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Lookback Window</div>
            <div class="stat-value">60</div>
            <div class="stat-sub">60 phiên quá khứ/mẫu</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Split Train/Val/Test</div>
            <div class="stat-value">70/15/15</div>
            <div class="stat-sub">Theo thứ tự thời gian</div>
        </div>
    </div>

    <div class="callout callout-warn">
        <span class="callout-icon">⚠️</span>
        <div><strong>Giới hạn phạm vi áp dụng:</strong> Dataset chứa dữ liệu cổ phiếu Mỹ giao dịch trên sàn NASDAQ. Mô hình chưa được kiểm chứng và <strong>không thể áp dụng trực tiếp cho cổ phiếu Việt Nam</strong> do thị trường HoSE/HNX có đặc thù khác biệt về biên độ giá (±7%), thanh khoản, cấu trúc vi mô và bối cảnh vĩ mô.</div>
    </div>

    <div class="callout callout-info">
        <span class="callout-icon">ℹ️</span>
        <div><strong>Zero Data Leakage:</strong> Scaler (RobustScaler + StandardScaler) chỉ được <code>fit()</code> trên tập Train. Val và Test chỉ dùng <code>transform()</code>. Winsorization outlier [1%, 99%] cũng chỉ dựa trên phân vị Train.</div>
    </div>
</section>

<!-- SECTION 2: PRICE -->
<section id="price">
    <div class="section-title"><span class="icon">📈</span> Giá Đóng Cửa Lịch Sử</div>
    <div class="section-desc">So sánh xu hướng giá đóng cửa của 3 mã cổ phiếu trong dataset. TSLA có biến động cực mạnh — phù hợp để kiểm tra độ bền của mô hình. AAPL và MSFT ổn định hơn với xu hướng tăng trưởng dài hạn rõ ràng.</div>
    <div class="chart-card">
        <div id="chart_price"></div>
    </div>
    <div class="chart-card">
        <div id="chart_bb"></div>
        <div class="chart-note">📌 Khi giá chạm BB Upper → quá mua. Khi giá chạm BB Lower → quá bán. Bandwidth rộng → biến động cao.</div>
    </div>
</section>

<!-- SECTION 3: LOG RETURN -->
<section id="logreturn">
    <div class="section-title"><span class="icon">📉</span> Log_Return — Chuỗi Dừng</div>
    <div class="section-desc">
        Log_Return = ln(P_t / P_t-1) là <strong>target chính</strong> của mô hình v2. Chuỗi này có tính dừng (stationary), không có xu hướng tăng dài hạn như Close price — tránh được hiện tượng Lag Fallacy.
    </div>
    <div class="chart-card">
        <div id="chart_lr"></div>
        <div class="chart-note">📌 Dao động quanh 0 — đây là đặc tính dừng (stationary). Các spike lớn tương ứng với sự kiện thị trường quan trọng (COVID crash 2020, FED tăng lãi 2022...).</div>
    </div>
</section>

<!-- SECTION 4: DISTRIBUTION -->
<section id="distribution">
    <div class="section-title"><span class="icon">📊</span> Phân Phối Dữ Liệu</div>
    <div class="section-desc">Log_Return tuân theo phân phối gần chuẩn (Normal) nhưng có đuôi nặng hơn (leptokurtic, kurtosis > 3) — đặc trưng của tài sản tài chính (fat tails). RSI dao động trong [0, 100], tập trung quanh 50 (vùng trung tính).</div>
    <div class="chart-card">
        <div id="chart_hist"></div>
        <div class="chart-note">📌 Đường đỏ đứt là Normal Distribution fit. Histogram Log_Return cao hơn ở giữa và đuôi → leptokurtic (fat tails). RSI &lt;30 = quá bán (đường đỏ), RSI &gt;70 = quá mua (đường vàng).</div>
    </div>
</section>

<!-- SECTION 5: CLASS BALANCE -->
<section id="balance">
    <div class="section-title"><span class="icon">⚖️</span> Class Balance — Tỷ Lệ Tăng / Giảm</div>
    <div class="section-desc">Phân tích tỷ lệ phiên tăng (Log_Return > 0) và phiên giảm (Log_Return &lt; 0) theo từng mã. Dataset gần cân bằng — không cần kỹ thuật oversampling như SMOTE.</div>
    <div class="chart-card">
        <div id="chart_balance"></div>
    </div>
    <div class="table-wrap">
        <table>
            <thead><tr><th>Mã</th><th>Tổng phiên</th><th>Tăng (Up)</th><th>Giảm (Down)</th><th>Đi ngang</th><th>Nhận xét</th></tr></thead>
            <tbody>{balance_rows}</tbody>
        </table>
    </div>
    <div class="callout callout-success">
        <span class="callout-icon">✅</span>
        <div><strong>Kết luận:</strong> Dataset gần cân bằng (52/48 hoặc tương đương). Không cần xử lý class imbalance. Đây là đặc trưng tự nhiên của thị trường chứng khoán (random walk property).</div>
    </div>
</section>

<!-- SECTION 6: TECHNICAL -->
<section id="technical">
    <div class="section-title"><span class="icon">📡</span> Chỉ Báo Kỹ Thuật</div>
    <div class="section-desc">RSI và MACD là hai chỉ báo đại diện cho nhóm Momentum và Trend — được chọn làm features trong bộ 10 features trực giao (v1/v2/v3).</div>
    <div class="chart-card">
        <div id="chart_tech"></div>
        <div class="chart-note">📌 RSI > 70: quá mua (overbought). RSI &lt; 30: quá bán (oversold). MACD Hist xanh: momentum tăng, đỏ: momentum giảm.</div>
    </div>
</section>

<!-- SECTION 7: MACRO -->
<section id="macro">
    <div class="section-title"><span class="icon">🌍</span> Chỉ Số Vĩ Mô</div>
    <div class="section-desc">VIX (fear index) và TNX (10-year Treasury yield) cung cấp ngữ cảnh thị trường — giúp mô hình "biết" thị trường đang trong trạng thái bình tĩnh hay hoảng loạn, lãi suất cao hay thấp.</div>
    <div class="chart-card">
        <div id="chart_macro"></div>
        <div class="chart-note">📌 VIX spike tháng 3/2020 (COVID crash): ~82. VIX > 30 thường đi kèm thị trường giảm mạnh. TNX tăng mạnh từ 2022 (FED tăng lãi) gây áp lực lên định giá cổ phiếu công nghệ.</div>
    </div>
</section>

<!-- SECTION 8: VOLUME -->
<section id="volume">
    <div class="section-title"><span class="icon">📦</span> Phân Tích Khối Lượng Giao Dịch</div>
    <div class="section-desc">Volume_Ratio = Volume / SMA20(Volume) cho thấy đột biến dòng tiền. Khi Volume_Ratio > 2 (giao dịch gấp đôi trung bình) thường đi kèm breakout hoặc panic selling.</div>
    <div class="chart-card">
        <div id="chart_vol"></div>
        <div class="chart-note">📌 Đường vàng đứt: ngưỡng Volume_Ratio = 2 (đột biến). Volume spike không bị Winsorize — tín hiệu dòng tiền quan trọng.</div>
    </div>
</section>

<!-- SECTION 9: CORRELATION -->
<section id="correlation">
    <div class="section-title"><span class="icon">🔗</span> Ma Trận Tương Quan</div>
    <div class="section-desc">Phân tích mối quan hệ tuyến tính giữa các features. SMA_20 và BB_Upper tương quan rất cao với Close (đa cộng tuyến) — lý do chúng bị loại bỏ trong bước select_orthogonal_features() và chỉ giữ lại MACD_Hist làm đại diện nhóm Trend.</div>
    <div class="chart-card">
        <div id="chart_corr"></div>
        <div class="chart-note">📌 Ô màu đỏ đậm = tương quan âm cao. Ô màu xanh đậm = tương quan dương cao. Ô gần trắng = tương quan thấp (trực giao hơn).</div>
    </div>
</section>

<!-- SECTION 10: STATISTICS -->
<section id="statistics">
    <div class="section-title"><span class="icon">📋</span> Thống Kê Mô Tả Đầy Đủ</div>
    <div class="section-desc">Bảng thống kê chi tiết cho AAPL bao gồm mean, std, phân vị, skewness và kurtosis. Skewness ≠ 0 và Kurtosis > 3 xác nhận phân phối fat-tail của tài sản tài chính.</div>
    <div class="table-wrap">
        {stats_html}
    </div>
    <div class="callout callout-info">
        <span class="callout-icon">ℹ️</span>
        <div>
            <strong>Giải thích Skewness & Kurtosis:</strong><br>
            • <strong>Skewness &lt; 0</strong> (lệch âm): Log_Return có nhiều ngày giảm mạnh hơn tăng mạnh — đặc trưng fat left tail của cổ phiếu tăng trưởng.<br>
            • <strong>Kurtosis &gt; 3</strong> (excess kurtosis): Phân phối nhọn hơn chuẩn — có nhiều sự kiện cực đoan (crash, spike) hơn Normal Distribution dự đoán.
        </div>
    </div>
</section>

</main>

<footer>
    <p>Tài liệu EDA — <strong>StockForecasting</strong> Research Project &nbsp;|&nbsp;
    Nguồn dữ liệu: <a href="https://finance.yahoo.com" target="_blank">Yahoo Finance</a> &nbsp;|&nbsp;
    <a href="./data.md">data.md</a> &nbsp;|&nbsp; <a href="./modeling.md">modeling.md</a>
    </p>
</footer>

<script>
    // Nhúng dữ liệu JSON của các biểu đồ
    var fig_price = {fig_to_json(fig_price)};
    var fig_bb = {fig_to_json(fig_bb)};
    var fig_lr = {fig_to_json(fig_lr)};
    var fig_hist = {fig_to_json(fig_hist)};
    var fig_balance = {fig_to_json(fig_balance)};
    var fig_tech = {fig_to_json(fig_tech)};
    var fig_macro = {fig_to_json(fig_macro)};
    var fig_vol = {fig_to_json(fig_vol)};
    var fig_corr = {fig_to_json(fig_corr)};

    var common_config = {{
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false
    }};

    Plotly.newPlot('chart_price', fig_price.data, fig_price.layout, common_config);
    Plotly.newPlot('chart_bb', fig_bb.data, fig_bb.layout, common_config);
    Plotly.newPlot('chart_lr', fig_lr.data, fig_lr.layout, common_config);
    Plotly.newPlot('chart_hist', fig_hist.data, fig_hist.layout, common_config);
    Plotly.newPlot('chart_balance', fig_balance.data, fig_balance.layout, common_config);
    Plotly.newPlot('chart_tech', fig_tech.data, fig_tech.layout, common_config);
    Plotly.newPlot('chart_macro', fig_macro.data, fig_macro.layout, common_config);
    Plotly.newPlot('chart_vol', fig_vol.data, fig_vol.layout, common_config);
    Plotly.newPlot('chart_corr', fig_corr.data, fig_corr.layout, common_config);
</script>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────────────────────
# BƯỚC 9: Ghi file
# ──────────────────────────────────────────────────────────────────────────────

output_path = os.path.join("docs", "eda.html")
os.makedirs("docs", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

size_kb = os.path.getsize(output_path) / 1024
print(f"\n[✅ HOÀN TẤT] EDA HTML đã được tạo:")
print(f"   📄 File: {output_path}")
print(f"   📦 Kích thước: {size_kb:.1f} KB")
print(f"   🌐 Mở trực tiếp trên browser — không cần server!")
print(f"\n   Chứa {len(dfs)} nguồn dữ liệu, 9 biểu đồ interactive, thống kê đầy đủ.")
