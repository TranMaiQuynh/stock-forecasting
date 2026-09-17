import os
import sys
import time
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS_18 = [
    "AMZN", "NVDA", "GOOGL", "ORCL", "IBM", 
    "INTC", "META", "JPM", "BAC", "MA", 
    "MCD", "NKE", "KO", "PEP", "COST", 
    "LLY", "JNJ", "ABBV"
]

def fetch_and_clean(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    print(f"\n[{ticker}] Đang tải dữ liệu từ {start} đến {end}...")
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Đảm bảo các cột cần thiết tồn tại và đúng thứ tự
            required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            avail_cols = [c for c in required_cols if c in df.columns]
            df = df[avail_cols].copy()
            df.dropna(inplace=True)
            
            if len(df) == 0:
                raise ValueError(f"Không có dữ liệu cho {ticker}")
            
            print(f"[{ticker}] Tải thành công {len(df)} phiên giao dịch (từ {df.index.min().strftime('%Y-%m-%d')} đến {df.index.max().strftime('%Y-%m-%d')})")
            return df
        except Exception as e:
            print(f"[{ticker}] Lần thử {attempt}/{retries} thất bại: {e}")
            if attempt < retries:
                time.sleep(2)
            else:
                raise

def save_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path)
    print(f"  -> Đã lưu {len(df)} dòng vào: {path}")

def main():
    print("=================================================================")
    print("BẮT ĐẦU TẢI VÀ ĐỒNG BỘ DỮ LIỆU CỔ PHIẾU TỪ YAHOO FINANCE")
    print("=================================================================")

    # 1. Tải 18 mã mới: 2018-01-01 đến 2026-12-31
    print(f"\n--- Giai đoạn 1: Tải 18 mã cổ phiếu ({len(TICKERS_18)} mã) từ 2018-01-01 đến 2026-12-31 ---")
    for ticker in TICKERS_18:
        try:
            df = fetch_and_clean(ticker, "2018-01-01", "2026-12-31")
            
            # File chính: 2018-01-01 đến 2026-12-31
            save_csv(df, f"{ticker}_2018-01-01_2026-12-31.csv")
            
            # File phân đoạn như AAPL (phục vụ đồng bộ cả train 2018-2024 và inference 2024-2026)
            df_train = df.loc[:'2024-12-31']
            if len(df_train) > 0:
                save_csv(df_train, f"{ticker}_2018-01-01_2024-12-31.csv")
                
            df_inf = df.loc['2024-01-01':]
            if len(df_inf) > 0:
                save_csv(df_inf, f"{ticker}_2024-01-01_2026-12-31.csv")
                
            time.sleep(0.5) # Tránh Yahoo Finance rate limit
        except Exception as e:
            print(f"LỖI tải {ticker}: {e}")

    # 2. Tải mã MSFT và TSLA: 2025-01-01 đến 2026-12-31 (và bổ sung 2024-2026, 2018-2026)
    print(f"\n--- Giai đoạn 2: Tải MSFT và TSLA từ Yahoo Finance ---")
    for ticker in ["MSFT", "TSLA"]:
        try:
            # Tải toàn diện từ 2024 đến 2026 để có cả 2025-2026 và 2024-2026
            df_recent = fetch_and_clean(ticker, "2024-01-01", "2026-12-31")
            
            # Theo yêu cầu chính xác: 2025-01-01 đến 2026-12-31
            df_2025 = df_recent.loc['2025-01-01':]
            save_csv(df_2025, f"{ticker}_2025-01-01_2026-12-31.csv")
            
            # Đồng bộ thêm phiên bản 2024-01-01 đến 2026-12-31 như AAPL cho inference pipeline
            save_csv(df_recent, f"{ticker}_2024-01-01_2026-12-31.csv")
            
            # Đồng bộ phiên bản toàn diện 2018-01-01 đến 2026-12-31
            path_prev = os.path.join(DATA_DIR, f"{ticker}_2018-01-01_2024-12-31.csv")
            if os.path.exists(path_prev):
                df_prev = pd.read_csv(path_prev, index_col=0, parse_dates=True)
                df_full = pd.concat([df_prev, df_recent])
                df_full = df_full[~df_full.index.duplicated(keep='last')].sort_index()
                save_csv(df_full, f"{ticker}_2018-01-01_2026-12-31.csv")
                
            time.sleep(0.5)
        except Exception as e:
            print(f"LỖI tải {ticker}: {e}")

    # 3. Đồng bộ AAPL toàn diện 2018-01-01 đến 2026-12-31 nếu chưa có
    aapl_full_path = os.path.join(DATA_DIR, "AAPL_2018-01-01_2026-12-31.csv")
    if not os.path.exists(aapl_full_path):
        print(f"\n--- Giai đoạn 3: Đồng bộ AAPL toàn diện 2018-01-01 đến 2026-12-31 ---")
        try:
            aapl_2018 = pd.read_csv(os.path.join(DATA_DIR, "AAPL_2018-01-01_2024-12-31.csv"), index_col=0, parse_dates=True)
            aapl_2024 = pd.read_csv(os.path.join(DATA_DIR, "AAPL_2024-01-01_2026-12-31.csv"), index_col=0, parse_dates=True)
            df_aapl_full = pd.concat([aapl_2018, aapl_2024])
            df_aapl_full = df_aapl_full[~df_aapl_full.index.duplicated(keep='last')].sort_index()
            save_csv(df_aapl_full, "AAPL_2018-01-01_2026-12-31.csv")
        except Exception as e:
            print(f"Lỗi đồng bộ AAPL full: {e}")

    print("\n=================================================================")
    print("HOÀN TẤT QUÁ TRÌNH TẢI VÀ LƯU TRỮ DỮ LIỆU!")
    print("=================================================================")

if __name__ == "__main__":
    main()
