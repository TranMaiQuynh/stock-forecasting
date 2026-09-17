import os
import sys
import glob
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

DATA_DIR = "data"

ALL_TICKERS = [
    "^TNX", "^VIX",
    "AAPL", "MSFT", "TSLA",
    "AMZN", "NVDA", "GOOGL", "ORCL", "IBM", 
    "INTC", "META", "JPM", "BAC", "MA", 
    "MCD", "NKE", "KO", "PEP", "COST", 
    "LLY", "JNJ", "ABBV"
]

def main():
    print("=================================================================")
    print("TIẾN HÀNH DỌN DẸP VÀ CHUẨN HÓA THƯ MỤC DATA")
    print("Mỗi mã cổ phiếu & chỉ số chỉ giữ duy nhất 1 file 2018-01-01_2026-12-31.csv")
    print("=================================================================")

    # 1. Đảm bảo ^TNX và ^VIX có file 2018-01-01_2026-12-31.csv
    for macro in ["^TNX", "^VIX"]:
        full_file = os.path.join(DATA_DIR, f"{macro}_2018-01-01_2026-12-31.csv")
        if not os.path.exists(full_file):
            print(f"\nĐang tạo file tổng {macro}_2018-01-01_2026-12-31.csv...")
            f_2018 = os.path.join(DATA_DIR, f"{macro}_2018-01-01_2024-12-31.csv")
            f_2024 = os.path.join(DATA_DIR, f"{macro}_2024-01-01_2026-12-31.csv")
            if os.path.exists(f_2018) and os.path.exists(f_2024):
                df1 = pd.read_csv(f_2018, index_col=0, parse_dates=True)
                df2 = pd.read_csv(f_2024, index_col=0, parse_dates=True)
                df = pd.concat([df1, df2])
                df = df[~df.index.duplicated(keep='last')].sort_index()
                df.to_csv(full_file)
                print(f"  -> Hợp nhất thành công {len(df)} dòng vào: {full_file}")
            else:
                print(f"  -> Tải mới {macro} từ Yahoo Finance...")
                df = yf.download(macro, start="2018-01-01", end="2026-12-31", progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(inplace=True)
                df.to_csv(full_file)
                print(f"  -> Đã tải và lưu {len(df)} dòng vào: {full_file}")

    # 2. Kiểm tra tất cả các mã đều có file full hợp lệ
    missing = []
    for ticker in ALL_TICKERS:
        full_file = os.path.join(DATA_DIR, f"{ticker}_2018-01-01_2026-12-31.csv")
        if not os.path.exists(full_file):
            missing.append(ticker)
        else:
            df = pd.read_csv(full_file, index_col=0, parse_dates=True)
            print(f"[OK] {ticker}: {len(df)} phiên ({df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')})")
            
    if missing:
        print(f"\n[CẢNH BÁO] Thiếu file full cho các mã: {missing}")
        return

    # 3. Tiến hành xóa các file phân đoạn thừa
    print("\n--- Tiến hành xóa các file phân đoạn thừa ---")
    patterns_to_delete = [
        os.path.join(DATA_DIR, "*_2018-01-01_2024-12-31.csv"),
        os.path.join(DATA_DIR, "*_2024-01-01_2026-12-31.csv"),
        os.path.join(DATA_DIR, "*_2025-01-01_2026-12-31.csv"),
    ]
    
    deleted_count = 0
    for pattern in patterns_to_delete:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"  Đã xóa: {os.path.basename(file_path)}")
                deleted_count += 1
            except Exception as e:
                print(f"  Lỗi khi xóa {file_path}: {e}")

    print(f"\nTổng số file phân đoạn đã xóa: {deleted_count}")

    # 4. Liệt kê lại các file còn lại trong data/
    remaining_files = sorted(os.listdir(DATA_DIR))
    print(f"\n--- DANH SÁCH {len(remaining_files)} FILE CÒN LẠI TRONG THƯ MỤC data/ ---")
    for f in remaining_files:
        size_kb = os.path.getsize(os.path.join(DATA_DIR, f)) / 1024
        print(f"  ✓ {f} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
