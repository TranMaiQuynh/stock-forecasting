# StockForecasting

Dự án học và phát triển kỹ thuật dự báo chuỗi thời gian, kế thừa
[Quant Forecast Lab](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning)
của Jordi Corbilla. Mã nguồn hiện nằm trực tiếp tại Git root
`D:\Github\StockForecasting`; metadata kế thừa khai báo phiên bản 2.1.0.
Chưa xác định upstream commit.

Giữ nguyên [LICENSE Apache-2.0 của upstream](LICENSE)
và [CITATION](CITATION.cff).
Các kết quả lịch sử trong tài liệu upstream chưa được tái lập ở workspace này.

Phiên bản phát triển hiện có một pipeline nghiên cứu offline:
CSV một tài sản → kiểm tra dữ liệu → chia theo thời gian →
ADF/KPSS và ACF/PACF trên training → thử lookback/Ridge alpha →
chọn bằng validation → chấm final test → lưu artifact và nạp lại để dự báo.

Bằng chứng, quyết định và công việc còn lại:
[Engineering review](docs/ENGINEERING_REVIEW.md).
Các lệnh LSTM kế thừa được lưu trong [Upstream guide](docs/UPSTREAM_GUIDE.md).

## Cấu trúc dự án

~~~text
StockForecasting/               # Git root: chạy lệnh tại đây
├── quant_forecast_lab/         # Thư viện nghiên cứu và CLI
├── tests/                      # Test nhẹ và offline integration tests
├── tests_runtime/              # Test cần TensorFlow và model mẫu
├── benchmarks/                 # Script và kết quả benchmark kế thừa
├── examples/                   # Notebook và reference model đang được dùng
├── docs/                       # Kiến trúc, phương pháp, audit, hướng dẫn gốc
├── log/                        # Kết quả chạy và hồ sơ di chuyển; Git ignore
├── .github/                    # CI và mẫu issue/PR
├── stock_prediction_*.py       # Entry points LSTM kế thừa, giữ tên tương thích
├── pyproject.toml              # Khai báo package và dependency chính
├── requirements.txt            # Cài package từ pyproject.toml
├── environment.yml             # Tạo Python 3.12 bằng Conda nếu sử dụng
├── LICENSE / CITATION.cff      # Giấy phép và ghi nhận nguồn
└── README.md                   # Hướng dẫn hiện tại
~~~

## Chạy từ source trên PowerShell

Runtime đầy đủ của upstream yêu cầu Python 3.11 hoặc 3.12 và
TensorFlow 2.18.1. Môi trường hiện tại chỉ có Python 3.13.3:
đã kiểm tra phần offline, chưa xác nhận runtime TensorFlow.

Khi có Python 3.12, chạy từ Git root:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe -m quant_forecast_lab.cli experiment --synthetic --asset SYNTHETIC
~~~

Nếu chỉ cần phần offline, cài stack nhẹ trong môi trường riêng:

~~~powershell
python -m pip install numpy pandas scikit-learn matplotlib "statsmodels>=0.14,<0.16" pytest ruff
~~~

Các lệnh `python -m quant_forecast_lab.cli ...` bên dưới chạy trực tiếp từ
`D:\Github\StockForecasting`. Chúng không cần cài package editable hoặc tải
dữ liệu mạng khi chạy từ source.

## Dữ liệu và protocol

CSV cần hai cột `Date,Close`, một tài sản, ít nhất 60 quan sát,
ngày duy nhất theo thứ tự tăng và giá dương/hữu hạn. Ví dụ schema:

~~~csv
Date,Close
2024-01-02,100.0
2024-01-03,101.0
~~~

Hai dòng minh họa chưa đủ để chạy. Không tự sắp xếp, fill hoặc
loại dòng thiếu để làm dữ liệu hợp lệ. Lag tính theo quan sát đầu vào;
chưa xác minh lịch sàn, corporate actions hoặc point-in-time correctness.

Với snapshot có sẵn:

~~~powershell
python -m quant_forecast_lab.cli experiment --csv path\to\close.csv --asset AAPL --data-kind market --lookbacks 1 5 10 --ridge-alphas 1 10
~~~

`--data-kind market` là mô tả nguồn do người cung cấp xác nhận,
không phải chứng nhận dữ liệu của chương trình.

Mặc định chia 60% training, 20% validation, 20% final test.
Các ứng viên dùng chung target timestamps; StandardScaler chỉ fit training.
Validation chọn cấu hình Ridge. Không refit; final test không chọn tham số.
Baseline return bằng 0 tương ứng giữ nguyên giá gần nhất.

ACF/PACF ghi ứng viên và interval theo từng lag; chúng không tự quyết định
lookback hoặc hyperparameters. Model hiện chỉ dùng lịch sử log-return,
chưa có ablation nhóm technical/macro features hoặc walk-forward training
trong runner mới.

## Log và artifact

Mỗi run tạo `log/ridge_log_return/<run_id>/`, được Git ignore:

- `manifest.json`, `config.json`, `environment.json`: trạng thái, cấu hình,
  phiên bản, source hash và artifact checksums.
- `input_snapshot.csv`, `data_manifest.json`, `split_manifest.json`:
  dữ liệu thực dùng và ranh giới đánh giá.
- `analysis/ACF_PACF_Analysis_<date>.txt`, `z_table_<date>.txt`,
  `stationary_test.txt`, CSV/JSON và hình ACF/PACF.
- `model_summary.txt`, `trials.csv`, `artifacts/model.json`:
  cấu hình được chọn, các trial đã chạy và tham số model/scaler.
- `predictions.csv`, `metrics.json`, `residual_diagnostics.json`:
  kết quả ngoài mẫu và chẩn đoán.
- `future_forecast.json`, `forecast_values.txt`, `figures/forecast.png`,
  `summary.md` và `run.log`.

Artifact JSON được nạp lại và đối chiếu dự báo ở mọi test origin.
Dự báo tương lai chưa có actual và chưa đoán ngày giao dịch kế tiếp.
Run thất bại giữ traceback và trạng thái failed.

Dữ liệu tổng hợp dùng kiểm tra kỹ thuật, không phải benchmark thị trường.
Chưa chứng minh lợi thế dự báo trên thị trường hoặc khả năng sinh lợi.
Runner mới chưa có calibrated uncertainty, trading execution hoặc deployment.

## Kiểm tra chất lượng và học tập

~~~powershell
python -m pytest -q tests
ruff check quant_forecast_lab tests tests_runtime
python -m compileall -q quant_forecast_lab
python -m quant_forecast_lab.cli --help
~~~

`tests_runtime` cần đầy đủ TensorFlow và fixture; test nhẹ không thay thế
việc chạy bộ này. Workflow ở Git root kiểm tra phần offline trên Python
3.12; chưa xác minh một lần chạy GitHub Actions trong phiên local.

Phần kế thừa gồm kiến trúc LSTM, data/training scripts, evaluation utilities
và các thí nghiệm lịch sử. Phần phát triển tại workspace này gồm sửa lỗi
đánh giá/drawdown/uncertainty, training diagnostics, vòng đời run và pipeline
offline có kiểm tra leakage. Chưa tuyên bố tính mới nghiên cứu.

Bài tập người học: dùng một chuỗi sáu mức giá để tính tay hai return inputs
và next-return target; đối chiếu `return_windows` và
`tests/test_research.py::test_window_targets_and_origin_history_are_aligned`.
Sau đó giải thích vì sao thay đổi final test không được làm đổi scaler,
training diagnostics hoặc cấu hình đã chọn từ validation.
