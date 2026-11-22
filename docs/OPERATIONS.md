# Vận hành batch và hiểu kết quả

## Hệ thống hiện làm gì?

Pipeline nghiên cứu đã có bước kiểm tra dữ liệu, chọn mô hình trên development
folds, đánh giá holdout và lưu model. Phần vận hành mới dùng **checkpoint đã
đóng băng** để ghi dự báo trước khi biết kết quả, rồi chấm dự báo khi nhãn xuất
hiện. Batch không train thêm hoặc tự nâng cấp model.

Luồng hàng ngày:

1. Đọc deployment đang active; kiểm tra checksum code, model và protocol.
2. Dùng lịch sở giao dịch xác định phiên vừa đóng và phiên mục tiêu kế tiếp.
3. Nhận quyền xử lý job cho từng tài sản/phiên trong SQLite operations store.
4. Lấy snapshot giá; kiểm tra nguồn, tính liên tục, ngày mới nhất và checksum.
5. Chấm các dự báo cũ đã có giá thực tế; ghi dự báo mới nếu còn trong cửa sổ.
6. Lưu checkpoint từng bước, summary và manifest; cập nhật heartbeat.
7. Một tác vụ health độc lập tìm phiên bị bỏ lỡ, nhãn quá hạn và job hết lease.

Lỗi chấm nhãn của một tài sản không được âm thầm làm mất dự báo độc lập hôm
nay hoặc chặn tài sản khác. Batch trả mã lỗi và giữ đầy đủ failed-run artifacts.
Chạy lại dùng snapshot/checkpoint cũ; ledger giữ dự báo đầu tiên và chống trùng.

**Thuật ngữ:** snapshot là bản dữ liệu đã lưu cùng nguồn gốc; artifact là file
kết quả có checksum; protocol là quy tắc đánh giá đã đóng băng; origin là phiên
cuối dùng làm đầu vào; target là phiên cần dự báo; label là kết quả thực tế.
Lease là quyền xử lý có hạn giờ, owner token ngăn tiến trình cũ ghi tiếp sau
khi bị thay thế. Checkpoint ghi bước nào đã xong. Idempotency nghĩa là chạy lại
không nhân đôi cùng một dự báo. Heartbeat là dấu thời gian tác vụ còn hoạt động.
Deployment ở đây là cấu hình vận hành, không phải chứng nhận chất lượng ML.

## Cài bản nhẹ và chạy demo

Dùng Python 3.12 và uv 0.13.7. Trong Git root, tạo môi trường riêng:

```powershell
python -m pip install "uv==0.13.7"
$env:UV_PROJECT_ENVIRONMENT = '.venv/replay-demo'
uv sync --locked --extra research --extra dev --no-editable
.\.venv\replay-demo\Scripts\python.exe -I -c "import importlib.util, quant_forecast_lab; print(quant_forecast_lab.__file__); assert importlib.util.find_spec('tensorflow') is None"
.\.venv\replay-demo\Scripts\python.exe -I -m pytest tests --import-mode=importlib
.\.venv\replay-demo\Scripts\python.exe scripts/demo_batch.py --installed-package
```

Môi trường neural đầy đủ vẫn cài qua `requirements.txt` hoặc thêm `--extra neural`.
Không sync đè một môi trường đang phục vụ pilot; giữ release cũ để khôi phục.
`uv.lock` khóa dependency; source hash trong deployment khóa code Python.
Đổi code cần kiểm thử, cài release riêng, tạo deployment ID mới và kích hoạt rõ ràng.

Demo tự tạo dữ liệu giả và thử: chưa từng chạy job, tải tạm thời thất bại,
lỗi ngay sau khi đã ghi dự báo, chạy lại, chấm nhãn cuối sau khi hết budget,
rollback và backup/restore. Receipt ghi `market_evidence=false`.
Hai nhãn giả chỉ xác minh đường đi dữ liệu, không chứng minh model dự báo tốt.

## Pilot AAPL/MSFT hiện tại

Cấu hình active: `log/operations/live/config-release-v3.json`, ID `equity-pilot-v3`.
Đây là phiên bản **vận hành**, khác nghiên cứu v2 monthly-refit trước đây.
Các release/runtime v1 và v2 được giữ để rollback; release active sử dụng
`.venv/batch312-v3`, thêm nghiệm thu theo phiên và báo cáo vi phạm thời hạn cũ.
Model và protocol đến từ `log/validation_20261116/market_preparation.json`.
Đường dẫn tuyệt đối trong receipt thuộc máy này; clone mới phải tự chuẩn bị
candidate và receipt bằng các lệnh `shadow` được mô tả trong README/CLI.
Không lấy model mẫu synthetic để chạy live.

Hai task Windows đã đăng ký dưới tài khoản `aliza`, quyền Limited:

| Task | Chu kỳ | Mục đích |
| --- | --- | --- |
| StockForecasting-Pilot-Run | 15 phút | Kiểm tra lịch, acquire/score/record khi cần |
| StockForecasting-Pilot-Health | 5 phút | Phát hiện job không chạy và dữ liệu bị thiếu |

Task chạy ẩn, tránh chồng nhiều instance, timeout 10 phút. Wrapper đã copy vào
`log/operations/live/runner/6a375d9/`, dùng package cài trong `.venv/batch312-v3`,
làm việc từ thư mục state thay vì source root. Lúc khởi tạo, cả hai đã chạy thật
và trả exit 0 trước phiên bắt đầu. Sau khi có vi phạm, health trả exit 1 để báo
cảnh báo; batch có thể vẫn trả 0 vì đã xử lý xong công việc hiện tại.
ExecutionPolicy Bypass chỉ áp dụng cho tiến trình task;
không thay đổi execution policy toàn máy.

**Máy phải bật, có mạng, đồng hồ đúng và tài khoản phải đang đăng nhập.** Đây
là Interactive Task Scheduler pilot. Health cùng máy không thể báo khi cả máy
tắt; chưa có giám sát từ host bên ngoài hoặc gửi cảnh báo. Không coi đây là
dịch vụ production chạy liên tục.

Receipt Windows ngày 23/11 xác nhận task không chạy khi dùng pin, dừng khi
chuyển sang pin và không tự đánh thức máy. Các tùy chọn này được giữ nguyên;
cần cắm nguồn và chủ động bảo đảm máy còn hoạt động/đăng nhập trong phiên pilot.
Đây là điều kiện có thể gây gián đoạn, chưa phải bằng chứng nguyên nhân từng
khoảng trống lịch sử. Muốn vận hành liên tục cần chuyển sang host phù hợp.

Ở lần khởi tạo, lần đầu đủ điều kiện cho origin 2026-11-16 là khoảng
04:30 ngày 17/11/2026 giờ Việt Nam (21:30 UTC), còn phụ thuộc dữ liệu nhà cung
cấp và lần polling tiếp theo. Calendar xử lý DST/ngày nghỉ/đóng cửa sớm;
không suy diễn mọi phiên đều cùng giờ UTC. Calendar XNAS hiện là alias XNYS
trong dependency; chưa kiểm chứng riêng lịch Nasdaq.

Kiểm tra thủ công, từ Git root:

```powershell
.\.venv\batch312-v3\Scripts\python.exe -I -m quant_forecast_lab.cli batch health --state-db log/operations/live/operations.sqlite --mode live
Get-ScheduledTaskInfo -TaskName StockForecasting-Pilot-Run
Get-ScheduledTaskInfo -TaskName StockForecasting-Pilot-Health
```

Mở `log/operations/live/health/live/health.html` để xem báo cáo local; F5 để
cập nhật. `health.json` dùng cho máy đọc. `task_logs/` giữ stdout/stderr của
wrapper; `runs/daily_batch/<run-id>/` giữ log, summary, config và manifest.
`operations.sqlite` chứa jobs/leases/events/active deployment;
`forecasts.sqlite` chứa dự báo và nhãn append-only. Không sửa SQLite bằng tay.

Trạng thái `not_due` trước origin đầu là bình thường. `unhealthy` hoặc exit
khác 0 cần đọc alerts và failed run. `late_origins` lưu vi phạm thời hạn ngay
cả sau khi phục hồi. `collection_complete` nghĩa là đủ budget và nhãn, không
có nghĩa model đủ điều kiện promotion. RMSE skill = 1 − RMSE model / RMSE
baseline zero-return; số âm nghĩa là kém baseline, không phải lợi nhuận đầu tư.
Candidate log return dùng log(P_target/P_origin), khác phần trăm sinh lời đơn.

## Khôi phục và đổi release

### Nghiệm thu theo dữ liệu thực tế, cập nhật 23/11/2026

Hai khái niệm cần tách rõ: **chạy thành công** là chương trình xử lý xong;
**đúng hạn** là bản ghi bền vững xuất hiện trước deadline đã đặt. Chạy lại hôm
sau có thể khôi phục dữ liệu nhưng không biến lần giao trễ thành đúng hạn.

| Mỗi mã AAPL/MSFT | Số lượng qua origin 20/11 |
| --- | --- |
| Phiên cần có dự báo | 5 |
| Dự báo đã ghi | 4 |
| Thiếu | 1, origin 16/11 |
| Dự báo đúng hạn / muộn | 2 / 2; muộn ở 18/11 và 20/11 |
| Nhãn đã chấm / đang chờ | 3 / 1 |
| Nhãn đã chấm nhưng trễ | 2, ứng với origin 17/11 và 19/11 |

Tổng cộng có 8 dự báo, 6 nhãn. Nhãn của target 23/11 chưa đến hạn ở thời điểm
kiểm tra. Cửa sổ hiện tại có **0 phiên liên tiếp đáp ứng đủ cả dự báo và nhãn**.
Khoảng không có tác vụ chạy được xác minh từ log; chưa chứng minh nguyên nhân
là tắt máy, đăng xuất, sleep hay điều kiện nguồn điện của Windows.

```powershell
.\.venv\batch312-v3\Scripts\python.exe -I -m quant_forecast_lab.cli batch acceptance --state-db log/operations/live/operations.sqlite --mode live
```

Lệnh chỉ đọc, trả exit 1 khi `not_ready`, exit 2 khi không kiểm tra được, và
exit 0 khi `ready_for_operational_review`. Live giữ nguyên quy tắc 20 phiên;
không giảm `--sessions` sau khi thấy dữ liệu. Replay có thể dùng cửa sổ nhỏ để
kiểm thử, nhưng không được tính là bằng chứng prospective.

Mỗi phiên ghi riêng hạn dự báo, hạn nhãn, thời điểm đã nhận và lý do thất bại.
Phải đủ phiên liên tiếp cho **mọi mã**; thiếu/trễ/trùng hoặc nhãn quá hạn làm
đứt chuỗi. Nhãn mới chưa đến hạn được giữ `pending`; không điền actual giả.
Nhãn từng về trễ vẫn giữ `late_label` sau khi chấm xong. Lệnh còn kiểm tra
checksum model/input/inference của cửa sổ và các runtime blocker như heartbeat
stale, job hết lease hay batch lỗi. Chuỗi nguồn của nhãn được kiểm tra thêm
trong recovery bundle. Các lịch/điểm bắt đầu không trùng nhau phải được đánh
giá riêng, không âm thầm lấy giao các phiên để bỏ qua dữ liệu thiếu.

Health report phiên bản 2 giữ các cảnh báo lịch sử `late_forecasts` và
`late_labels`. Vì vậy health có thể còn đỏ vì sự cố cũ trong khi một cửa sổ
20 phiên mới đã đủ điều kiện xem xét; báo cáo nghiệm thu nêu rõ khoảng ngày.
Không xóa sự cố để đổi màu trạng thái. `ready_for_operational_review` không
chứng minh lợi thế dự báo, không tự promotion và không chứng nhận production.

### Backup phục vụ và diễn tập trên bản sao

`scripts/pilot_recovery.py` bổ sung bundle chứa hai database, các model,
protocol, đầu vào dự báo, nguồn nhãn từ score jobs, source runtime và lockfile.
Script từ chối job còn chạy hoặc database có commit mới trong lúc chụp.
Kiểm tra thay đổi dùng cùng các kết nối đọc theo
[SQLite data_version](https://www.sqlite.org/pragma.html#pragma_data_version).
Vẫn cần tạm dừng lịch để có cửa sổ chụp ổn định; script không tự quản lý OS task.

Sau khi Disable hai task và xác nhận không còn Running, chạy bằng runtime
đang active; dùng `try/finally` để Enable lại task kể cả khi lệnh thất bại:

```powershell
.\.venv\batch312-v3\Scripts\python.exe -I scripts/pilot_recovery.py --installed-package backup --state-db log/operations/live/operations.sqlite --mode live --output-root log/recovery_manual
# Lấy run_directory từ JSON trả về:
.\.venv\batch312-v3\Scripts\python.exe -I scripts/pilot_recovery.py --installed-package drill --backup <run_directory> --output-root log/recovery_manual
```

Diễn tập ngày 23/11 đã chụp release v2 trước khi nâng lên v3, khi hai task đã
tạm dừng. Bundle chứa **26 thư mục nguồn, 271 artifact**; bản diễn tập có
**254 artifact**, khôi phục đúng 8 dự báo/6 nhãn và lịch sử operations. Hai
phép inference offline từ model/input đã sao chép khớp bản gốc. Task đã bật lại.
Log và receipt nằm ở `log/validation_20261123/live_backup.json`,
`live_recovery_drill.json` và `log/recovery_20261123/`.

Đây là backup/restore local có kiểm chứng. Bundle giữ đường dẫn tuyệt đối
trong ledger, chưa tự kích hoạt hệ thống live trên máy/đường dẫn khác; không
chứa mọi dependency wheel hay cấu hình Task Scheduler. Chưa có backup offsite,
thử mất cả host, hay đo RTO/RPO cho hệ thống production. RTO là mục tiêu thời
gian phục hồi; RPO là lượng dữ liệu tối đa chấp nhận mất. Không ghi đè ledger
thật để diễn tập và không gọi inference offline là dự báo prospective mới.

### Các thao tác SQLite riêng lẻ

Lỗi nguồn dữ liệu tạm thời có retry giới hạn. Dữ liệu sai schema hoặc tương
lai bị chặn, không retry như lỗi mạng. Không backfill dự báo live cho phiên đã
qua hạn. Job cũ hết lease được cảnh báo; cần điều tra thay vì xóa lịch sử để xanh.

Trước backup phối hợp, tạm Disable cả hai task và chờ task đang chạy kết thúc.
Không Stop cưỡng bức hoặc ghi đè database hiện hành để thử recovery.

```powershell
Disable-ScheduledTask -TaskName StockForecasting-Pilot-Run
Disable-ScheduledTask -TaskName StockForecasting-Pilot-Health
# Sau khi xác nhận cả hai không còn Running; chọn đích mới mỗi lần:
.\.venv\batch312-v3\Scripts\python.exe -I -m quant_forecast_lab.cli batch backup --source log/operations/live/forecasts.sqlite --destination log/backups/manual/forecasts.sqlite
.\.venv\batch312-v3\Scripts\python.exe -I -m quant_forecast_lab.cli batch backup --source log/operations/live/operations.sqlite --destination log/backups/manual/operations.sqlite
.\.venv\batch312-v3\Scripts\python.exe -I -m quant_forecast_lab.cli batch restore --backup log/backups/manual/forecasts.sqlite --destination log/recovery/manual/forecasts.sqlite
Enable-ScheduledTask -TaskName StockForecasting-Pilot-Health
Enable-ScheduledTask -TaskName StockForecasting-Pilot-Run
```

SQLite backup API cho snapshot nhất quán từng database; hai DB và artifact
không tạo thành một transaction chung. Lưu cả config/protocol/model/snapshot,
lockfile và release đang cài ở nơi dự phòng. Bản backup chứa đường dẫn cũ nên
restore sang máy/đường dẫn khác còn cần cấu hình mới và xác minh liên kết.
Diễn tập đã chứng minh thêm phục hồi artifact phục vụ và inference offline;
chưa là disaster recovery toàn hệ thống hoặc backup offsite tự động.

`batch activate --config FILE --reason TEXT` chỉ chuyển pointer sau xác minh;
`batch rollback --state-db FILE --mode live --reason TEXT` trở về cấu hình
trước đã xác minh. Release trước v3 là v2, giữ tại
`log/operations/live/config-release-v2.json` và `.venv/batch312-v2`.
Rollback khác code hash cần chọn đúng runtime cũ trước;
đồng thời tạm dừng và đổi Python path trong **cả hai task** rồi kiểm tra exit code.
Lệnh rollback không
tự phục hồi executable, dependency, artifact hay OS task. Không sửa ID cũ.

Đăng ký lại trên máy mới: chạy `scripts/register_batch_tasks.ps1` bằng
`powershell -NoProfile -ExecutionPolicy Bypass -File ...`, truyền `-Python`,
`-StateDatabase`, `-WorkingDirectory` tuyệt đối; xem plan mặc định, thêm
`-Register` khi muốn tạo task. Script từ chối ghi đè task đã tồn tại.

## Điều kiện qua giai đoạn pilot

Theo dõi **20 phiên hợp lệ liên tiếp trong tương lai**: mỗi asset có đúng một
dự báo trước deadline, nhãn về đúng hạn, checksum hợp lệ, không duplicate,
health/task exit không có lỗi chưa xử lý. Nếu bỏ lỡ phiên, ghi incident và bắt
đầu lại cửa sổ nghiệm thu sau khi sửa nguyên nhân; giữ nguyên dữ liệu cũ.
Thử phục hồi trên bản sao, ghi thời gian khôi phục và người xử lý. Kiểm tra dung
lượng/retention log hàng tuần; không xóa run đang là nguồn của ledger/protocol.

20 phiên là điều kiện vận hành, không phải cỡ mẫu xác nhận lợi thế ML.
Budget 252 và ngưỡng RMSE 1% hiện chưa được hiệu chỉnh bằng power analysis.
Chỉ mở quyết định promotion khi có đánh giá prospective đủ lực, so sánh paired
với baseline, bất định/multiple testing và chi phí nếu nghiên cứu trading.
Trước production còn cần host chạy liên tục, giám sát ngoài máy, nguồn dữ liệu
được cấp phép, backup offsite và diễn tập recovery toàn hệ thống. Docker daemon
local đang không chạy; chưa có bằng chứng container build/run.

CV phù hợp hiện tại: “Built a reproducible time-series research and prospective
batch inference pipeline with immutable forecast records, resumable jobs,
calendar-aware scheduling, independent health checks and verified SQLite
recovery.” Chỉ thêm số liệu sau khi đo; chưa ghi profitable/production-ready.

Tài liệu gốc: [SQLite backup API](https://docs.python.org/3.12/library/sqlite3.html#sqlite3.Connection.backup),
[uv locking/sync](https://docs.astral.sh/uv/concepts/projects/sync/) và
[PowerShell execution policies](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-5.1).
Cache key cho wheel local được mở rộng theo
[uv caching](https://docs.astral.sh/uv/concepts/cache/#dynamic-metadata);
demo `--installed-package` kiểm tra source hash để chặn dùng nhầm build cũ.
