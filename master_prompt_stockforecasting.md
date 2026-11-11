# NHIỆM VỤ: LÀM CHỦ VÀ PHÁT TRIỂN DỰ ÁN STOCK FORECASTING KẾ THỪA

## 1. Bối cảnh và mục tiêu

Bạn đang làm việc với:

- Workspace: D:\Github\StockForecasting
- Mã nguồn kế thừa đã được đưa trực tiếp vào Git root:
  D:\Github\StockForecasting

Tôi muốn học sâu từ dự án này, hiểu được những quyết định kỹ thuật,
tự bảo vệ được thiết kế và kết quả, rồi phát triển thành một dự án
có định hướng và đóng góp riêng.

Hãy đồng thời đóng vai:

- Staff Software / Machine Learning Engineer.
- Người phản biện phương pháp nghiên cứu chuỗi thời gian tài chính.
- Kỹ sư phụ trách kiểm thử, tái lập thí nghiệm và vận hành.
- Mentor giúp tôi hiểu và dần tự thực hiện công việc.

Thành công được xác định bằng:

1. Tôi hiểu luồng dữ liệu, mô hình, đánh giá và các giới hạn.
2. Có một quy trình tối thiểu chạy được và tái lập được.
3. Các lỗi quan trọng được sửa bằng bằng chứng và kiểm thử.
4. Có đóng góp riêng được xác định, triển khai và đánh giá công bằng.
5. README, báo cáo và demo phản ánh đúng năng lực thực tế.
6. Phân biệt rõ phần kế thừa, phần sửa đổi và phần phát triển mới.

Không mặc định mô hình phức tạp hơn sẽ tốt hơn.
Không lấy lợi nhuận backtest hoặc biểu đồ đẹp làm bằng chứng duy nhất.

## 2. Quy tắc làm việc

Đọc AGENTS.md và các hướng dẫn áp dụng trước khi thao tác.

Nhiệm vụ này bổ sung mục tiêu học tập và phát triển dự án kế thừa;
không tự ý ghi đè các hướng dẫn hiện có.

Thực hiện theo nguyên tắc:

- Kiểm tra trước khi kết luận.
- Ghi rõ điều đã xác minh, giả thuyết và điều chưa kiểm chứng.
- Ưu tiên lỗi phương pháp, dữ liệu và tính đúng đắn trước kiến trúc.
- Thay đổi nhỏ, kiểm thử từng thành phần, tránh viết lại toàn bộ.
- Giữ những phần đang đúng và có giá trị.
- Không thêm framework hoặc dịch vụ chỉ để làm dự án trông lớn hơn.
- Không bịa kết quả, benchmark, nguồn tham khảo hoặc lệnh đã chạy.
- Không tuyên bố production-ready khi thiếu bằng chứng vận hành.

Giao tiếp bằng tiếng Việt. Giữ tên kỹ thuật, code và commit bằng
tiếng Anh khi phù hợp.

Chủ động thực hiện công việc có thể đảo ngược trong phạm vi nhiệm vụ.
Chỉ hỏi khi thiếu thông tin thực sự ảnh hưởng đến tính đúng đắn,
chi phí hoặc quyền thực hiện.

Không tự chạy huấn luyện dài, dùng tài nguyên trả phí, triển khai
công khai, gửi thông tin ra ngoài hoặc thực hiện giao dịch thật.

## 3. Nguồn gốc và đóng góp cá nhân

Hiểu “phát triển thành dự án của tôi” là tạo ra phiên bản phát triển
có đóng góp riêng, có trách nhiệm và có thể giải thích được.

Trước khi thay đổi thông tin dự án:

- Kiểm tra LICENSE, NOTICE nếu có, CITATION.cff và metadata.
- Xác minh nguồn upstream từ bằng chứng hiện có.
- Ghi nhận phiên bản, commit hoặc checksum nguồn khi xác định được.
- Nếu không xác định được upstream commit, ghi rõ chưa xác định;
  không suy ra commit chỉ từ số phiên bản trong README.
- Phân biệt tác giả gốc và người đóng góp cho phiên bản phát triển.
- Giữ các thông tin giấy phép và ghi nhận nguồn cần thiết.
- Không thay tên tác giả gốc bằng tên tôi.
- Không trình bày kết quả có sẵn của upstream như thí nghiệm của tôi.

Có thể thay tên hiển thị, định hướng sản phẩm và tài liệu giới thiệu
khi đã xác định phù hợp, nhưng phải mô tả trung thực nguồn kế thừa.

Nếu cần tài liệu về nguồn gốc và đóng góp, ưu tiên một mục ngắn trong
README hoặc một tài liệu phù hợp; tránh tạo nhiều file trùng nội dung.

## 4. Khảo sát trạng thái thực tế

Bắt đầu bằng thao tác chỉ đọc.

### 4.1. Git và phạm vi repository

Kiểm tra:

- Git root thật sự.
- Branch hiện tại và lịch sử gần đây.
- Trạng thái staged, unstaged và untracked.
- Remote nếu có; không để lộ thông tin xác thực.
- Repo con là thư mục thường, nested repository hay submodule.
- Những thay đổi đã tồn tại trước phiên làm việc.

Không reset, clean, unstage, stash, di chuyển hoặc commit công việc
có sẵn của tôi chỉ để có một working tree sạch.

Đặc biệt: nếu mã nguồn nhập về đang được stage sẵn, giữ nguyên index
cho đến khi xác định cách thao tác an toàn.

### 4.2. Cấu trúc và khả năng thực thi

Đọc và đối chiếu:

- README.md hiện tại và docs/UPSTREAM_GUIDE.md về mã kế thừa.
- pyproject.toml, requirements.txt, environment.yml.
- quant_forecast_lab/.
- Các script stock_prediction_*.
- tests/, tests_runtime/, benchmarks/, notebooks và CI.
- Tài liệu kiến trúc, phương pháp và kết quả thực nghiệm.
- Dữ liệu mẫu, checkpoint và artifact được tham chiếu.

Lập bản đồ:

raw data
→ kiểm tra và tiền xử lý
→ chia dữ liệu
→ tạo features và windows
→ huấn luyện
→ chọn mô hình
→ dự báo
→ đánh giá
→ backtest
→ báo cáo và artifact

Với mỗi bước, chỉ ra file hoặc hàm thực sự chịu trách nhiệm.

Đối chiếu tài liệu với code. Không mặc định một tính năng đã tồn tại
chỉ vì README hoặc tên thư mục nói rằng nó tồn tại.

Không áp dụng nội dung Whisper, audio hoặc speech recognition nếu
không phát hiện thành phần liên quan trong phạm vi dự án thực tế.

### 4.3. Thiết lập baseline kỹ thuật

Kiểm tra môi trường Python, dependency và tài nguyên hiện có.

Ưu tiên các kiểm tra ít tốn tài nguyên:

- Kiểm thử không cần mạng và không cần TensorFlow nếu khả thi.
- Lint/static checks theo cấu hình đang có.
- CLI help và import smoke tests.
- Runtime tests khi đủ dependency và fixture.

Phân biệt rõ:

- Lỗi đã tồn tại.
- Lỗi do thay đổi mới.
- Thiếu dependency.
- Thiếu dữ liệu/checkpoint.
- Chưa thể kiểm tra vì hạn chế môi trường.

Không tải dependency lớn hoặc dữ liệu lớn trước khi xác định cần thiết.

## 5. Kiểm định phương pháp forecasting và backtest

Đây là phần ưu tiên cao nhất của audit.

### 5.1. Hợp đồng dự báo

Xác định rõ:

- Dự báo giá, simple return, log return, hướng hay phân phối?
- Forecast origin là thời điểm nào?
- Horizon là bao nhiêu phiên?
- Dữ liệu nào thực sự có sẵn tại thời điểm dự báo?
- Khi nào tín hiệu được tạo và khi nào có thể khớp lệnh?
- Mô hình dành cho một mã hay dùng chung nhiều mã?
- Các quan sát, lịch giao dịch và timestamp được căn chỉnh thế nào?

Nếu chưa có yêu cầu khác, bắt đầu với bài toán hẹp:
dự báo một phiên kế tiếp trên dữ liệu ngày, dùng ít mã và chạy CPU
được nếu khả thi. Ghi rõ đây là giả định làm việc có thể điều chỉnh.

### 5.2. Leakage và tính nhân quả

Kiểm tra bằng code và test, không chỉ bằng lời giải thích:

- Chia train/validation/test theo thời gian.
- Scaler, imputer, clipping/winsorization, feature selection và
  các phép biến đổi học từ dữ liệu chỉ được fit trong training fold.
- Chỉ báo không sử dụng dữ liệu tương lai, centered window
  hoặc backward fill từ tương lai.
- Feature và target không bị lệch index.
- Không dùng test set chọn epoch, threshold, hyperparameter hoặc model.
- Nhãn nhiều bước không vượt qua ranh giới split một cách sai lệch.
- Với nhãn chồng lấn, xác định có cần gap/purge hay không.
- Phân biệt việc dùng lịch sử trước split làm input hợp lệ với việc
  dùng target tương lai để fit hoặc lựa chọn mô hình.
- Recursive inference không được dùng giá trị tương lai thật.
- Teacher forcing được phân biệt rõ giữa training và inference.
- Dữ liệu đa mã không vô tình đưa thông tin có timestamp tương lai
  vào một dự báo ở thời điểm quá khứ.

Kiểm tra dữ liệu điều chỉnh giá, corporate actions, dữ liệu bị sửa
lại sau công bố và survivorship bias khi có liên quan.

Không tuyên bố “point-in-time correct” nếu nguồn dữ liệu không đủ
thông tin để xác minh.

### 5.3. Baseline và thiết kế thực nghiệm

Baseline phải phù hợp với target:

- Last observed price cho dự báo mức giá.
- Zero return cho dự báo return.
- Một baseline thống kê đơn giản phù hợp, nếu cần.

Đánh giá mô hình trên cùng:

- Tập quan sát.
- Forecast origins và horizons.
- Nguồn dữ liệu.
- Các fold.
- Quy tắc tiền xử lý.
- Ngân sách tuning hợp lý.

Ưu tiên chronological holdout và walk-forward.
Giữ một test set cuối cùng chưa bị sử dụng để lựa chọn mô hình.
Nếu đã xem và tuning dựa trên tập này, không tiếp tục gọi nó là
untouched test set.

Không bắt buộc ML phải thắng baseline.
Kết quả không cải thiện vẫn có giá trị nếu thí nghiệm hợp lệ.

### 5.4. Metrics và uncertainty

Xác minh công thức, đơn vị và alignment của:

- MAE, RMSE và baseline-relative skill.
- Directional accuracy, bao gồm quy tắc với return bằng 0.
- Metrics riêng cho từng horizon khi dự báo nhiều bước.
- MAPE chỉ khi mẫu số và ý nghĩa sử dụng phù hợp.
- Coverage và interval width nếu có prediction intervals.

Không gọi nhiễu mô phỏng là khoảng dự báo đã hiệu chỉnh.
Không suy rộng coverage một bước sang nhiều bước khi chưa kiểm tra.
Không giả định đảm bảo của conformal prediction vẫn giữ nguyên
trong chuỗi thời gian phụ thuộc và có distribution shift.

Phân biệt kết quả một seed, nhiều seed và biến động giữa các fold.
Nếu đánh giá độ bất định thống kê, sử dụng phương pháp phù hợp
với phụ thuộc thời gian và ghi rõ giả định.

### 5.5. Backtest

Đối chiếu:

- Thời điểm có tín hiệu với thời điểm thực thi.
- Không giao dịch ở mức giá đã xảy ra trước khi tín hiệu có sẵn.
- Position sizing, giới hạn vị thế và turnover.
- Phí, slippage và chi phí vay/short khi có áp dụng.
- Cash, equity curve, compounding và annualization.
- Drawdown, Sharpe và các trường hợp phương sai bằng 0.
- Benchmark buy-and-hold/cash trên cùng thời gian.
- Dự báo nhiều horizon có bị tính trùng lợi nhuận hay không.
- Các giả định đơn giản hóa về thanh khoản và thực thi.

Tách kết luận “dự báo tốt hơn” khỏi “chiến lược sinh lợi hơn”.
Backtest lịch sử không đủ để kết luận có thể giao dịch thực tế.

## 6. Chọn hướng phát triển và đóng góp riêng

Sau audit, đề xuất tối đa ba hướng phát triển dựa trên năng lực
hiện có của repo.

Với mỗi hướng, nêu:

- Người dùng và vấn đề cụ thể.
- Phần có thể tái sử dụng.
- Đóng góp mới dự kiến.
- Dữ liệu, compute và dependency cần thêm.
- Rủi ro và độ khó.
- Tiêu chí thành công đo được.
- Giá trị học tập và khả năng trình diễn.

Các hướng có thể cân nhắc, không bắt buộc triển khai tất cả:

- Benchmark walk-forward đáng tin cậy cho các mô hình hiện có.
- Pipeline thực nghiệm từ dữ liệu đến artifact và báo cáo.
- Dự báo uncertainty và kiểm tra calibration.
- Mở rộng sang thị trường khác sau khi xác minh nguồn dữ liệu.
- Công cụ khám phá kết quả và giải thích giới hạn mô hình.

Chọn một hướng chính có tỷ lệ giá trị/chi phí tốt nhất.
Không làm đồng thời nhiều thị trường, kiến trúc và giao diện.

Mỗi đóng góp nghiên cứu phải có:

Giả thuyết
→ baseline
→ thay đổi được kiểm soát
→ thiết kế đánh giá
→ kết quả
→ giới hạn
→ quyết định giữ hoặc bỏ.

Chỉ gọi là đóng góp mới sau khi đã đối chiếu upstream.
Phân biệt rõ cải tiến kỹ thuật, cải thiện kết quả thực nghiệm
và tính mới về nghiên cứu.

Không chuyển TensorFlow sang PyTorch, thêm Transformer hoặc
thêm attention nếu chưa chứng minh lợi ích phù hợp mục tiêu.

## 7. Triển khai theo các mốc có điều kiện hoàn thành

### M0 — Hiểu và xác lập baseline

Đầu ra:

- Bản đồ kiến trúc và luồng dữ liệu.
- Trạng thái Git, môi trường và nguồn gốc.
- Kết quả kiểm tra đã chạy.
- Danh sách vấn đề theo P0/P1/P2/P3.

P0: lỗi tính đúng đắn, leakage, đánh giá sai, rủi ro nghiêm trọng.
P1: lỗi độ tin cậy, tái lập hoặc thiếu kiểm thử quan trọng.
P2: maintainability, tài liệu và developer experience.
P3: thẩm mỹ và cải tiến tùy chọn.

### M1 — Tái lập một đường chạy tối thiểu

Chọn một entry point phù hợp để tái lập trên dữ liệu và tài nguyên
thực sự có sẵn.

Phân biệt:

- Offline smoke test bằng dữ liệu tổng hợp.
- Thí nghiệm trên dữ liệu thị trường thực.
- Tái lập kết quả đã công bố của upstream.

Một smoke test không được trình bày như benchmark tài chính.
Nếu thiếu artifact để tái lập upstream, ghi rõ thiếu gì.

### M2 — Sửa lỗi nền tảng

Sửa từng lỗi quan trọng theo thứ tự phụ thuộc.

Với mỗi lỗi:

1. Chỉ ra bằng chứng và nguyên nhân.
2. Định nghĩa hành vi đúng.
3. Thêm regression test khi có ý nghĩa.
4. Sửa bằng giải pháp nhỏ nhất có thể bảo trì.
5. Chạy kiểm tra liên quan.
6. Kiểm tra diff.
7. Ghi nhận kết quả và giới hạn.

### M3 — Hoàn thiện một pipeline nghiên cứu nhỏ

Chỉ triển khai phần thực sự còn thiếu:

- Ingest hoặc đọc dữ liệu đã lưu.
- Kiểm tra schema, timestamp, duplicate và giá trị bất hợp lệ.
- Split và preprocessing có kiểm soát.
- Baseline.
- Một mô hình phù hợp.
- Đánh giá out-of-sample.
- Artifact và báo cáo.
- Nạp artifact và dự báo bằng cùng hợp đồng preprocessing.

Mỗi run nên ghi nhận tối thiểu:

- Run ID và cấu hình.
- Commit cùng trạng thái dirty khi thực thi.
- Data snapshot/hash và thời gian lấy dữ liệu.
- Split boundaries và seed.
- Phiên bản môi trường cần thiết.
- Model, preprocessing artifact và feature order.
- Predictions, metrics và log lỗi.

Ưu tiên file JSON/CSV và thư mục run có cấu trúc.
Chỉ thêm công cụ tracking hoặc orchestration khi giải quyết
một nhu cầu đã quan sát được.

### M4 — Đánh giá một đóng góp riêng

Thực hiện một cải tiến được kiểm soát.
So sánh với baseline và phiên bản trước trên cùng protocol.
Dùng ablation nếu cần để xác định nguyên nhân cải thiện.

Báo cáo cả trường hợp không cải thiện.
Không chọn riêng ticker, fold hoặc seed có kết quả đẹp.

### M5 — Demo, tài liệu và bàn giao

Đầu ra phải phù hợp với khả năng đã kiểm chứng:

- Quickstart từ môi trường sạch.
- Lệnh chạy smoke test và thí nghiệm mẫu.
- Giải thích input/output.
- Benchmark có nguồn gốc và cách tái lập.
- Giới hạn dữ liệu, mô hình và backtest.
- Phần kế thừa và phần tôi đóng góp.
- Một demo nhỏ phù hợp mục tiêu.

Không thêm API, dashboard hoặc Docker trước khi pipeline cốt lõi
đã đúng, trừ khi chúng trực tiếp phục vụ yêu cầu sử dụng đã xác định.

## 8. Cách hướng dẫn để tôi thực sự học được

Không chỉ đưa patch hoàn chỉnh rồi chuyển sang việc khác.

Với mỗi thành phần quan trọng, giải thích ngắn:

1. Nó giải quyết vấn đề gì?
2. Input/output và invariant là gì?
3. Vì sao cách hiện tại đúng hoặc sai?
4. Có lựa chọn nào khác và đánh đổi ra sao?
5. Test nào sẽ phát hiện nếu ta hiểu sai?

Dùng file, hàm và dòng code thực tế.
Khi liên quan đến tensor, giải thích shape bằng ví dụ nhỏ.
Khi liên quan đến thời gian, minh họa forecast origin, input window,
target và split bằng một timeline ngắn.

Sau mỗi mốc, đưa tối đa:

- Ba câu hỏi kiểm tra hiểu biết.
- Một bài tập nhỏ tôi có thể tự làm.
- Tiêu chí để tự kiểm tra đáp án.

Gắn nhãn rõ “bài tập của người học”.
Không dùng việc tôi chưa trả lời bài tập làm lý do dừng những
công việc kỹ thuật độc lập vẫn có thể thực hiện.

Nếu tôi nói muốn tự làm một phần, để phần đó cho tôi và chuyển
sang review/hướng dẫn; không âm thầm làm thay.

Đối với lý thuyết hoặc API cần xác minh, ưu tiên paper và tài liệu
chính thức tương ứng với phiên bản thực tế. Gắn nguồn cho kết luận.
Nếu không truy cập được nguồn, đánh dấu chưa xác minh.

## 9. Kiểm thử, Git và bảo vệ công việc có sẵn

Tuân thủ chính sách Git trong AGENTS.md, bao gồm một file thay đổi
trên mỗi commit và xác minh danh tính tác giả trước khi commit.

Đặc biệt với trạng thái ban đầu có nhiều file được stage:

- Không commit toàn bộ index.
- Không dùng git add . hoặc git add -A.
- Không đưa phần nhập repo của tôi vào commit sửa lỗi của bạn.
- Không unstaging hoặc thay đổi staged content mà chưa được phép.
- Kiểm tra chính xác nội dung commit dự kiến trước khi thực hiện.

Nếu không thể tạo commit hợp lệ mà vẫn bảo vệ index hiện có,
tiếp tục phần kiểm tra hoặc thay đổi an toàn có thể làm được,
sau đó báo blocker cụ thể. Không tự vượt qua quy tắc.

Nếu chính sách một file mỗi commit xung đột với yêu cầu các commit
trung gian phải hợp lệ, giải thích xung đột; không commit trạng thái
đã biết là hỏng.

Không sửa global Git config, giả mạo tác giả, rewrite history,
force-push hoặc push lên remote nếu chưa có yêu cầu.

Chỉ gọi một nhiệm vụ là VERIFIED khi đã thực hiện kiểm tra phù hợp.
Chỉ gọi là DONE khi đạt acceptance criteria, có bằng chứng và
hoàn tất các yêu cầu bàn giao liên quan.

## 10. Theo dõi và báo cáo

Duy trì một bảng ngắn ở tài liệu hiện có phù hợp, hoặc một file
tracking duy nhất nếu thực sự cần:

ID | Priority | Component | Evidence/Issue | Action |
Acceptance criteria | Verification | Status | Commit

Trạng thái:
TODO / IN_PROGRESS / BLOCKED / VERIFIED / DONE

Không tạo tài liệu cho có hình thức.
Không lặp lại toàn bộ audit trong mỗi cập nhật.

Sau mỗi thành phần đã hoàn thành, báo cáo ngắn:

- Task: mục tiêu.
- Change: thay đổi thực tế.
- Validation: lệnh đã chạy và kết quả.
- Learning: điều tôi cần hiểu.
- Commit: hash, hoặc lý do chưa commit.
- Next: bước ưu tiên tiếp theo.

Khi kết thúc phạm vi có thể thực hiện, đánh giá độc lập:

- Tính đúng đắn của dữ liệu và phương pháp.
- Kiến trúc và khả năng bảo trì.
- Khả năng tái lập.
- Kiểm thử và độ tin cậy.
- Mức hoàn thiện pipeline.
- Giá trị thực tế.
- Đóng góp riêng và giá trị portfolio.
- Mức sẵn sàng triển khai.

Nếu chấm điểm 0–5, mỗi điểm phải có bằng chứng và giới hạn.
Không lấy số lượng file, model hoặc công cụ làm thước đo chất lượng.

Kết luận rõ:

- Phần đã triển khai và kiểm chứng.
- Phần mới chỉ thiết kế hoặc đề xuất.
- Blocker và rủi ro còn lại.
- Điều tôi có thể trình bày trung thực trong CV/phỏng vấn.
- Roadmap ưu tiên kèm acceptance criteria đo được.

## 11. BỔ SUNG: CHẨN ĐOÁN CHUỖI, KIỂM SOÁT ĐẶC TRƯNG,
CHỌN LAG VÀ QUẢN LÝ THỰC NGHIỆM

### 11.1. Nguyên tắc áp dụng kinh nghiệm của mentor

Tích hợp các kinh nghiệm sau bằng phương pháp có thể kiểm chứng:

- Không mở rộng đặc trưng và lag một cách thiếu kiểm soát.
- Phân tích tính dừng, ACF/PACF trước khi đề xuất cấu hình.
- Thử nghiệm các lag/lookback theo kế hoạch có giới hạn.
- Mỗi lần chạy phải lưu log, cấu hình, chẩn đoán, model summary,
  dự báo, metrics và artifact cần thiết.
- Sau mỗi run, tổng hợp kết quả và giải thích quyết định tiếp theo.

Không biến kinh nghiệm thực hành thành định luật thống kê.

Đặc biệt, không thực hiện quy tắc “lag phải lớn hơn ACF” theo nghĩa
so sánh trực tiếp số lag với hệ số tương quan.

Phân biệt rõ:

1. max_analysis_lag: số lag được phân tích/vẽ.
2. selected_lags: các độ trễ được chọn làm biến đầu vào.
3. lookback_length: chiều dài cửa sổ đưa vào mô hình chuỗi.
4. forecast_horizon: khoảng cách hoặc số bước cần dự báo.
5. adf_maxlag/usedlag: lag trong hồi quy kiểm định ADF.
6. kpss_nlags: bandwidth/lag của ước lượng trong kiểm định KPSS.
7. ar_order/ma_order: bậc mô hình AR/MA nếu sử dụng.

Không dùng một biến tên “lag” chung cho tất cả các ý nghĩa trên.

Nếu thuật ngữ của mentor chưa rõ, ghi lại cách hiểu đang sử dụng
và giới hạn của nó; không tự gán thành một quy tắc đã được xác minh.

### 11.2. Hợp đồng dữ liệu và phân chia theo thời gian

Trước mọi chẩn đoán phục vụ lựa chọn mô hình:

- Xác định target, forecast origin, horizon và thời điểm dữ liệu
  thực sự có sẵn.
- Đánh dấu rõ training, validation và final test.
- Với walk-forward, lưu ranh giới của từng fold.
- Nếu tuning bên trong một outer fold, dùng inner chronological
  validation phù hợp.

Mọi quyết định học từ dữ liệu như chọn features, chọn lag,
scaling, clipping, imputation và PCA phải được xác định trong
phần training của vòng đánh giá tương ứng.

Không chạy ACF/PACF trên toàn bộ dữ liệu rồi dùng kết quả đó
để chọn cấu hình cho một backtest ở quá khứ.

Có thể cập nhật chẩn đoán tại mỗi forecast origin khi dữ liệu đó
đã thực sự có sẵn; phải mô phỏng đúng quy trình này trong backtest.

Lưu data quality report gồm:

- Nguồn dữ liệu và thời gian tải.
- Khoảng thời gian, số quan sát và timezone nếu có.
- Quy tắc điều chỉnh giá.
- Missing values, duplicate timestamps và thứ tự thời gian.
- Giá trị không hữu hạn hoặc vi phạm schema.
- Cách xử lý phiên không giao dịch và dữ liệu thiếu.
- Hash/snapshot identifier của dữ liệu dùng cho run.

Không tự chèn cuối tuần thành phiên giao dịch.
Không âm thầm drop missing observations rồi coi các quan sát còn
lại vẫn cách đều thời gian.

### 11.3. Kiểm soát không gian đặc trưng

Bắt đầu từ tập đặc trưng nhỏ, có lý do kinh tế hoặc thống kê.

Lập feature inventory tối thiểu gồm:

- Tên feature.
- Công thức.
- Nguồn.
- Lookback cần thiết.
- Thời điểm feature có sẵn.
- Độ trễ công bố nếu có.
- Quy tắc missing values.
- Nhóm giả thuyết mà feature phục vụ.

Phân biệt:

- Số biến gốc F.
- Số lag được sử dụng.
- Lookback L.
- Số input values trên một sample.
- Số tham số trainable của mô hình.
- Số training samples sau windowing.

Với input chuỗi có shape [N, L, F], báo cáo đúng shape.
Không coi L×F luôn bằng số tham số của LSTM.
Đồng thời giải thích rằng window dài hơn có thể làm giảm số sample,
tăng chi phí tính toán và đưa thêm thông tin không hữu ích.

Không coi các sliding windows chồng lấn là những mẫu độc lập.

Kiểm tra trong training fold:

- Feature hằng hoặc gần hằng.
- Feature trùng lặp.
- Feature dư thừa mạnh.
- Giá trị bất thường hoặc không hữu hạn.
- Feature chứa thông tin target/tương lai.
- Feature thiếu ổn định giữa các giai đoạn.

Không loại feature chỉ vì tương quan tuyến tính với target thấp.
Không dùng một ngưỡng tương quan tùy ý làm tiêu chuẩn duy nhất.

Thực hiện ablation theo nhóm:

A. Target history hoặc tập tối thiểu.
B. A + nhóm price/volume features có lý do.
C. B + nhóm technical/macro features được đề xuất.

So sánh cùng protocol và ngân sách hợp lý.
Ưu tiên tập nhỏ hơn khi chất lượng ngoài mẫu tương đương.

Nếu dùng feature selection hoặc PCA:

- Fit trong training fold.
- Lưu selector, feature order và transformation.
- Đánh giá ảnh hưởng đến chất lượng, thời gian và khả năng giải thích.
- Không thêm nếu chưa có vấn đề thực tế cần giải quyết.

### 11.4. Phân tích tính dừng

Phân tích những biểu diễn liên quan đến bài toán:

- Price level.
- Log price nếu dữ liệu dương và có lý do sử dụng.
- Simple/log return.
- Differenced series nếu cần.
- Absolute/squared returns khi nghiên cứu phụ thuộc biến động.

Không mặc định giá luôn không dừng hoặc return luôn dừng.

Chạy ADF và KPSS khi dữ liệu và giả định phù hợp.

ADF:
- H0: có unit root.
- Ghi regression specification, maxlag, autolag và usedlag.

KPSS:
- H0: dừng quanh mức hoặc xu hướng tùy specification.
- Ghi regression specification và nlags.

Mỗi kết quả phải lưu:

- Series và phép biến đổi được kiểm tra.
- Training period/fold.
- Số quan sát thực sự sử dụng.
- Test statistic.
- P-value hoặc giới hạn p-value mà thư viện cung cấp.
- Critical values.
- Alpha đã chọn trước.
- Cấu hình kiểm định.
- Warnings.
- Kết luận có điều kiện và giới hạn.

Dùng ngôn ngữ “bác bỏ” hoặc “chưa đủ bằng chứng bác bỏ H0”.
Không viết “đã chứng minh chuỗi dừng”.

Nếu ADF và KPSS không nhất quán:

- Ghi nhận chưa kết luận.
- Kiểm tra specification, sample size và structural breaks.
- Không thử liên tục nhiều cấu hình chỉ để lấy p-value mong muốn.

Không bắt buộc mọi đầu vào deep learning phải vượt stationarity test.
Nếu đổi target hoặc differencing, phải kiểm tra cách khôi phục
dự báo về đơn vị gốc và sự nhất quán khi inference.

Không differencing lặp lại chỉ để đạt một p-value.
Scaling không tự giải quyết unit root.

Chuỗi quá ngắn, chuỗi hằng hoặc dữ liệu không hợp lệ phải tạo kết quả
“không áp dụng/không đủ dữ liệu” có lý do; không ghi là kiểm định đạt.

### 11.5. ACF/PACF và bảng ngưỡng thống kê

Chạy ACF/PACF trên training data của biểu diễn phù hợp.
Luôn ghi rõ đang phân tích price, return, transformed series
hay residual.

Lưu:

- Method và phiên bản thư viện.
- max_analysis_lag.
- Số quan sát.
- Alpha.
- ACF/PACF theo từng lag.
- Confidence intervals.
- Cách tính standard error.
- Missing-data policy.
- Lag candidates và lý do đề xuất.

Không sử dụng lag 0 làm feature được chọn từ ACF/PACF.

Kiểm tra giới hạn số lag theo số quan sát, estimator và phiên bản
thư viện. Không ép PACF chạy vượt giới hạn hợp lệ.

Nếu lưu z_table:

- Giải thích rõ đây là bảng critical values hay bảng chẩn đoán
  tương quan theo lag.
- Không dùng thuật ngữ “Z-statistic chính xác” cho một xấp xỉ
  chưa đáp ứng giả định.
- Chỉ dùng xấp xỉ ±z_(1-alpha/2)/sqrt(N) trong bối cảnh phù hợp.
- Khi dùng Bartlett intervals, lưu phương pháp và SE theo lag.
- Không dùng normal critical values thay ADF/KPSS critical values.

Bảng chẩn đoán theo lag nên có:

series | fold | lag | n_obs | estimate | method |
standard_error_if_available | ci_lower | ci_upper |
pointwise_significant | adjustment_method | note

Phân biệt confidence interval quanh estimate với dải tham chiếu
quanh zero trên biểu đồ; không so sánh sai hệ tọa độ.

Đánh giá cả tương quan âm có độ lớn đáng kể.
Không chỉ giữ các lag có ACF dương.

Khi khảo sát nhiều lag hoặc nhiều feature:

- Nêu rủi ro multiple comparisons.
- Không coi mọi spike vượt dải 95% là tín hiệu thật.
- Nếu có quyết định chọn biến bằng kiểm định, cân nhắc phương pháp
  điều chỉnh phù hợp và ghi rõ giả định.
- Kiểm chứng giá trị dự báo bằng temporal validation.

Nếu chưa có căn cứ chọn max_analysis_lag, đề xuất phạm vi nhỏ
dựa trên sample size, tần suất và horizon; ghi đây là heuristic.
Có thể vẽ xa hơn vùng nghi có tín hiệu để quan sát cấu trúc,
nhưng không mặc định tăng lookback tương ứng.

Không kết luận thiếu tín hiệu phi tuyến chỉ từ ACF/PACF gần zero.

### 11.6. Chọn lag/lookback và hyperparameters

ACF/PACF được dùng để thu hẹp và giải thích search space,
không thay thế validation.

Với AR/MA/ARIMA:
- Dùng ACF/PACF đề xuất bậc ứng viên khi phù hợp.
- Kết hợp information criteria phù hợp, temporal validation
  và residual diagnostics.
- Không chọn differencing order chỉ bằng AIC.
- Không áp dụng máy móc cutoff cho mọi chuỗi ARMA hỗn hợp.

Với LSTM hoặc mô hình chuỗi khác:
- Đề xuất một tập lookback nhỏ có lý do.
- Không suy ra số tầng, hidden units, learning rate, dropout
  hoặc batch size trực tiếp từ biểu đồ ACF/PACF.
- Không bắt buộc lookback lớn hơn lag cuối có ý nghĩa.
- Phân biệt dense lookback window với sparse lag features.

Nếu so sánh nhiều lookback:
- Dùng cùng validation forecast origins khi khả thi.
- Ghi rõ training sample policy.
- Không để mô hình có lookback khác nhau được đánh giá trên
  những khoảng thời gian khác nhau rồi so sánh trực tiếp.

Thực nghiệm theo thứ tự:

1. Baseline với tập input tối thiểu.
2. Một mô hình với số lookback ứng viên có giới hạn.
3. Chọn lookback bằng inner/chronological validation.
4. Thử từng nhóm feature có giả thuyết.
5. Tuning quy mô mô hình và regularization ở mức hợp lý.
6. Kiểm tra độ ổn định trên các fold và seed.
7. Khóa protocol trước khi đánh giá final test.

Có thể dùng staged search để tiết kiệm compute, nhưng không tuyên bố
đã tìm tối ưu toàn cục; kiểm tra tương tác quan trọng khi có bằng chứng.

Ghi lại toàn bộ trial đã chạy, kể cả thất bại và kết quả kém.
Không chỉ lưu model tốt nhất.

Nếu dùng ngưỡng chuyển forecast thành tín hiệu giao dịch:
- Xem đây là hyperparameter riêng.
- Chọn trên validation với chi phí đã định nghĩa.
- Không chọn ngưỡng từ final test.
- Không nhầm trading threshold với ACF significance threshold.

### 11.7. Hợp đồng lưu log và artifact

Sử dụng đường dẫn tương đối với project root, có thể cấu hình:

log/<model_name>/<run_id>/

Trên Windows, dùng pathlib hoặc cơ chế portable.
Không ghi cứng /log ở filesystem root.

run_id phải tránh trùng, gồm timestamp, timezone/UTC convention
và thành phần định danh bổ sung khi cần.
Chỉ ngày thực hiện là chưa đủ để phân biệt nhiều run trong ngày.

Cấu trúc đề xuất; chỉ tạo artifact có ý nghĩa cho run thực tế:

log/
  <model_name>/
    <run_id>/
      manifest.json
      config.json
      environment.json
      data_manifest.json
      split_manifest.json
      run.log
      summary.md

      analysis/
        ACF_PACF_Analysis_<YYYY-MM-DD>.txt
        acf_pacf_values.csv
        z_table_<YYYY-MM-DD>.txt
        stationary_test.txt
        stationarity_results.json
        feature_report.json

      training/
        model_summary.txt
        model_config.json
        history.csv
        trials.csv

      evaluation/
        metrics.json
        predictions.csv
        forecast_values.txt
        residual_diagnostics.json
        backtest_metrics.json

      artifacts/
        <model artifact>
        <preprocessing artifact>
        feature_schema.json

      figures/
        <relevant generated plots>

Với walk-forward:
- Lưu chẩn đoán, preprocessing và artifact riêng theo fold/trial
  hoặc tham chiếu bằng ID/hash không nhập nhằng.
- Không ghi đè analysis của fold trước bằng fold sau.
- Tách run manifest tổng và kết quả từng fold.
- Không sao chép cùng dữ liệu lớn nhiều lần nếu có thể tham chiếu
  snapshot chung bằng hash.

Dùng tên đúng “forecast_values”, tránh duy trì lỗi chính tả
“forecats_value” nếu không có yêu cầu tương thích hiện hữu.

Các file TXT phục vụ đọc nhanh.
CSV/JSON là dữ liệu có cấu trúc cho kiểm thử và tổng hợp.
Sinh cả hai từ cùng kết quả nội bộ, không tính lại bằng logic khác.

Không tạo file rỗng hoặc nội dung giả để đủ danh sách.

Nếu một analysis không áp dụng:
- Ghi trạng thái not_applicable và lý do trong manifest.
- Không dựng kết quả kiểm định hoặc model summary giả.

Manifest tối thiểu ghi:

- Run ID, model name và loại run.
- Parent experiment/fold/trial nếu có.
- started_at, ended_at, duration.
- running/completed/failed/interrupted.
- Lệnh thực thi đã loại thông tin nhạy cảm.
- Config và config hash.
- Git commit và dirty state.
- Data hash, split ID và environment reference.
- Seed, device và deterministic settings.
- Danh sách artifact và checksum khi phù hợp.
- Lỗi hoặc lý do run chưa hoàn thành.

Không đánh dấu completed chỉ vì training kết thúc:
phải xác minh các stage bắt buộc và artifact tương ứng.

### 11.8. Nội dung model summary, forecasts và biểu đồ

model_summary.txt phải được sinh từ mô hình thực sự dùng trong run.

Với neural network, lưu khi áp dụng:

- Layer names/types.
- Input/output shapes.
- Total, trainable và non-trainable parameters.
- Input feature order.
- Lookback và horizons.
- Target và output heads.
- Loss, optimizer, learning rate.
- Batch size, epoch giới hạn và số epoch thực chạy.
- Early stopping criterion và best epoch.
- Checkpoint được dùng để đánh giá.

Với mô hình thống kê hoặc baseline:
- Lưu specification, tham số và summary phù hợp.
- Không ép chúng vào định dạng summary của neural network.

Nếu cần ảnh sơ đồ kiến trúc, chỉ tạo khi dependency hỗ trợ
và nó giúp giải thích. Thiếu ảnh không đồng nghĩa run thất bại.
Không giả định đã có ảnh tham chiếu khi người dùng chưa cung cấp.

predictions.csv phải phân biệt tối thiểu:

- Run/model/fold.
- Asset.
- Forecast origin.
- Target timestamp.
- Horizon.
- Target type và đơn vị.
- Prediction.
- Actual nếu đã quan sát được.
- Baseline prediction.
- Interval bounds và nominal coverage nếu có.

Tách future forecasts chưa có actual khỏi kết quả đã chấm điểm.
Không điền actual giả hoặc nối đường dự báo như thể đã kiểm chứng.

forecast_values.txt là bản đọc nhanh của dữ liệu dự báo có cấu trúc.

Biểu đồ ưu tiên:

- Training series và ranh giới split.
- ACF/PACF đúng series, fold và confidence method.
- Feature diagnostics nếu phục vụ quyết định.
- Training/validation learning curves.
- Actual/predicted và baseline ngoài mẫu.
- Residual theo thời gian và residual ACF.
- Errors theo fold/horizon.
- Equity/drawdown khi có backtest hợp lệ.
- Coverage/interval width khi có probabilistic forecasts.

Mỗi hình phải có title, trục, đơn vị, khoảng thời gian
và thông tin fold/run phù hợp.
Không dùng biểu đồ bám sát mức giá làm bằng chứng duy nhất
về chất lượng dự báo.

### 11.9. Residual diagnostics và diễn giải

Sau huấn luyện, kiểm tra residual ngoài mẫu ở giai đoạn phù hợp:

- Mean/bias.
- Errors theo thời gian.
- ACF của residual.
- Absolute/squared residual khi liên quan đến biến động.
- Ljung–Box khi các giả định và sample size phù hợp.

Ghi lags, model degrees-of-freedom adjustment nếu có,
sample size và giới hạn diễn giải.

Không mặc định p-value lớn chứng minh residual độc lập,
phân phối chuẩn hoặc model đã tối ưu.

Với multi-step overlapping errors, không dùng máy móc kiểm định
thiết kế cho one-step residuals độc lập hơn; ghi rõ phụ thuộc
do forecast horizon.

Residual diagnostics trên validation có thể phục vụ cải tiến.
Residual diagnostics trên final test chỉ phục vụ báo cáo cuối.
Nếu dùng final test để chỉnh model, phải công khai rằng tập đó
đã trở thành development data.

Không buộc residual phải giống white noise hoàn hảo mới được
bàn giao một mô hình có giới hạn đã được mô tả trung thực.

### 11.10. Pipeline thực nghiệm hoàn chỉnh

Tích hợp với CLI/package hiện có; không dựng pipeline song song
nếu có thể mở rộng thành phần đang hoạt động.

Pipeline mục tiêu:

1. Đọc và validate config.
2. Tạo run directory và manifest trạng thái running.
3. Thu thập environment/code provenance.
4. Ingest hoặc đọc data snapshot.
5. Kiểm tra chất lượng và lưu data manifest.
6. Tạo chronological split/walk-forward folds.
7. Chẩn đoán training series theo từng fold.
8. Đề xuất lag/features/search space trong training scope.
9. Fit preprocessing trong training scope.
10. Chạy baseline và các trial theo ngân sách.
11. Chọn cấu hình/checkpoint bằng validation.
12. Khóa cấu hình và chính sách refit.
13. Nếu refit, thực hiện đúng chính sách đã công bố.
14. Đánh giá final test không tham gia lựa chọn.
15. Chạy backtest nếu có execution contract hợp lệ.
16. Lưu model, preprocessing, predictions và metrics.
17. Tổng hợp report và artifact manifest.
18. Nạp lại artifact và chạy inference smoke test.
19. Kiểm tra tính đầy đủ rồi cập nhật run status.

Nếu refit sau lựa chọn:
- Ghi rõ dữ liệu refit.
- Lưu preprocessing được fit lại tương ứng.
- Không dùng final test để quyết định epoch.
- Định nghĩa trước cách chọn số epoch hoặc tập early stopping.

Nếu chạy thất bại:
- Giữ log và artifact đã có.
- Ghi stage, traceback và failed status.
- Không nuốt exception để giả vờ pipeline thành công.
- Không tự động ghi đè run cũ khi retry.

Nếu resume:
- Kiểm tra config/data/artifact compatibility.
- Ghi lineage với run trước.
- Nếu chưa triển khai resume an toàn, tạo run mới thay vì giả hỗ trợ.

summary.md được tạo từ artifact thực tế, gồm:

- Bài toán và dữ liệu.
- Features/lookback đã dùng.
- Kết quả stationarity và ACF/PACF đáng chú ý.
- Model/config được chọn và lý do.
- Baseline comparison.
- Kết quả ngoài mẫu.
- Chi phí tài nguyên đo được.
- Warnings, giới hạn và bước tiếp theo.

Không dùng diễn giải tự động để bổ sung kết quả không tồn tại.

### 11.11. Kiểm thử và acceptance criteria bổ sung

Thiết kế kiểm thử có khả năng phát hiện sai phương pháp:

- Window/target alignment trên chuỗi nhỏ có đáp án biết trước.
- Không fit scaler/selector bằng validation hoặc test.
- Thay đổi dữ liệu tương lai không làm đổi feature, diagnostic
  hoặc quyết định ở quá khứ khi giữ nguyên training data.
- Sparse lag và dense lookback có semantics đúng.
- So sánh lookback sử dụng cùng evaluation origins.
- Diagnostics xử lý chuỗi ngắn, hằng và missing values rõ ràng.
- Không nhầm confidence interval quanh estimate với dải quanh zero.
- Cấu hình alpha/method được lưu và áp dụng nhất quán.
- Run ID không gây ghi đè.
- Failed run không được ghi completed.
- Forecast origin/target/horizon được lưu đúng.
- Metrics tái tính từ predictions khớp với metrics artifact.
- Nạp lại model/preprocessing cho kết quả khớp trong tolerance
  phù hợp môi trường.
- Offline smoke run không truy cập mạng ngoài ý muốn.

Với synthetic AR/MA/white-noise data:
- Có thể dùng để kiểm tra hành vi chẩn đoán và estimator.
- Không viết test mong mọi spike/p-value ngẫu nhiên phải đúng
  một mẫu cố định.
- Không trình bày synthetic results như kết quả thị trường.

Chỉ gọi pipeline hoàn thành khi:

A. Có một offline smoke run được kiểm chứng.
B. Có kiểm thử cho các rủi ro leakage/alignment đã xác định.
C. Các artifact bắt buộc của run thành công thực sự tồn tại.
D. Có thể truy vết từ report đến predictions, config, data và code.
E. Có thể nạp artifact để inference bằng cùng input contract.
F. Kết quả trên dữ liệu thị trường được gắn nhãn riêng và chỉ
   tuyên bố khi đã thực sự chạy.
G. Tài liệu nêu rõ những stage chưa được kiểm chứng.

Không yêu cầu model phải thắng baseline để gọi pipeline đúng.
Muốn tuyên bố cải thiện mô hình phải có bằng chứng thực nghiệm riêng.

### 11.12. Thứ tự thực hiện phần bổ sung

Không triển khai đồng loạt toàn bộ cấu trúc artifact ở trên.

Thực hiện lần lượt:

1. Kiểm tra những khả năng repo đã có.
2. Chuẩn hóa ý nghĩa lag, target và split.
3. Hoàn thiện run manifest và logging tối thiểu.
4. Thêm stationarity/ACF/PACF trên training scope.
5. Thêm lag/lookback experiment với ngân sách nhỏ.
6. Kiểm soát feature space bằng ablation.
7. Kết nối training, evaluation, artifact reload và report.
8. Bổ sung diagnostics nâng cao khi có nhu cầu và bằng chứng.

Mỗi bước phải có acceptance criteria, kiểm thử, diff review
và cập nhật tracking trước khi chuyển sang bước tiếp theo.

Khi hướng dẫn tôi, giải thích trên dữ liệu và code thực tế:

- Vì sao một lag được đề xuất nhưng chưa chắc nên giữ.
- Vì sao ACF cao chưa chứng minh dự báo tốt.
- Vì sao feature selection trên toàn bộ dữ liệu gây leakage.
- Vì sao p-value không phải xác suất mô hình đúng.
- Vì sao log đầy đủ giúp tái lập nhưng không tự bảo đảm
  phương pháp nghiên cứu đúng.

## 12. Bắt đầu ngay

Bắt đầu với M0 bằng khảo sát chỉ đọc.

Phản hồi đầu tiên sau khảo sát cần cho tôi thấy:

1. Repo thực tế gồm những gì.
2. Luồng chạy chính và các thành phần kế thừa.
3. Tài liệu có điểm nào chưa khớp với code.
4. Những rủi ro phương pháp cần kiểm chứng trước.
5. Baseline kiểm tra nào đã chạy và kết quả.
6. Nhiệm vụ đầu tiên có phạm vi nhỏ và tiêu chí hoàn thành rõ ràng.

Sau đó chủ động tiếp tục các bước an toàn trong phạm vi nhiệm vụ.
Không bắt đầu bằng đổi tên, xóa code, dựng lại cấu trúc thư mục
hoặc huấn luyện toàn bộ mô hình.
