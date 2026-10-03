# Phân tích kết quả Memory System

## 1. Phạm vi đánh giá

Bài làm so sánh hai kiến trúc trên cùng dữ liệu tiếng Việt và cùng cơ chế đo:

- **Baseline Agent** chỉ giữ short-term memory trong một thread. Khi chuyển sang
  thread recall mới, agent không còn lịch sử của thread trước.
- **Advanced Agent** kết hợp short-term memory, persistent memory trong `User.md`
  và compact memory cho hội thoại dài.

Benchmark chạy ở chế độ offline để kết quả lặp lại được, không phụ thuộc API key
hay độ ngẫu nhiên của mô hình. `Response quality` hiện được tính bằng tỷ lệ fact
mong đợi xuất hiện trong câu trả lời, không phải điểm do LLM judge chấm.

## 2. Kết quả benchmark

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2,111 | 18,727 | 0.0% | 0.0% | 0 bytes | 0 |
| Advanced | 3,338 | 27,521 | 92.9% | 92.9% | 275 bytes | 1 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 329 | 23,050 | 0.0% | 0.0% | 0 bytes | 0 |
| Advanced | 708 | 8,134 | 100.0% | 100.0% | 242 bytes | 23 |

Các con số trên là token ước lượng theo quy tắc xác định của lab, phù hợp để so
sánh tương đối giữa hai agent nhưng không nên xem là hóa đơn token chính xác của
một provider cụ thể.

## 3. Vì sao Advanced recall tốt hơn Baseline?

Baseline lưu lịch sử theo `thread_id`. Các câu hỏi recall được cố ý gửi vào một
thread mới, nên Baseline không có nguồn dữ liệu để khôi phục tên, nghề nghiệp,
nơi ở hay preference của người dùng. Kết quả recall 0% vì vậy là hành vi đúng với
thiết kế, không phải lỗi triển khai.

Advanced trích các fact ổn định thành field có cấu trúc rồi ghi vào
`state/profiles/<user>/User.md`. Khi sang thread mới, agent đọc lại file này và
đưa profile vào context. Persistent memory tách khỏi vòng đời của thread giúp
Advanced đạt 92.9% ở benchmark chuẩn và 100% ở stress benchmark.

Mức 92.9% thay vì 100% cũng cho thấy giới hạn thực tế: extractor dựa trên rule và
quality metric dựa trên chuỗi mong đợi chính xác. Một cách diễn đạt preference
khác với chuỗi trong đáp án có thể bị tính là sai dù ý nghĩa gần nhau. Đây là lý
do nên bổ sung synonym normalization hoặc LLM judge có rubric nếu cần đánh giá
ngữ nghĩa sâu hơn.

## 4. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Trong Standard Benchmark, Advanced xử lý 27,521 prompt token, cao hơn 18,727 của
Baseline. Mỗi lượt của Advanced phải mang thêm `User.md` và có thể cả compact
summary vào prompt. Khi lịch sử còn ngắn, phần overhead này lớn hơn lượng token
tiết kiệm được từ compaction; benchmark chuẩn cũng chỉ kích hoạt compaction một
lần. Advanced còn sinh câu trả lời chứa các fact đã recall nên `Agent tokens
only` cũng cao hơn.

Do đó compact memory không bảo đảm thắng ở mọi tình huống. Với hội thoại ngắn,
Baseline đơn giản hơn, ít I/O hơn và có prompt nhỏ hơn. Chi phí bổ sung của
Advanced chỉ hợp lý khi ứng dụng thật sự cần nhớ qua nhiều phiên hoặc có thread
dài.

## 5. Vì sao compact có lợi thế ở hội thoại dài?

Nếu luôn gửi lại toàn bộ lịch sử, context của Baseline tăng sau mỗi lượt. Tổng
`Prompt tokens processed` vì thế tăng gần theo cấp bậc hai: cùng một đoạn cũ bị
xử lý lại nhiều lần ở các prompt sau.

Advanced giữ một số message gần nhất ở dạng nguyên văn và nén phần cũ thành
summary có kích thước giới hạn. Stress benchmark kích hoạt 23 lần compact và làm
prompt token giảm từ 23,050 xuống 8,134, tương đương giảm khoảng **64.7%** so với
Baseline. Đây mới là lợi ích chính của compaction; nó tối ưu lượng context được
xử lý, không nhất thiết làm câu trả lời ngắn hơn. Thực tế Advanced vẫn dùng 708
output token so với 329 của Baseline.

Đổi lại, summary là một phép nén mất dữ liệu. Chi tiết ít được nhắc lại, quan hệ
giữa các sự kiện hoặc open thread có thể biến mất. Vì vậy các fact dài hạn quan
trọng phải được đưa vào persistent memory thay vì chỉ dựa vào compact summary.

## 6. Memory growth và rủi ro

Baseline không tạo file persistent nên memory growth bằng 0. Advanced tăng 275
bytes trong benchmark chuẩn và 242 bytes trong stress benchmark. Mức tăng hiện
tại nhỏ vì `User.md` chỉ chứa một tập field đã chuẩn hóa, nhưng trong hệ thống sử
dụng lâu dài file vẫn có thể phình theo số lượng entity và preference.

Các rủi ro chính gồm:

- **Fact sai được lưu lâu dài:** câu nói đùa, giả định hoặc phát biểu không chắc
  chắn có thể làm recall sai ở mọi session sau.
- **Xung đột thông tin:** giá trị cũ như `backend engineer` hoặc `Đà Nẵng` phải
  được thay thế khi người dùng correction, thay vì tồn tại đồng thời với giá trị
  mới.
- **Thông tin lỗi thời:** một fact từng đúng có thể không còn đúng dù không có
  correction rõ ràng; hệ thống hiện chưa có memory decay hay thời hạn sử dụng.
- **Riêng tư và bảo mật:** profile bền vững cần chính sách đồng ý, giới hạn dữ
  liệu, quyền xem/xóa và bảo vệ file phù hợp.
- **Prompt tăng dần:** nếu lưu quá nhiều field, toàn bộ profile được chèn vào mỗi
  prompt sẽ làm mất một phần lợi ích token của compact memory.

Bonus confidence threshold giảm rủi ro đầu tiên: phát biểu rõ ràng có confidence
`0.95`, correction tường minh có `0.99`, còn câu dè dặt chỉ có `0.55`. Với ngưỡng
mặc định `0.8`, fact không chắc chắn vẫn nằm trong short-term memory nhưng không
được ghi vào `User.md`. Guardrail này cải thiện độ tin cậy của recall, song rule
theo từ khóa có thể bỏ sót cách diễn đạt mới; ngưỡng quá cao cũng tạo false
negative và làm agent quên fact hợp lệ.

## 7. Kết luận

Kết quả thể hiện đúng trade-off của một memory system:

1. Baseline rẻ và đơn giản trong hội thoại ngắn nhưng không recall qua session.
2. Persistent `User.md` giúp Advanced nhớ người dùng qua thread mới.
3. Advanced có overhead rõ rệt khi context còn ngắn.
4. Khi hội thoại dài, compact memory giảm mạnh prompt token phải xử lý.
5. Khả năng nhớ tốt hơn đi kèm độ phức tạp, rủi ro lưu sai, dữ liệu lỗi thời và
   yêu cầu quản trị quyền riêng tư.

Vì vậy, thiết kế phù hợp không phải là “lưu mọi thứ”, mà là phân loại đúng dữ liệu
cho short-term, persistent và compact memory, đồng thời đặt guardrail trước khi
biến một phát biểu thành ký ức dài hạn.
