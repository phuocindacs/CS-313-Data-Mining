Dưới đây là nội dung giai đoạn **Feature Engineering** được cập nhật và trình bày theo định dạng Markdown (.md) để bạn lưu vào tài liệu dự án, hoàn toàn không kèm các trích dẫn:

# Kế hoạch Feature Engineering - Hệ thống Cảnh báo Sớm (OULAD)

## 1. Tiền xử lý dữ liệu & Làm sạch (Semantic Nulls)
Dựa trên phân tích ngữ nghĩa của các giá trị trống, các bước xử lý sau là bắt buộc để đảm bảo chất lượng dữ liệu đầu vào:
* **date_registration**: Điền các giá trị thiếu bằng giá trị trung vị (median) của đợt học tương ứng để tránh mất mẫu.
* **imd_band**: Chuyển đổi các giá trị NULL thành nhãn "Unknown". Việc này giúp giữ lại thông tin về những sinh viên không có dữ liệu vùng nhưng vẫn có thể mang đặc điểm hành vi riêng biệt.
* **score**: Thực hiện điền giá trị 0 cho các cột điểm bị trống, vì trong ngữ nghĩa của bộ dữ liệu OULAD, NULL thường đồng nghĩa với việc sinh viên không nộp bài.
* **Loại bỏ cột nhiễu**: Xóa bỏ các cột **week_from** và **week_to** do tỷ lệ trống quá cao. Loại bỏ **date_unregistration** khỏi danh sách tính năng dự báo để tránh rò rỉ dữ liệu (vì đây là biến định nghĩa mục tiêu).

## 2. Nhóm tính năng động lực trước khóa học (Pre-Course Engagement)
Đây là tập hợp các tính năng quan trọng đo lường sự chuẩn bị của sinh viên trước ngày khai giảng chính thức (ngày < 0):
* **pre_course_clicks**: Tổng lượng tương tác của sinh viên trước khi khóa học bắt đầu.
* **pre_course_active_days**: Số ngày sinh viên truy cập vào hệ thống học tập trước ngày 0.
* **has_pre_course_activity**: Biến nhị phân xác định sinh viên có chuẩn bị sớm hay không. Đây là tín hiệu mạnh để phân biệt giữa nhóm có nguy cơ rút môn và nhóm học tập tích cực.

## 3. Nhóm tính năng tương tác VLE (Temporal Features)
Tập trung vào các đặc trưng hành vi biến thiên theo thời gian tính đến ngày 135:
* **last_active_day**: Ngày tương tác cuối cùng của sinh viên. Đây là chỉ số quan trọng nhất để nhận diện nhóm sắp rút môn.
* **std_clicks**: Độ lệch chuẩn của lượng click hàng ngày để đo lường tính ổn định và kỷ luật trong việc học.
* **active_days**: Tổng số ngày thực tế có phát sinh tương tác trên hệ thống.
* **Weekly Clicks**: Tổng lượng click được chia theo từng tuần để mô hình nắm bắt được xu hướng (tăng dần hoặc giảm dần) của sinh viên.
* **Activity Breakdown**: Tách riêng lượng click của các loại tài nguyên có tính tương tác cao như Bài kiểm tra (quiz), Nội dung bài học (oucontent) và Diễn đàn (forumng).
   * clicks_oucontent: nội dung chính
   * clicks_forumng: forum — tương quan cao với pass
   * clicks_quiz, clicks_resource: secondary signals

## 4. Nhóm tính năng đánh giá & Học tập (Assessment Features)
Chuẩn hóa dữ liệu đánh giá để đảm bảo tính công bằng giữa các module khác nhau:
* **weighted_score_before_cutoff = sum(score × weight) / sum(weight)**: Tổng điểm tích lũy có trọng số cho các bài kiểm tra diễn ra trước ngày cut-off.

* **tma_submission_rate & cma_submission_rate**: Tỷ lệ số bài thực tế đã nộp trên tổng số bài phải nộp của từng loại hình đánh giá.
   * tma_submission_rate = n_tma_submitted / n_safe_tma_in_module
   * cma_submission_rate = n_cma_submitted / n_safe_cma_in_module 
   * AAA/EEE không có CMA → fillna(0) hoặc -1
* **avg_days_before_deadline**: Trung bình số ngày sinh viên nộp bài trước hạn. Chỉ số này phản ánh thói quen và sự chủ động của người học.
* **n_missing_submission**: Đếm tổng số bài đánh giá mà sinh viên đã bỏ lỡ không nộp.
* **avg_days_before_deadline**: trung bình nộp trước deadline bao nhiêu ngày

## 5. Nhóm tính năng bối cảnh & Nhân khẩu học
* **module_semester**: Tạo biến kết hợp giữa mã môn học và học kỳ (ví dụ: AAA_B, AAA_J) để mô hình xử lý được sự khác biệt về độ khó hoặc đặc điểm của từng đợt dạy.
* **num_of_prev_attempts**: Giữ lại tính năng này vì số lần thi lại có tương quan rất mạnh đến khả năng tiếp tục trượt hoặc rút môn.
* **Mã hóa dữ liệu**: Sử dụng mã hóa thứ tự cho **highest_education** và **age_band**.
* Xóa bỏ tính năng giới tính **gender** do không mang lại giá trị dự báo đáng kể trong các thử nghiệm thống kê.

## 6. Danh mục kiểm tra khi huấn luyện (Modeling Checklist)
* **Stratified Split**: Chia tập dữ liệu Train/Test theo tỷ lệ cân bằng của biến mục tiêu.
* **Xử lý mất cân bằng**: Áp dụng trọng số lớp (class weights) cho nhóm sinh viên đạt loại Giỏi (Distinction) do nhóm này chiếm tỷ lệ nhỏ.
* **Leakage Audit**: Kiểm tra lại toàn bộ quy trình để đảm bảo không sử dụng bất kỳ thông tin nào phát sinh sau ngày 135 (đặc biệt là điểm thi cuối kỳ).
* **Module Consistency**: Đảm bảo tất cả các mã môn học đều xuất hiện ở cả hai tập dữ liệu huấn luyện và kiểm tra.

---
 
## 7. Objective & Metric
 
* **Bài toán**: Phân loại 4 class (Pass / Fail / Withdrawn / Distinction), ưu tiên phát hiện **Withdrawn**.
* **Metric chính**: `Recall(Withdrawn)` — tối ưu hóa khả năng phát hiện sinh viên sắp rút môn.
* **Guardrail**: `Macro-F1` — đảm bảo mô hình không bỏ qua hoàn toàn các class còn lại.
* **Không dùng**: Overall Accuracy — bị méo do mất cân bằng class.
 
---
 
## 8. Setup & Reproducibility
 
* `CUTOFF_DAY = 135`
* `RANDOM_SEED = 42` dùng xuyên suốt toàn bộ pipeline
* Ghi lại phiên bản dữ liệu và feature set khi lưu artifact để tái hiện thí nghiệm.
 
---
 
## 9. Định nghĩa "Safe Assessment"
 
* **Safe** = `assessment.deadline <= CUTOFF_DAY`
* Định nghĩa dựa trên **deadline của bài**, không phải ngày sinh viên thực nộp — vì nộp trễ chính là signal rủi ro quan trọng.
* `n_safe_tma_in_module` = số assessment có `type == 'TMA'` và `deadline <= 135` trong module đó.
* `n_safe_cma_in_module` = số assessment có `type == 'CMA'` và `deadline <= 135` trong module đó.
* Toàn bộ Exam bị drop trước khi tính feature vì Exam luôn diễn ra sau ngày 135 → leakage 100%.
 
---
 
## 10. Edge Cases Assessment
 
### Module GGG (TMA weight = 0)
* `weighted_score_before_cutoff = NaN` (mẫu số = 0, không tính được)
* Thêm cột `has_weighted_score = 0` để mô hình phân biệt GGG với các module bình thường.
* Tín hiệu thay thế: dùng **TMA submission count** vì nộp bài vẫn là engagement signal dù weight = 0.
 
### Module AAA, EEE (không có CMA trước cutoff)
* `cma_submission_rate = -1` (phân biệt "module không có CMA" với "có CMA nhưng không nộp")
* Không dùng `0` vì gây nhầm lẫn với trường hợp sinh viên không nộp bài.
 
### Quy tắc chung cho submission rate
| Trường hợp | Giá trị |
|---|---|
| Module không có CMA/TMA trước cutoff | `-1` |
| Có assessment nhưng sinh viên không nộp | `0` |
| Bình thường | `n_submitted / n_safe_in_module` |
 
### avg_days_before_deadline âm
* Giá trị âm = sinh viên nộp trễ → **giữ nguyên**, không clip.
* Đây là signal hành vi quan trọng, clip về 0 sẽ mất thông tin.
 
---
 
## 11. Xử lý Student không có VLE Record
 
Sinh viên không xuất hiện trong bảng `studentVle` (không có bất kỳ tương tác nào):
* Toàn bộ VLE features → fill `0`
* Thêm cột `has_vle_activity = 0` để mô hình phân biệt với sinh viên có tương tác thấp.
 
---
 
## 12. Weekly Clicks — Scope
 
* Số tuần = `ceil(135 / 7) = 20 tuần` → tạo 20 cột `clicks_week_1` đến `clicks_week_20`.
* Tuần sinh viên không active → fill `0`.
* Tuần nằm ngoài khoảng active của sinh viên (ví dụ đăng ký muộn) → fill `0`.
 
---
 
## 13. Pipeline Order
 
Thứ tự thực hiện bắt buộc để tránh leakage:
 
```
Load 5 bảng
    → Tiền xử lý & Semantic Nulls (Section 1)
    → Filter: drop Exam, chỉ giữ data <= CUTOFF_DAY
    → Tính toán toàn bộ features (Section 2–5 + 10–12)
    → Split theo Student-level (id_student) + Stratified (final_result × code_module)
    → Encode & Scale chỉ fit trên Train, transform cả Train và Test
```
 
* **Encode/Scale phải sau split** — nếu làm trước sẽ bị data leakage từ Test vào Train.
* **Student-level split**: nhóm toàn bộ records của cùng một `id_student` về cùng một phía (Train hoặc Test) để tránh student leakage (12.3% sinh viên có nhiều records).
 
---
 
## 14. Encoding & Scaling Chi Tiết
 
### Ordinal Encoding
* `highest_education`: No Formal Quals < Lower Than A Level < A Level or Equivalent < HE Qualification < Post Graduate Qualification
* `age_band`: 0-35 < 35-55 < 55<=
* `imd_band`: 0-10% < 10-20% < ... < 90-100% < Unknown (Unknown = 10, cuối thang)
 
### Categorical Encoding
* `module_semester` (AAA_B, AAA_J, ...): Label Encoding — đủ dùng cho tree-based model, không cần one-hot.
 
### Scaling
* Tree-based model (Random Forest, XGBoost): **không cần scaling**.
* Nếu thử Logistic Regression: dùng `StandardScaler`, fit chỉ trên Train.