Dưới đây là nội dung giai đoạn **Feature Engineering** được cập nhật và trình bày theo định dạng Markdown (.md) để bạn lưu vào tài liệu dự án, hoàn toàn không kèm các trích dẫn:

---

# Kế hoạch Feature Engineering - Hệ thống Cảnh báo Sớm (OULAD)

## 0. Thiết lập chung (Setup & Reproducibility)
* **CUTOFF_DAY**: 135 (Điểm cắt dữ liệu để dự báo).
* **RANDOM_SEED**: 42 (Dùng xuyên suốt toàn bộ pipeline để đảm bảo tính tái lập).
* **Mục tiêu**: Phân loại 4 lớp (Pass / Fail / Withdrawn / Distinction), ưu tiên tối đa việc phát hiện sớm nhóm **Withdrawn**.
* **Metric chính**: `Recall (Withdrawn)` và `Macro-F1` để kiểm soát sự cân bằng giữa các lớp.

## 1. Tiền xử lý dữ liệu & Làm sạch (Semantic Nulls)
Dựa trên phân tích ngữ nghĩa, các bước sau là bắt buộc để đảm bảo chất lượng dữ liệu:
* **date_registration**: Điền các giá trị thiếu bằng giá trị trung vị (median) của đợt học tương ứng.
* **imd_band**: Chuyển đổi các giá trị NULL thành nhãn "Unknown" (gán giá trị 10, cuối thang đo).
* **score**: Điền giá trị 0 cho các cột điểm bị trống (NULL = không nộp bài).
* **Loại bỏ cột nhiễu**: Xóa bỏ `week_from`, `week_to` (tỷ lệ trống cao) và `date_unregistration` (tránh rò rỉ dữ liệu).

## 2. Nhóm tính năng động lực trước khóa học (Pre-Course Engagement)
Đo lường sự chuẩn bị của sinh viên trước ngày khai giảng (ngày < 0):
* **pre_course_clicks**: Tổng lượng tương tác của sinh viên trước ngày 0.
* **pre_course_active_days**: Số ngày sinh viên truy cập hệ thống trước ngày 0.
* **has_pre_course_activity**: Biến nhị phân xác định sinh viên có chuẩn bị sớm hay không.

## 3. Nhóm tính năng tương tác VLE (Temporal Features)
Tập trung vào đặc trưng hành vi biến thiên tính đến ngày 135:
* **last_active_day**: Ngày tương tác cuối cùng (chỉ số quan trọng nhất để nhận diện Withdrawn).
* **std_clicks**: Độ lệch chuẩn lượng click hàng ngày để đo tính kỷ luật.
* **active_days**: Tổng số ngày thực tế có tương tác trên hệ thống.
* **Weekly Clicks**: Tạo 20 cột (`clicks_week_1` đến `clicks_week_20`) dựa trên `ceil(135 / 7)`. Điền 0 cho các tuần không hoạt động.
* **Activity Breakdown**: Tách riêng lượng click của các tài nguyên quan trọng:
   * `clicks_oucontent`: Nội dung bài học chính.
   * `clicks_forumng`: Tương tác diễn đàn (tương quan cao với kết quả Pass).
   * `clicks_quiz`, `clicks_resource`: Các tín hiệu bổ trợ.

## 4. Nhóm tính năng đánh giá & Học tập (Assessment Features)
Quy tắc xác định **Safe Assessment**: Chỉ tính các bài có `deadline <= CUTOFF_DAY`. Toàn bộ bài thi cuối kỳ (Exam) bị loại bỏ vì gây rò rỉ dữ liệu 100%.

* **weighted_score_before_cutoff**: Tính theo công thức `sum(score × weight) / sum(weight)`.
* **tma_submission_rate & cma_submission_rate**: 
   * Công thức: `n_submitted / n_safe_in_module`.
   * Nếu module không có loại bài đó trước ngày 135: Điền `-1`.
   * Nếu có bài nhưng sinh viên không nộp: Điền `0`.
* **avg_days_before_deadline**: Trung bình số ngày nộp trước hạn (Giữ nguyên giá trị âm nếu nộp trễ, không thực hiện clip).
* **n_missing_submission**: Tổng số bài đánh giá "Safe" mà sinh viên bỏ lỡ.
* **Xử lý Edge Cases**:
   * **Module GGG**: Do trọng số TMA = 0 nên không tính được điểm tích lũy. Thêm cột `has_weighted_score = 0` để mô hình phân biệt và sử dụng `TMA submission count` làm tín hiệu thay thế.
   * **Module AAA, EEE**: Điền `-1` cho `cma_submission_rate` do không có bài CMA trước ngày 135.

## 5. Nhóm tính năng bối cảnh & Nhân khẩu học
* **module_semester**: Kết hợp mã môn và học kỳ (Ví dụ: AAA_B) -> Sử dụng Label Encoding.
* **num_of_prev_attempts**: Giữ nguyên do có tương quan mạnh với rủi ro trượt/rút môn.
* **Mã hóa thứ tự (Ordinal Encoding)**:
   * `highest_education`: Sắp xếp theo trình độ từ thấp đến cao.
   * `age_band`: 0-35 < 35-55 < 55+.
   * `imd_band`: 0-10% < ... < 90-100% < Unknown.
* **Loại bỏ**: Xóa tính năng `gender` do không đóng góp giá trị dự báo.

## 6. Danh mục kiểm tra & Tiền xử lý (Modeling Prep)
* **Stratified Split**: Chia tập dữ liệu dựa trên nhãn mục tiêu và mã môn học để đảm bảo tính đồng nhất.
* **Student-level Split**: Đảm bảo tất cả bản ghi của cùng một `id_student` phải nằm hoàn toàn trong tập Train hoặc tập Test để tránh rò rỉ thông tin sinh viên.
* **Scaling**:
    * Với mô hình cây (Random Forest, XGBoost): Không cần scaling.
    * Với mô hình tuyến tính: Sử dụng `StandardScaler`, chỉ **fit trên tập Train** và áp dụng (transform) cho tập Test.

## 7. Pipeline thực thi (Workflow Order)
Để tránh rò rỉ dữ liệu, thứ tự thực hiện phải được tuân thủ nghiêm ngặt:
1. Load dữ liệu và xử lý Semantic Nulls.
2. Lọc bỏ Exam và các dữ liệu phát sinh sau ngày 135.
3. Tính toán toàn bộ các tính năng (Sections 2-5).
4. Thực hiện Split dữ liệu (Train/Test).
5. Thực hiện Encoding và Scaling (Chỉ fit trên tập Train).
6. Huấn luyện và đánh giá mô hình.