# Tóm tắt notebook: OULAD FE — Sequence Version (Realtime EWS)

## Mục tiêu chính
- Chuyển dữ liệu OULAD từ dạng tabular sang dạng chuỗi theo tuần (causal weekly panel).
- Giữ tính causal (nhìn về quá khứ): mỗi tính năng tuần `t` chỉ dùng dữ liệu từ ngày `0` đến ngày `7*(t+1)-1` (không leak dữ liệu tương lai).
- Chuẩn bị dữ liệu cho mô hình LSTM/Transformer dự đoán realtime Early Warning System (EWS).
- Xuất dữ liệu ở cả 3 định dạng: `LONG panel`, `WIDE snapshot`, và `TENSOR`.

## 1. Thiết lập và tiền xử lý dữ liệu
- Tải các file gốc: `assessments.csv`, `studentInfo.csv`, `studentAssessment.csv`, `studentRegistration.csv`, `studentVle.csv`, `courses.csv`, `vle.csv`.
- Đặt cutoff: `CUTOFF_DAY = 170` ngày, tương ứng **25 tuần** (`T_MAX = 25`).
- Xây target nhị phân: `Fail/Withdrawn = 1` (cần can thiệp), `Pass/Distinction = 0`.
- Làm sạch và chuẩn hóa:
  - Điền giá trị `date_registration` bằng median theo module/presentation.
  - Loại bỏ cột `date_unregistration` nếu có.
  - Chuẩn hóa `imd_band`, `age_band` thiếu bằng `Unknown`.
  - Loại bỏ cột `week_from/week_to` nếu có trong `df_vle`.
  - Điền điểm thi `score` bị missing = 0.

## 2. Static features
- Tạo `df_static` chứa các thông tin không thay đổi theo tuần của sinh viên:
  - Demographics: `gender`, `age_band`, `imd_band`, `region`, `highest_education`, `disability`, `studied_credits`, `num_of_prev_attempts`.
  - Thông tin đăng ký & context: `date_registration`, `code_module`, `code_presentation`, `module_semester`, `semester`.
  - Pre-course activity (date < 0): `pre_course_clicks`, `pre_course_active_days`, `has_pre_course_activity`.

## 3. Dynamic per-week features (Weekly Causal)
### 3.1 Chuẩn bị panel tuần
- Lọc dữ liệu VLE trong khoảng `[0, 170]` và gán `week = date // 7`.
- Lọc assessment không phải Exam, deadline <= cutoff, submission trong khoảng valid (`[0, 170]`).

### 3.2 Tổng hợp VLE theo tuần
- Tổng clicks (`clicks_this_week`) và số ngày hoạt động (`active_days_this_week`) mỗi tuần.
- Tính `max/median/std` clicks hàng ngày trong tuần.
- Phân tích clicks theo loại hoạt động: `oucontent`, `forumng`, `homepage`, `subpage`, `url`, `quiz`.
- Tổng clicks cuối tuần (weekend).

### 3.3 Tổng hợp assessment theo tuần
- Số bài đã nộp, tổng điểm, tổng trọng số, điểm có trọng số, days-before-deadline.
- Số TMA/CMA đã nộp và số TMA/CMA dự kiến theo deadline của tuần hiện tại.

### 3.4 Panel full-grid
- Tạo grid đầy đủ: `N attempts x T_MAX (25) weeks`.
- Merge tất cả aggregation ở trên vào panel, fill 0 cho các tuần sinh viên không có hoạt động.

### 3.5 Tạo tính năng causal/cumulative (chỉ tính lũy kế về quá khứ)
- **VLE cumulative:** `cum_clicks`, `cum_active_days`, `active_rate_so_far`, `clicks_per_active_day_cum`.
- **Hoạt động gần nhất:** `days_since_last_active` (tính đến cuối tuần `t`), `active_span_so_far`.
- **Ratios:** `ratio_{type}_so_far` (cho từng loại hoạt động), `weekend_ratio_so_far`.
- **Slopes:** `click_slope_recent`, `click_slope_r2_recent` (regress trên cửa sổ 4 tuần gần nhất).
- **Mật độ hoạt động & Gaps:** `activity_density_so_far`, `gap_std_so_far`, `gap_trend_so_far` (tính trên tập hợp các ngày hoạt động trước cuối tuần `t`).
- **Assessment cumulative:** `mean_score_so_far`, `weighted_score_so_far`, `has_weighted_score_so_far`, `avg_days_before_deadline_so_far`, `tma_submission_rate_so_far`, `cma_submission_rate_so_far`, `n_missing_so_far`.

## 4. Encoding và chia tập
- Mã hóa categorical static features:
  - `highest_education_encoded`, `age_band_encoded`, `imd_band_encoded` bằng thứ tự ordinal.
  - `semester_enc`, `gender_encoded`, `disability_encoded`, `module_semester_encoded`.
  - `region` được one-hot encode thành 12 cột dummy nhị phân.
- Danh sách các tập tính năng:
  - `DYNAMIC_FEATURES`: gồm **33** tính năng động thay đổi theo tuần.
  - `STATIC_FEATURES`: gồm **25** tính năng tĩnh (bao gồm cả các cột region dummy).
- Chia tập Train/Test theo thời gian (OOT - Out-Of-Time):
  - **Train cohorts:** Các học kỳ `2013B`, `2013J`, `2014B` (21,333 attempts).
  - **Test cohort:** Học kỳ cuối cùng `2014J` (11,260 attempts).

## 5. Chuyển sang tensor và scale
- **Tensor format:**
  - `X_seq`: dạng 3D `(N, T_MAX, 33)` cho LSTM/Transformer.
  - `X_static`: dạng 2D `(N, 25)`.
  - `mask`: dạng 2D `(N, T_MAX)` đánh dấu các vị trí padding/truncation (mặc định ban đầu là `False` hết vì panel full-grid).
  - `y`: dạng 1D `(N,)` nhãn nhị phân.
- **Scale dữ liệu:**
  - Chuẩn hóa dynamic features bằng `StandardScaler` fit trên train (chỉ tính trên các vị trí không bị pad/mask).
  - Chuẩn hóa static features bằng `StandardScaler` độc lập.

## 6. Phân tích tính năng và lọc bỏ (Feature Selection)
- Đánh giá correlation (Pearson) cho các tính năng số với target nhị phân (ngưỡng lọc `|corr| >= 0.03`).
- Dùng kiểm định Chi-square cho các tính năng phân loại static với target (ngưỡng lọc `p-value <= 0.05`).
- Rút gọn danh sách tính năng cho tập Train:
  - **Dynamic features:** từ 33 rút gọn xuống **24** tính năng động (`FINAL_DYNAMIC_FEATURES`). Loại bỏ 9 tính năng: `active_rate_so_far`, `active_days_this_week`, `max_daily_clicks_week`, `median_daily_clicks_week`, `n_missing_so_far`, `active_span_so_far`, `cum_clicks`, `weighted_score_so_far`, `std_clicks_week`.
  - **Static features:** từ 25 rút gọn xuống **14** tính năng tĩnh (`FINAL_STATIC_FEATURES`). Loại bỏ 11 tính năng không đạt ngưỡng: `highest_education_encoded`, `imd_band_encoded`, `module_semester_encoded`, `region_Scotland`, `region_Ireland`, `region_North Region`, `region_North Western Region`, `region_West Midlands Region`, `region_South Region`, `region_London Region`, `region_South West Region`.
