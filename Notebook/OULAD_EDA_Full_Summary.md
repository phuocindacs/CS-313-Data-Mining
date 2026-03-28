# OULAD — EDA Full Summary & Feature Engineering Decisions

> Tổng hợp toàn bộ từ eda
> Dùng làm reference trước khi bước vào Feature Engineering & Modeling

---

## Schema tổng quan

| File | Rows | Null columns |
|------|------|-------------|
| `studentInfo.csv` | 32,593 | `imd_band` (1,111 — 3.4%) |
| `studentRegistration.csv` | 32,593 | `date_unregistration` (22,521 — 69.1%), `date_registration` (45 — 0.1%) |
| `studentAssessment.csv` | 173,912 | `score` (173 — 0.1%) |
| `studentVle.csv` | 10,655,280 | Clean |
| `assessments.csv` | 206 | `date` (11 — 5.3%) |
| `courses.csv` | 22 | Clean |
| `vle.csv` | 6,364 | `week_from`, `week_to` (82.4%) |

---

## Section 2 — Semantic Null Analysis

**Quan trọng nhất: null không phải lúc nào cũng là "thiếu data".**

| Column | Ý nghĩa thực | Xử lý |
|--------|-------------|-------|
| `date_unregistration` NULL | Sinh viên **không rút môn**, học đến cuối | KHÔNG dropna() |
| `date_unregistration` có giá trị | Sinh viên đã rút môn | Giữ nguyên |
| `score` NULL | Sinh viên **không nộp bài** | `fillna(0)` |
| `imd_band` NULL | Địa chỉ không map được vào vùng hành chính UK | Encode thành category "Unknown" |
| `week_from`, `week_to` NULL | Resource available cả module, không gắn tuần cụ thể | Drop cả 2 cột |
| `date` (assessments) NULL | Không biết ngày deadline | Drop khi join |

**Pattern null score theo final_result:**
- Withdrawn: ~0.40% null score — cao nhất
- Fail: ~0.25%
- Pass: ~0.03%
- Distinction: 0.0%

→ `fillna(0)` semantically correct: không nộp bài = điểm 0, và pattern này giúp model học signal.

---

## Section 3 — Target Variable

### Phân phối

| Label | Count | % |
|-------|-------|---|
| Pass | 12,361 | 37.9% |
| Withdrawn | 10,156 | 31.2% |
| Fail | 7,052 | 21.6% |
| Distinction | 3,024 | 9.3% |

⚠️ Class imbalance — Distinction chỉ 9.3%. Cần stratified split + class weight khi modeling.

### Feature signal từ demographics

| Feature | Signal | Ghi chú |
|---------|--------|---------|
| `num_of_prev_attempts` | **Mạnh** | Càng cao càng risk — sinh viên thi lại vì đã fail/withdraw trước đó, không phải vì thi lại làm họ fail hơn |
| `highest_education` | Trung bình | HE Qualification pass cao nhất, No Formal Quals Withdrawn cao nhất |
| `imd_band` | Trung bình | 90-100% (giàu) pass tốt hơn, 0-10% (nghèo) Withdrawn nhiều hơn |
| `age_band` | Trung bình | 0-35 Withdrawn nhiều nhất, 35+ pass tốt hơn. Note: `55<=` là label gốc = ≥55 tuổi |
| `studied_credits` | Yếu | Overlap quá lớn giữa 4 nhóm, Withdrawn variance rộng hơn một chút |
| `gender` | Rất yếu | Gần như không có diff M vs F |

**`studied_credits` là số tín chỉ đăng ký, KHÔNG phải điểm số.**

---

## Section 4 — Module Structure Audit

**Insight cốt lõi: cấu trúc đánh giá khác nhau hoàn toàn giữa các module. Không thể so sánh raw score cross-module.**

| Module | CMA count | Exam weight | TMA weight | Đặc điểm |
|--------|-----------|-------------|------------|----------|
| AAA | 0 | 200 | 200 | Không có CMA |
| BBB | 15 | 400 | 385 | Nặng TMA |
| CCC | 8 | 400 | 150 | CMA chiếm tỉ lệ đáng kể |
| DDD | 7 | 400 | 375 | Nặng TMA |
| EEE | 0 | 300 | 300 | Không có CMA |
| FFF | 28 | 400 | 400 | CMA nhiều nhất |
| GGG | 18 | 300 | **0** | TMA không tính điểm |

**GGG đặc biệt:** TMA weight = 0 → feature weighted_score từ TMA vô nghĩa với module này.

**Hướng xử lý:** Không train separate model per module (chỉ có 7 modules, AAA chỉ 1 presentation). Thay vào đó:
1. Thêm `code_module` làm categorical feature — để model tự học distribution shift
2. Tính `weighted_score` đúng cách thay vì `mean(score)`
3. Normalize submission rate: `n_submitted / n_safe_available_in_module` thay vì đếm tuyệt đối

---

## Section 5 — Presentation Effect (B vs J)

| Semester | Pass rate | n_students |
|----------|-----------|-----------|
| B (Feb) | 43.67% | 12,488 |
| J (Oct) | 49.40% | 20,105 |

**Chênh lệch per module:**

| Module | Diff (B - J) | Vượt ngưỡng ±5pp? |
|--------|-------------|-------------------|
| CCC | -6.3pp | ✅ |
| EEE | -6.3pp | ✅ |
| DDD | -4.2pp | Gần |
| BBB | -3.8pp | Không |
| GGG | -3.5pp | Không |
| FFF | -1.5pp | Không |

→ `semester` phải là feature vì effect không đồng đều giữa modules.

```python
df['semester'] = df['code_presentation'].str[-1]          # 'B' hoặc 'J'
df['module_semester'] = df['code_module'] + '_' + df['semester']  # interaction
```

---

## Section 7 — Leakage Audit (cut-off = day 135)

**Cut-off day 135** = capture 80% Withdrawn, còn ~126 ngày để can thiệp.

### SAFE vs LEAKAGE tại day 135

| Type | SAFE (≤135) | LEAKAGE (>135) | Quyết định |
|------|-------------|-----------------|-----------|
| TMA | **71** | 35 | ✅ Dùng — majority safe |
| CMA | 17 | **59** | ⚠️ Dùng thận trọng — per module |
| Exam | 0 | **6** | ❌ Không dùng tuyệt đối |

### Submit ratio (submit_day / module_length)

| Type | Median | Ý nghĩa |
|------|--------|---------|
| TMA | 0.353 | Nộp sớm, phần lớn trước cut-off |
| CMA | 0.577 | Nộp giữa-cuối module, nhiều bài sau cut-off |
| Exam | 0.927 | Luôn cuối khóa |

### Lưu ý quan trọng về CMA

CMA phân bổ không đều giữa modules:
- FFF: 28 CMA tổng → có nhiều safe hơn
- AAA, EEE: 0 CMA → không có gì để dùng

```python
# SAI — đếm tuyệt đối
n_cma_submitted = 3

# ĐÚNG — tỉ lệ so với available trong module
cma_submission_rate = n_submitted / n_safe_cma_available_in_module
# AAA/EEE → fillna(0) hoặc -1
```

---

## Tổng hợp Feature Engineering

### DROP — không bàn cãi

| Cột/Data | Lý do |
|----------|-------|
| `date_unregistration` | 99.1% Withdrawn có giá trị này → proxy của label → leakage 100% |
| Exam score | 0 safe assessment, median submit day 0.927 → leakage |
| Assessment `date > 135` | Temporal leakage |
| Assessment `date` null | Không phân loại được safe/leak → drop |
| `week_from`, `week_to` | 82.4% null, không dùng trong pipeline |
| `sum_click` tổng | Không phân biệt "burst rồi biến mất" vs "đều đặn" |
| `date_registration` null (45 rows) | Tỉ lệ nhỏ, drop được |

### GIỮ NGUYÊN

| Feature | Signal | Ghi chú |
|---------|--------|---------|
| `num_of_prev_attempts` | Mạnh | Monotonically negative |
| `code_module` | Quan trọng | Categorical, để model học distribution shift |
| `highest_education` | Trung bình | Encode ordinal |
| `imd_band` | Trung bình | Null → "Unknown" category |
| `age_band` | Trung bình | Encode ordinal |
| `studied_credits` | Yếu | Giữ, không kỳ vọng nhiều |
| `gender` | Rất yếu | Có thể drop |

### TẠO MỚI

**Từ assessment (chỉ SAFE ≤ day 135):**
```python
weighted_score_before_cutoff  = sum(score × weight) / sum(weight)
tma_submission_rate           = n_tma_submitted / n_safe_tma_in_module
cma_submission_rate           = n_cma_submitted / n_safe_cma_in_module  # 0 nếu module không có CMA
n_missing_submission          = số bài không nộp (score ban đầu là null)
```

**Từ VLE (chỉ date ≤ day 135):**
```python
total_clicks_before_cutoff    # tổng click
days_active_before_cutoff     # số ngày có ít nhất 1 click
click_per_active_day          # intensity = total / days_active
last_active_day               # recency
clicks_oucontent              # nội dung chính — quan trọng nhất
clicks_forumng                # forum — tương quan với pass rate
clicks_resource
clicks_quiz
```

**Từ registration & presentation:**
```python
semester                      = code_presentation.str[-1]       # 'B' hoặc 'J'
module_semester               = code_module + '_' + semester     # interaction
days_since_registration       = 135 - date_registration         # engagement từ đầu
```

---

## Checklist trước khi modeling


- [ ] Kiểm tra `n_safe_cma_per_module` để normalize đúng
- [ ] Stratified split theo `final_result` vì Distinction chỉ 9.3%
- [ ] AAA chỉ 1 presentation — cẩn thận khi split
- [ ] Set class weight trong model để handle imbalance

---

*Tổng hợp từ EDA*
