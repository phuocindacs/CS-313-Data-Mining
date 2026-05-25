# Hướng Dẫn Thuyết Trình — OULAD Early Warning System Demo

**Dành cho:** Web developer thuyết trình demo (không phải data scientist)
**Mục đích:** Hiểu + giải thích webapp dễ hiểu, chính xác, tự tin trả lời thầy
**Ngày:** 2026-05-23

> **Cách dùng nhanh:**
> - **PHẦN 1** — nền tảng (data, model, từng trang web). Đọc để hiểu tổng thể.
> - **PHẦN 2** — 7 câu hỏi KHÓ của thầy + giải thích sâu. Đây là phần quan trọng nhất.
> - **PHẦN 3** — câu hỏi nhanh khác + cách trả lời.
> - **PHẦN 4** — script thuyết trình, flow, glossary.

---
---

# PHẦN 1 — NỀN TẢNG

## 1.1. Bộ dữ liệu (Dataset)

### OULAD là gì?
- **OULAD** = Open University Learning Analytics Dataset
- Bộ dữ liệu thực tế từ Open University (Anh) về hành vi + kết quả học tập của ~32,000 sinh viên
- Gồm: tương tác trực tuyến (clicks), điểm số, nộp bài, hoạt động theo tuần, thông tin nhân khẩu (vùng, học vấn…)

### "At-Risk" (Nguy cơ bỏ học) là gì?
- **At-Risk** = sinh viên có nguy cơ bỏ học / không hoàn thành khóa trong kì
- **Dấu hiệu:** tương tác giảm, không nộp bài, lâu không đăng nhập, điểm thấp
- **Not At-Risk** = sinh viên tiếp tục học bình thường

### Tập test dùng trong demo
- **11,260 sinh viên** (từ tập test, không phải toàn bộ dataset)
- Dữ liệu theo tuần, dùng các tuần cutoff **4 → 24**
- Tổng **281,500 dòng** (mỗi sinh viên × mỗi tuần = 1 dòng)

**Tại sao chỉ test data?**
→ Test set để **đánh giá** model trên dữ liệu chưa từng thấy. Train set dùng để train trước đó (notebook khác). Nếu đánh giá trên train set sẽ bị overfitting (số đẹp giả tạo).

---

## 1.2. Các Model trong Demo

### ML Models (Machine Learning) — nhìn "snapshot"
1. **XGBoost** — gradient boosting (cây quyết định). Nhanh, dễ giải thích, chống overfitting tốt.
   - Train **riêng từng tuần** (week 4, 8, 12, 16, 20, 24) → **6 models**
2. **LightGBM** — gradient boosting nhẹ hơn, hiệu suất cao với data lớn.
   - Cũng **6 models** (1 cho mỗi tuần)

### DL Models (Deep Learning) — nhìn "cả chuỗi thời gian"
3. **LSTM** — recurrent neural network, đọc chuỗi tuần liên tiếp. **1 model duy nhất.**
4. **Transformer** — attention-based neural network, xử lý chuỗi nhanh hơn. **1 model duy nhất.**

**Tóm lại con số:**
- **4 models** hiển thị trên web
- Backend thật: **12 ML sub-models** (6 XGBoost + 6 LightGBM) + **2 DL models** = 14 file model

---

## 1.3. Giải thích từng trang Web

### 📈 Risk Timeline — Đường cong dự đoán theo tuần
- Chọn 1 sinh viên → xem **xác suất at-risk qua các tuần** (4 → 24), 4 đường cong:
  - **XGBoost** (xanh đậm, nét liền) · **LightGBM** (vàng, nét liền)
  - **LSTM** (xanh lá, nét đứt) · **Transformer** (đỏ, nét đứt)
- **Trục X (ngang):** tuần cutoff. "Cutoff week 8" = "dừng dữ liệu ở week 8 rồi dự đoán"
- **Trục Y (dọc):** xác suất at-risk (0% → 100%)
- **Threshold 0.5** (nét đứt ngang): xác suất ≥ 50% → coi là At-Risk
- **Lưu ý màu:** màu chỉ để **phân biệt 4 model**, KHÔNG thể hiện mức risk. Mức risk = vị trí trên trục Y.

```
Ví dụ Sinh viên #123 (thực tế At-Risk):
- Week 4 : ~30% (quá sớm, signal yếu)
- Week 12: ~75% (signal rõ hơn)
- Week 24: ~95% (gần chắc chắn, cuối khóa)
```

### 🔍 Week Detail — Chi tiết tại 1 tuần cụ thể
- Chọn 1 sinh viên + 1 tuần → xem **4 dự đoán side-by-side**
- Mỗi model: **xác suất %** + **badge** (đỏ = At-Risk / xanh = Not At-Risk) + **progress bar**
- Dùng để xem snapshot cụ thể, dễ nói về việc can thiệp sớm.
- **Badge ở đây:** Đỏ = At-Risk (cảnh báo), Xanh = Not At-Risk (an toàn).

### 📊 Explainability — Tại sao model dự đoán vậy?
- **ML (SHAP):** biểu đồ waterfall — mỗi feature đẩy dự đoán lên/xuống.
- **DL (Integrated Gradients):** biểu đồ cột theo tuần — tuần nào ảnh hưởng nhiều nhất.
- 👉 **Cách đọc chi tiết: xem PHẦN 2 — Câu 5 & 6.**

### 📋 Model Metrics — Độ chính xác của model
- **Thống kê tập test:** tổng SV (~11,260), số At-Risk / Not At-Risk, tỉ lệ At-Risk (~30%, imbalanced)
- **Bảng metrics mỗi model:**
  - **AUC** — khả năng phân biệt (0→1, càng cao càng tốt; 0.5 = random; điển hình 0.75–0.95)
  - **Precision** — "model nói at-risk thì bao % đúng?"
  - **Recall** — "bắt được bao % sinh viên thực sự at-risk?"
  - **F1** — cân bằng Precision & Recall (hữu ích khi data imbalanced)

```
Ví dụ XGBoost Week 12:
- AUC 0.82       → tốt
- Precision 78%  → nói at-risk thì 78% đúng (22% false positive)
- Recall 71%     → bắt được 71% at-risk (29% miss = false negative)
```

---
---

# PHẦN 2 — 7 CÂU HỎI KHÓ CỦA THẦY (GIẢI THÍCH SÂU)

## ❓ Câu 1: Tại sao 12 model ML? Sao thầy không thấy trên web?

### Tóm tắt
- **Backend có 12 models:** 6 XGBoost + 6 LightGBM (mỗi tuần 1)
- **Web chỉ show 4:** XGBoost, LightGBM, LSTM, Transformer
- **Lý do:** giữ giao diện đơn giản, không overwhelm người dùng

### Chi tiết
**Tại sao cần 12 models?** Vì **dấu hiệu at-risk thay đổi theo tuần:**
```
Week 4 : data ít, signal yếu  → cần model "nhạy" với pattern tuần 4
Week 12: data nhiều, signal rõ → model tuần 12 khác model tuần 4
Week 20: pattern rất rõ        → model detect dễ
```
Nên train riêng: `xgb_model_week_4.pkl`, `..._week_8.pkl`, `..._week_12.pkl`, … (tương tự LightGBM).

**Tại sao web chỉ show 4?** Show 12 cái thì sidebar quá rối. Web "abstract" lại:
```
User chọn: Student 123, Week 12
Backend tự động:
  - XGBoost  → load xgb_model_week_12.pkl
  - LightGBM → load lgbm_model_week_12.pkl
  - LSTM / Transformer → masked tới week 12
→ Hiển thị 4 dự đoán. User không cần biết backend có 12 models.
```

**Nói với thầy:**
> "Chúng em train 12 model ML riêng, mỗi cái cho 1 tuần cắt, vì dấu hiệu at-risk thay đổi theo thời gian. Nhưng web chỉ hiện 'XGBoost' chung chung — người dùng chỉ quan tâm dự đoán, không cần biết backend có bao nhiêu model. Giống dùng Google Search mà không cần biết có 1000 service phía sau."

---

## ❓ Câu 2: Tại sao dự đoán "At-Risk" mà actual là "Not At-Risk"?

### Tóm tắt
- Đây là **False Positive** — lỗi bình thường của mọi model. Không model nào 100% đúng.
- Là trade-off giữa **Precision & Recall**.

### Chi tiết — 4 loại kết quả
```
                     │ Actual At-Risk │ Actual Not-Risk
─────────────────────┼────────────────┼─────────────────
Predict At-Risk      │ TP ✓           │ FP ✗ (cảnh báo nhầm)
Predict Not At-Risk  │ FN ✗ (miss)    │ TN ✓
```

**Tại sao sai?**
| Nguyên nhân | Ví dụ |
|---|---|
| Không đủ signal | SV bận việc tuần 8 → tương tác ít → flag nhầm; tuần 12 lại bình thường |
| Pattern giống nhưng kết quả khác | Hành vi giống at-risk nhưng lý do khác → model học pattern, đời thực có ngoại lệ |
| Threshold là ranh giới mờ | Model đoán 52% → flag (ngưỡng 50%), nhưng thực ra ~48% (close call) |
| Data thiếu thông tin | Model không biết SV có việc làm / gia đình hỗ trợ |

**Trade-off:**
```
Precision cao (ngưỡng 0.9): ít cảnh báo nhầm, NHƯNG miss nhiều at-risk
Recall cao    (ngưỡng 0.3): bắt nhiều at-risk, NHƯNG cảnh báo nhầm nhiều
```
**Early warning → ưu tiên Recall** vì miss SV bỏ học tệ hơn cảnh báo nhầm.

**Nói với thầy:**
> "Model không thể 100% chính xác. False Positive (cảnh báo nhầm) thì có thể follow-up thêm; False Negative (miss SV at-risk) mới tệ. Nên hệ thống ưu tiên giảm miss → Recall cao hơn Precision."

---

## ❓ Câu 3: Tại sao chia nhiều tuần (4,8,12,16,20,24)? Dự đoán khác nhau sao?

### Tóm tắt
- Mỗi tuần = một **checkpoint** trong khóa. Signal at-risk **thay đổi theo thời gian** (sớm yếu, muộn rõ).
- Mỗi tuần dự đoán khác vì **dữ liệu khác**.

### Chi tiết — tại sao là các tuần đó?
```
Khóa ~32 tuần:
Week 1-3 : QUÁ SỚM (data sparse) → bỏ
Week 4   : checkpoint đầu, detect siêu sớm (precision thấp)
Week 8   : pattern bắt đầu rõ
Week 12  : giữa khóa — cân bằng (đủ sớm để can thiệp, đủ rõ để chính xác)
Week 16  : signal rất rõ
Week 20  : gần cuối — cơ hội cuối để giúp
Week 24  : muộn nhưng vẫn hữu ích cho khóa sau
Week 25+ : QUÁ MUỘN → bỏ
```

**Tại sao dự đoán khác nhau ở mỗi tuần?**
```
Sinh viên X:
  Week 4 : {login=3, submit=2, score=65} → "bình thường" → 30% at-risk
  Week 12: {login=3, submit=0, score=45} → "có vấn đề"  → 75% at-risk
Lý do: dữ liệu khác + model tuần khác → dự đoán khác.
"Không nộp bài" là signal mạnh ở week 12, nhưng ở week 4 chưa rõ.
```

**Có model nào dự đoán "flexible" theo tuần không?**
```
ML : ❌ Mỗi model chỉ cho 1 tuần (week 4 model không dùng cho week 12)
DL : ✅ 1 model duy nhất, chỉ đổi input sequence theo tuần (xem Câu 7)
```

**Nói với thầy:**
> "At-risk là một quá trình theo thời gian. Tuần 4 dấu hiệu yếu, tuần 12 rõ hơn (không nộp bài, login ít), tuần 24 thì gần chắc chắn. Nên train riêng từng tuần để bắt đúng pattern từng giai đoạn."

---

## ❓ Câu 4: DL khác ML sao? Tại sao chia làm 2 loại?

### Tóm tắt
- **ML:** nhìn **snapshot** tại 1 tuần. **DL:** nhìn **cả trajectory** từ week 1 → cutoff.
- Cùng 1 SV nhưng góc nhìn khác → dự đoán khác.

### Chi tiết — ví dụ
```
Sinh viên X tương tác: W4=10, W8=9, W12=3, W16=1, W20=0

ML (XGBoost Week 12): chỉ nhìn snapshot tuần 12 {interactions=3,...}
  → "tương tác ít → 60% at-risk"
  ❌ KHÔNG thấy: từ 10 giảm xuống 3 (trend nguy hiểm)

DL (LSTM): nhìn cả chuỗi [10, 9, 3, ...]
  → "engagement đang DECLINING! → 85% at-risk"
  ✅ Thấy trajectory (đường đi)
```

| | ML (XGBoost, LightGBM) | DL (LSTM, Transformer) |
|---|---|---|
| Input | features tại week X | chuỗi nhiều tuần |
| Cách nhìn | snapshot (chụp hình) | video (cả quá trình) |
| Mạnh | nhanh, dễ giải thích (SHAP), pattern theo tuần | bắt được trend dài hạn |
| Yếu | không thấy trend | chậm, khó giải thích (black box) |
| Số lượng | 6 + 6 = 12 models | 1 + 1 = 2 models |

**Nói với thầy:**
> "ML giống bác sĩ đo mạch tại thời điểm hiện tại — nhanh, rõ. DL giống bác sĩ theo dõi cả quá trình bệnh nhân suy yếu dần — bắt được xu hướng. Dùng cả 2 vì nhìn 2 khía cạnh khác nhau."

---

## ❓ Câu 5: SHAP là gì? Đọc đúng biểu đồ (số bên trái, f(x), màu sắc)

> ⚠️ **2 điểm dễ hiểu sai** (đã kiểm chứng từ code `shap.plots.waterfall`):
> - Số bên trái dấu `=` **KHÔNG phải ID** → là **GIÁ TRỊ của feature** cho SV đó.
> - Màu **ĐỎ = TĂNG** at-risk; màu **XANH = GIẢM** at-risk (an toàn hơn).

### Tóm tắt cách đọc
- **SHAP** = giải thích mỗi feature đẩy dự đoán lên/xuống bao nhiêu.
- **Số bên TRÁI dấu `=`** = giá trị feature của SV này (vd `57 = mean_score_so_far` → điểm TB = 57).
- **Thanh ĐỎ (số +)** = đẩy LÊN → tăng At-Risk (mũi tên sang phải).
- **Thanh XANH (số −)** = đẩy XUỐNG → tăng Not At-Risk / an toàn (mũi tên sang trái).
- **Độ DÀI thanh** = mức độ ảnh hưởng (dài = quan trọng).
- **f(x)** = điểm thô cuối cùng cho SV này (dạng log-odds).

### Đọc từng phần (screenshot Week 20, Student 6516, XGBoost)
| Em thấy trên web | Nghĩa là gì |
|---|---|
| `57 = mean_score_so_far` | feature `mean_score_so_far` có **giá trị = 57** (điểm TB tích lũy đến tuần 20) |
| `−0 = days_since_last_active` | giá trị ≈ 0 → SV **vừa mới hoạt động** (0 ngày kể từ lần active cuối) |
| `1 = tma_submission_rate_so_far` | tỉ lệ nộp bài TMA = 1 = **100%** |
| `2 = avg_days_before_deadline_so_far` | TB nộp trước hạn 2 ngày |
| `+0.91` (thanh đỏ) | feature đó **đẩy +0.91 về phía At-Risk** |
| `−0.91` (thanh xanh) | feature đó **đẩy −0.91 về phía an toàn** |
| `f(x) = −1.411` | điểm thô: âm → xác suất thấp → **Not At-Risk** |

### ❗ Tại sao số bên trái đổi khi đổi tuần (54 → 57)?
Đây là **bằng chứng nó là GIÁ TRỊ, không phải ID:**
```
Hầu hết feature có đuôi "_so_far" = tích lũy / trung bình TÍNH ĐẾN tuần cutoff.
mean_score_so_far:
  Week 12 → điểm TB tuần 1→12 = 54
  Week 20 → điểm TB tuần 1→20 = 57   (SV này điểm cải thiện dần)
Đổi tuần → thêm dữ liệu → giá trị đổi. Nếu là ID thì KHÔNG BAO GIỜ đổi.
```

### f(x) và E[f(x)]
```
f(x) = điểm thô của model cho riêng SV này, dạng log-odds (KHÔNG phải % trực tiếp).
  Đổi sang %: probability = sigmoid(f(x))
    f(x) = −1.411 → ~20% → Not At-Risk
    f(x) =  0.098 → ~52% → hơi nghiêng At-Risk (sát ranh giới)
  Ranh giới quyết định: f(x) = 0  ⇔  prob = 50%.
    f(x) > 0 → nghiêng At-Risk ;  f(x) < 0 → nghiêng an toàn.

E[f(x)] = base value = điểm "mặc định" (TB f(x) trên toàn bộ data, vd 0.139)
  = điểm XUẤT PHÁT của biểu đồ.
```

> ⚠️ **Phân biệt 2 thứ dễ nhầm:**
> - **f(x)** = DỰ ĐOÁN của model cho SV này.
> - Badge **"Actual: ..."** = nhãn THẬT (ground truth), KHÔNG phải dự đoán.
> - Hai cái có thể KHÁC nhau → đó chính là lúc model đoán sai (xem ví dụ dưới).

### Cơ chế cộng dồn (waterfall)
```
Bắt đầu: E[f(x)]  → + đỏ (đẩy lên) / − xanh (đẩy xuống) → … → Kết thúc: f(x)
Tổng tất cả đóng góp = f(x) − E[f(x)]
```

### 🔍 Đọc thử screenshot THẬT (Week 12, XGBoost, SV 6516)
```
E[f(X)] = 0.139  (góc dưới phải) = điểm trung bình mọi SV (xuất phát)
f(x)    = 0.098  (góc trên phải) = điểm riêng SV 6516 (kết thúc)

Cộng dồn từ trên xuống:
  0.139  (baseline)
  +0.30  mean_score_so_far = 54     🔴 kéo LÊN (về at-risk) — mạnh nhất
  −0.24  tma_submission_rate = 1    🔵 nộp đủ 100% → kéo XUỐNG (an toàn)
  −0.08  days_since_last_active = 3  🔵 an toàn
  −0.05  avg_days_before_deadline   🔵 nộp sớm → an toàn
  +0.03  weighted_score = 52        🔴 nhẹ
  ... (28 feature còn lại ~0)
  = 0.098  (f(x))

Đọc: đỏ và xanh gần triệt tiêu, f(x)=0.098 < baseline 0.139
     → SV rủi ro DƯỚI trung bình.
     Đổi %: sigmoid(0.098) ≈ 52% → model hơi nghiêng At-Risk (gần 50/50).
     Nhưng badge "Actual: Not At-Risk" → nhãn thật là an toàn.
     ⇒ Week 12 model đoán nhầm nhẹ (false positive sát ranh giới).
```

**Nối với Week 20 (cùng SV):** f(x) = −1.411 → ~20% → Not At-Risk → **ĐÚNG**.
> 💡 Câu chuyện đẹp để kể: "Week 12 model còn lưỡng lự (52%, nhầm nhẹ). Đến Week 20 có thêm dữ liệu → model tự tin SV an toàn (20%) và đoán đúng. Càng nhiều dữ liệu, model càng chính xác."

### ⚠️ Đừng over-interpret 1 feature lẻ (kẻo bị thầy bắt bí)
Ví dụ Week 20: `57 = mean_score_so_far` lại **đỏ (+0.91, tăng risk)** — nghe nghịch lý (điểm cao mà tăng risk?).
- SHAP đo **so với mức trung bình** E[f(x)], không phải so với 0.
- Điểm 57/100 thực ra **không cao** → so với SV "an toàn điển hình" thì hơi thấp → đẩy nhẹ về at-risk.
- Các feature **tương tác lẫn nhau** trong cây → hướng 1 feature có thể phản trực giác.
- **Quan trọng:** tổng tất cả → f(x). Ở đây thanh xanh (vừa active, nộp đủ bài) thắng → f(x) âm → **Not At-Risk** (khớp Actual ✓).

**Câu trả lời an toàn cho thầy:**
> "SHAP cho thấy mỗi yếu tố đẩy dự đoán lên/xuống so với mức trung bình. Một yếu tố lẻ có thể phản trực giác vì các yếu tố tương tác với nhau; điều quyết định là TỔNG tất cả. Ở SV này, tổng nghiêng về an toàn nên kết quả Not At-Risk."

---

## ❓ Câu 6: "Importance" là gì? Tại sao SHAP cho ML, Integrated Gradients cho DL?

### "Importance" (độ quan trọng) là gì?
```
Importance của 1 feature = nó ảnh hưởng MẠNH hay YẾU đến dự đoán.
Trên biểu đồ SHAP = ĐỘ DÀI thanh = |giá trị SHAP|.

LƯU Ý: importance KHÔNG nói "tốt/xấu", chỉ nói "mạnh/yếu".
  → Hướng tốt/xấu = MÀU (đỏ=rủi ro / xanh=an toàn)
  → Độ mạnh       = ĐỘ DÀI thanh
```
| Loại | Nghĩa | Ở đâu |
|---|---|---|
| **Local** | feature quan trọng cho **1 SV cụ thể** | SHAP waterfall (thanh dài nhất) |
| **Global** | feature quan trọng **nói chung** | TB \|SHAP\| qua tất cả SV |

Ví dụ Week 20, SV 6516: `days_since_last_active` & `mean_score_so_far` (~|0.91|) quan trọng nhất.

### Biểu đồ DL (Integrated Gradients) — đọc screenshot LSTM Week 12
```
Trục X = tuần (W1 → W12);  Trục Y = "Total |attribution|" = mức ảnh hưởng của tuần đó
Cột thấp ở W1 (~0.02) → tăng dần → cao vọt ở W11, W12 (~0.48-0.49)
→ Model LSTM dựa NHIỀU NHẤT vào hành vi các tuần GẦN ĐÂY, ít quan tâm tuần đầu.
```
**3 lưu ý quan trọng (kẻo bị thầy bắt bí):**
1. **Màu xanh lá KHÔNG có nghĩa** — chỉ là màu thanh. (Khác SHAP: SHAP đỏ/xanh = hướng.)
2. Đây là **|attribution|** (giá trị tuyệt đối) → chỉ cho biết **MỨC ĐỘ ảnh hưởng**, KHÔNG cho biết hướng (về at-risk hay an toàn).
3. Khác SHAP (importance theo **feature**), DL cho importance theo **tuần** — vì DL xử lý chuỗi thời gian.

### Tại sao 2 kỹ thuật khác nhau?
| | SHAP (ML) | Integrated Gradients (DL) |
|---|---|---|
| Dùng cho | XGBoost, LightGBM (cây) | LSTM, Transformer (mạng neural) |
| Cách tính | đi qua cấu trúc cây → chính xác | dùng đạo hàm (gradient) baseline → SV |
| Tốc độ | nhanh | chậm hơn |
| Vì sao hợp | cây rời rạc → khai thác cấu trúc cây | mạng neural liên tục → có gradient để đo |

(SHAP về lý thuyết dùng được cho DL nhưng chậm & kém ổn định — code ghi chú DeepExplainer không đáng tin cho demo. Nên DL dùng Integrated Gradients.)

**Nói với thầy:**
> "Importance = mức độ ảnh hưởng của một yếu tố — trên biểu đồ là độ dài thanh; màu là hướng. Em dùng 2 công cụ vì 2 loại model khác nhau: SHAP cho model cây (chính xác, nhanh), Integrated Gradients cho mạng neural (dùng gradient). ML chỉ ra FEATURE nào quan trọng; DL chỉ ra TUẦN nào quan trọng."

---

## ❓ Câu 7: Đổi tuần → tại sao cả 4 model đổi? "Sequence masked to week" là gì?

### Tóm tắt
- **ML:** đổi tuần → load model khác + features khác → dự đoán khác.
- **DL:** đổi tuần → **cùng model**, nhưng input sequence khác (mask) → dự đoán khác.
- **Sequence masked:** che các tuần > cutoff, chỉ cho model nhìn weeks 1 → cutoff.

### ML — dễ hiểu
```
Week 4 : load xgb_week_4.pkl  + data tuần 4  → 30% at-risk
Week 12: load xgb_week_12.pkl + data tuần 12 → 75% at-risk
Model khác + features khác = output khác.
```

### DL — khó hiểu (cùng 1 model, đổi input)
```
LSTM, Week 4 cutoff:           LSTM, Week 12 cutoff:
  W1-4 : [real data]             W1-12 : [real data]
  W5-25: [0,0,...] PADDED        W13-25: [0,0,...] PADDED
  mask = [1,1,1,1,0,...,0]       mask = [1×12, 0,...,0]
  → Prediction A                 → Prediction B (khác)
```
**Nguyên lý:** LSTM/Transformer đọc chuỗi từ trái qua phải, chỉ xử lý tuần có `mask=1`, bỏ qua `mask=0`. Cùng model nhưng "thấy" dữ liệu khác → output khác.

**"Sequence masked to week" =** nói với model "chỉ nhìn weeks 1→cutoff, ignore phần sau".
```python
# Pseudocode
cutoff_week = 12
mask = [1]*cutoff_week + [0]*(25 - cutoff_week)   # [1×12, 0×13]
output = model(input_sequence, attention_mask=mask)  # chỉ attend tuần 1-12
```

```
Ví dụ Student X engagement giảm dần [10,9,8,7,6,5,4,3,2,1,0.5,0.3]:
  LSTM Week 4 : thấy [10,9,8,7] → "declining nhẹ" → 40% at-risk
  LSTM Week 12: thấy cả 12 tuần → "declining mạnh!" → 90% at-risk
Cùng 1 model, dữ liệu thấy khác → dự đoán khác.
```

**Nói với thầy:**
> "Đổi tuần thì dữ liệu thay đổi. ML load model khác + data khác. DL dùng cùng 1 model nhưng 'che' các tuần sau cutoff — week 4 chỉ thấy tuần 1-4, week 12 thấy tuần 1-12. Cùng model, input khác → dự đoán khác."

---
---

# PHẦN 3 — CÂU HỎI NHANH KHÁC

### ❓ "Data lấy từ đâu? Có ý nghĩa không?"
> "OULAD — bộ dữ liệu công khai, thực tế từ Open University Anh, hành vi học tập của ~32,000 SV. Nhiều nghiên cứu đã dùng nó để xây model dự đoán."
> *Nếu hỏi dùng cho trường mình:* "Có thể dùng pattern, nhưng cần retrain với data của trường để hợp behavior SV địa phương."

### ❓ "Tại sao biểu đồ lên xuống, không smooth?"
> "Mỗi tuần ML dùng model riêng + data riêng nên không nhất thiết liên tục. Quan trọng hơn: dấu hiệu at-risk thật sự thay đổi theo thời gian (sớm yếu, muộn rõ)."
> *Smooth hơn?* "DL (LSTM/Transformer) nhìn cả chuỗi nên thường smooth hơn."

### ❓ "Màu sắc nghĩa là gì?"
> **Risk Timeline:** màu chỉ để phân biệt 4 model (xanh đậm=XGBoost, vàng=LightGBM, xanh lá=LSTM, đỏ=Transformer). Mức risk = vị trí trục Y, không phải màu.
> **Week Detail badge:** đỏ = At-Risk, xanh = Not At-Risk.
> **SHAP waterfall:** đỏ = đẩy về at-risk, xanh = đẩy về an toàn (xem Câu 5).

### ❓ "Threshold 0.5 là gì? Tại sao 0.5?"
> "Ngưỡng quyết định: xác suất ≥ 50% → At-Risk, < 50% → Not At-Risk. 0.5 là mặc định cân bằng. Đời thực có thể dùng 0.3 (thận trọng hơn) hay 0.7 (chỉ alert ca rất rõ)."

### ❓ "AUC, F1, Precision, Recall — cái nào quan trọng?"
| Metric | Ý nghĩa |
|---|---|
| AUC | khả năng phân biệt (0→1, cao tốt) |
| Precision | "nói at-risk thì bao % đúng" |
| Recall | "bắt được bao % at-risk thực sự" |
| F1 | cân bằng Precision & Recall |
> "Ở đây ưu tiên **Recall** — không miss SV at-risk quan trọng hơn tránh cảnh báo nhầm (miss → SV bỏ học; nhầm → chỉ cần follow-up)."

### ❓ "Tại sao 4 model? Cái nào tốt nhất?"
> "Mỗi cái có ưu điểm: ML nhanh, dễ giải thích, train theo tuần; DL bắt xu hướng dài hạn. Không có 'tốt nhất' — có thể ensemble (kết hợp) để lấy ưu điểm cả 4."

### ❓ "Tập test có bao nhiêu At-Risk?"
> "~11,260 SV, khoảng 29-31% at-risk (~3,300), còn lại Not At-Risk. Tỉ lệ imbalanced — thực tế của hầu hết dataset."

### ❓ "Tại sao chỉ có test set, không có train set?"
> "Test set để đánh giá trên data chưa từng thấy. Train set đã dùng để train trước; nếu đánh giá trên train sẽ overfitting (số đẹp giả). Demo hiện kết quả test → đáng tin hơn."

---
---

# PHẦN 4 — CHUẨN BỊ THUYẾT TRÌNH

## 4.1. Flow gợi ý (~8-10 phút)
1. **Intro (1-2′):** OULAD là gì, At-Risk là gì, tại sao cần dự đoán.
2. **Risk Timeline (2-3′):** chọn 1-2 SV, giải thích 4 đường cong, threshold, tại sao lên xuống.
3. **Week Detail (1-2′):** cùng SV, chọn week 12, 4 dự đoán side-by-side, agree/disagree.
4. **Explainability (1-2′):** mở SHAP, đọc số bên trái = giá trị feature, màu đỏ/xanh, f(x).
5. **Model Metrics (1′):** AUC/F1 → model tốt; nêu tỉ lệ At-Risk.
6. **Q&A (2-3′):** dùng PHẦN 2 & 3.

## 4.2. Script mẫu

**Mở đầu:**
> "Thầy và các bạn, nhóm em xây dựng hệ thống Early Warning dự đoán sinh viên có nguy cơ bỏ học (at-risk) trong khóa online. Dữ liệu là OULAD từ Open University Anh, ~11,000 SV test. Mục tiêu: nếu dừng dữ liệu ở tuần X, có dự đoán được SV sẽ bỏ học không? Em dùng 4 model: XGBoost, LightGBM (ML) và LSTM, Transformer (Deep Learning) để nhìn vấn đề từ 2 góc độ."

**Demo Risk Timeline:**
> "Đây là 1 sinh viên với 4 đường cong dự đoán qua các tuần. Week 4 ~30% (signal yếu), Week 12 ~75% (rõ hơn), Week 24 ~95% (gần chắc chắn). Nó lên xuống vì dữ liệu thay đổi — tuần 4 SV còn ổn, đến tuần 12 không nộp bài + tương tác ít nên signal rõ."

**Demo Week Detail:**
> "Ở week 12, cả 4 model đều cao (~70-80% at-risk) — chúng đồng thuận nên ta tự tin hơn."

**Demo SHAP (⚠️ đã sửa đúng màu):**
> "SHAP giải thích vì sao model ra kết quả này. Số bên trái dấu '=' là GIÁ TRỊ của đặc trưng cho SV này (vd mean_score = 57); nó đổi theo tuần vì là giá trị tích lũy. Thanh ĐỎ = đẩy về at-risk, thanh XANH = đẩy về an toàn, thanh càng dài càng quan trọng. f(x) là điểm thô cuối — âm là an toàn, dương là at-risk. Ví dụ SV này 'vừa mới active' và 'nộp bài đủ' (thanh xanh) kéo về an toàn → kết quả Not At-Risk."

**Đóng:**
> "Tóm lại: (1) 4 model ML + DL nhìn 2 góc độ; (2) dự đoán ở 6 tuần cắt vì signal thay đổi theo thời gian; (3) giải thích được bằng SHAP / Integrated Gradients. Metrics: AUC ~0.82, Precision ~78%, Recall ~71%. Cảm ơn thầy!"

## 4.3. Nên / Không nên
**Nên:** tập trung ý nghĩa data; chỉ cụ thể trên biểu đồ; giải thích *tại sao* model khác nhau; nêu use case thực tế (alert sớm + can thiệp).
**Không nên:** sa đà vào code; giải thích math sâu của SHAP/attention; nói model nào "tốt nhất"; claim chính xác trên dữ liệu đời thực (chỉ đúng trên test set).

## 4.4. Bảng tóm tắt 7 câu khó
| # | Câu hỏi | Trả lời nhanh |
|---|---|---|
| 1 | 12 ML models? | Backend 12, web show 4 (abstract). Signal at-risk khác nhau theo tuần. |
| 2 | Dự đoán sai? | False positive — lỗi bình thường. Trade-off Precision/Recall, ưu tiên Recall. |
| 3 | Chia tuần? | Signal thay đổi theo thời gian (sớm yếu, muộn rõ). Train riêng từng tuần. |
| 4 | DL vs ML? | ML: snapshot tuần X. DL: cả trajectory 1→X. Góc nhìn khác → dự đoán khác. |
| 5 | SHAP? | Số trái dấu `=` là GIÁ TRỊ feature (không phải ID). Đỏ=tăng at-risk, xanh=an toàn. f(x)=log-odds. |
| 6 | SHAP vs IG? | SHAP cho cây (nhanh, chính xác). IG cho mạng neural (gradient). ML→feature, DL→tuần. |
| 7 | Đổi tuần? | ML: load model khác. DL: cùng model, mask input sequence khác. |

## 4.5. Glossary — "chữ" trong demo
| Từ | Ý nghĩa |
|---|---|
| Cutoff week | tuần dừng dữ liệu để dự đoán |
| At-Risk / Not At-Risk | nguy cơ bỏ học / an toàn (ngưỡng 50%) |
| Probability | xác suất at-risk (0-100%) |
| Threshold | ngưỡng quyết định (mặc định 0.5) |
| Feature | đặc trưng (vd `mean_score_so_far`, `days_since_last_active`) |
| `_so_far` | giá trị tích lũy/trung bình tính đến tuần cutoff |
| Importance | mức độ ảnh hưởng của feature (độ dài thanh SHAP) |
| SHAP | giải thích model ML (waterfall theo feature) |
| Integrated Gradients | giải thích model DL (theo tuần) |
| f(x) / E[f(x)] | điểm thô của SV / điểm trung bình (baseline) |
| log-odds → sigmoid | dạng điểm thô → đổi ra xác suất % |
| AUC / Precision / Recall / F1 | các chỉ số đánh giá model |

## 4.6. Bonus — nếu thầy hỏi "Cải thiện thế nào?"
- Thêm features (forum, chat, nhóm học)
- Ensemble ML + DL
- Real-time prediction (cập nhật mỗi tuần)
- Personalized threshold theo từng SV
- Intervention engine (gợi ý cách hỗ trợ SV at-risk)

---

**Chúc em thuyết trình thành công! 🎓**
