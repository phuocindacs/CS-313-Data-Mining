# CS-313-Data-Mining

# OULAD Early Warning System (EWS)

Hệ thống cảnh báo sớm (Early Warning System - EWS) hỗ trợ dự đoán nguy cơ học tập gặp rủi ro (`At-Risk` bao gồm Fail hoặc Withdrawn) của sinh viên dựa trên bộ dữ liệu **OULAD (Open University Learning Analytics Dataset)**. Hệ thống kết hợp phương pháp Machine Learning truyền thống theo snapshot tĩnh và Deep Learning xử lý chuỗi thời gian (Temporal Panel Data) nhằm tối ưu hóa độ chính xác và khả năng can thiệp sớm xuyên suốt học kỳ.

---

## 1. Kiến trúc Hệ thống (Dual-Branch Framework)

Hệ thống được thiết kế theo cấu trúc hai nhánh xử lý độc lập, tối ưu hóa cả dữ liệu tổng hợp tĩnh và chuỗi tương tác động:

### Nhánh 1: Machine Learning (Static Snapshot Engine)
* **Mô hình:** `XGBoost` và `LightGBM`.
* **Cơ chế:** Dữ liệu tương tác và điểm số được tổng hợp lũy kế thành dạng bảng phẳng (Wide Format) tại các snapshot cố định: Tuần 4, 8, 12, 16, 20, và 24. Mô hình chỉ sử dụng thông tin phát sinh từ ngày 0 đến ngày $7(t+1)-1$ tại tuần $t$ tương ứng để mô phỏng chính xác kịch bản dự báo thời gian thực và ngăn chặn rò rỉ dữ liệu (Data Leakage).
* **Diễn giải (XAI):** Tích hợp framework `SHAP` để bóc tách và định lượng đóng góp của các đặc trưng cụ thể (ví dụ: số lượt click tài nguyên, điểm số bài tập) vào xác suất rủi ro của từng học viên tại tuần được chọn.

### Nhánh 2: Deep Learning (Sequence Temporal Engine)
* **Mô hình:** `LSTM` và `Transformer` xây dựng trên nền tảng PyTorch.
* **Cơ chế:** Giữ nguyên cấu trúc chuỗi thời gian của hành vi học tập dưới dạng bảng tuần tuần hoàn (Long Format). Mô hình tiếp nhận song song cấu trúc chuỗi động (`X_seq` gồm 20 đặc trưng) và vector thông tin cố định (`X_static` gồm 20 đặc trưng) qua cơ chế che nhân quả (Causal Masking), đảm bảo tại tuần $t$ mô hình không nhìn thấy tương lai.
* **Diễn giải (XAI):** Sử dụng framework `Captum` (thuật toán Integrated Gradients) để tính toán điểm đóng góp của từng bước thời gian (Time-step Attribution), chỉ ra khoảng thời gian hành vi nào khiến sinh viên rơi vào nhóm nguy cơ cao.

---

## 2. Thiết kế Pipeline Dữ liệu (EDA & Feature Engineering)

### Tiền xử lý & Xử lý giá trị khuyết
* Loại bỏ hoàn toàn biến `date_unregistration` do chứa rủi ro rò rỉ dữ liệu nghiêm trọng (99.1% sinh viên Withdrawn có ghi nhận ngày này).
* Điền giá trị khuyết (Missing Values): `date_registration` điền bằng trung vị theo đợt học; `imd_band` và `age_band` được gán nhãn `"Unknown"`; `score` khuyết được điền bằng `0` (phản ánh bài tập không nộp).
* Loại bỏ toàn bộ cấu trúc bài thi cuối kỳ (`Exam`) và các cột `week_from`, `week_to`, `date_unregistration`.

### Mốc cắt dữ liệu (Cut-off Selection)
* Hệ thống thiết lập mốc cắt dữ liệu tại **Ngày 170** (tương ứng tối đa 25 tuần học, $T_{MAX} = 25$) dựa trên phân tích phân phối ngày rút môn (giữ lại 89.3% số lượng học viên Withdrawn và bảo toàn 35.1% thời gian học còn lại để thực hiện can thiệp học thuật).

### Phân chia dữ liệu kiểm thử (Out-of-Time Split)
Để đánh giá mô hình một cách khách quan và chống Overfitting theo thời gian, dữ liệu được phân tách theo kỳ học thực tế (OOT Split):
* **Tập Huấn luyện (Train):** 21.333 lượt học thuộc các kỳ cũ (`2013B`, `2013J`, `2014B`).
* **Tập Kiểm thử (Test):** 11.260 lượt học thuộc kỳ cuối cùng (`2014J`).

### Lọc chọn đặc trưng (Feature Selection)
* **Fairness Filtering:** Loại bỏ hoàn toàn biến `gender` và `disability` để đảm bảo mô hình không phân biệt đối xử đạo đức.
* **Statistical Filtering:** Kết hợp kiểm định tương quan Pearson ($|corr| \ge 0.05$ tại Tuần 24) và kiểm định Chi-square ($p \le 0.05$), tinh lọc hệ thống còn lại **20 đặc trưng động** và **20 đặc trưng tĩnh**.

---

## 3. Cấu trúc Thư mục Ứng dụng

```text
webapp/
├── api/                  # FastAPI Backend Engine
│   ├── routes/           # Các endpoints: predict, explain, timeline, metrics
│   ├── data_manager.py   # Quản lý nạp và truy xuất dữ liệu học viên
│   ├── model_loader.py   # Tải trước các mô hình ML/DL vào bộ nhớ (Lifespan)
│   └── main.py           # Điểm khởi chạy API và cấu hình Middleware CORS
├── ui/                   # Streamlit Frontend Dashboard
│   ├── views/            # Giao diện chức năng: Risk Timeline, Week Detail, Shap
│   ├── utils/            # Client HTTP giao tiếp Backend
│   └── app.py            # Entry point của giao diện người dùng
├── config.json           # Cấu hình đường dẫn mô hình và tham số hệ thống
└── requirements.txt      # Định nghĩa các thư viện phụ thuộc của dự án
```
## 4. Hướng dẫn Triển khai & Khởi chạy
Cài đặt môi trường
Đảm bảo hệ thống đã cài đặt Python (khuyến nghị phiên bản 3.10 hoặc 3.11). Khởi tạo môi trường ảo và cài đặt các thư viện dependencies:

```Bash
pip install -r webapp/requirements.txt
```

Khởi chạy tự động (Windows)
Sử dụng file script tích hợp để khởi chạy đồng thời cả Backend và Giao diện:

```Bash
run.bat
```
Khởi chạy thủ công (Debug)
Nếu cần chạy hoặc kiểm thử độc lập từng phân hệ, mở hai terminal song song:

Khởi chạy FastAPI Backend (Terminal 1):

```Bash
cd webapp
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Tài liệu Swagger UI sẽ khả dụng tại: http://127.0.0.1:8000/docs

Khởi chạy Streamlit Dashboard (Terminal 2):

```Bash
cd webapp
streamlit run ui/app.py
```
