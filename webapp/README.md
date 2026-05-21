# OULAD Early Warning System — Web Demo

Demo web cho CS-313 Data Mining. Gồm 2 service:
- **FastAPI** (port 8000) — load model, tính prediction + SHAP
- **Streamlit** (port 8501) — giao diện demo

---

## Yêu cầu

- Python **3.10 – 3.13** (Windows)
- Tất cả model đã train xong (xem mục [Cấu trúc](#cấu-trúc-thư-mục))

---

## Cài đặt (1 lần duy nhất)

```powershell
cd webapp
pip install -r requirements.txt
```

---

## Khởi động

### Cách 1 — Tự động (khuyên dùng)

Mở **2 terminal riêng**:

**Terminal 1 — API:**
```powershell
cd D:\...\CS-313-Data-Mining\webapp
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Chờ đến khi terminal in ra:
```
✓ Ready — 12 ML models, 2 DL models, 11260 students
```
(Mất ~30–40 giây do load model + dữ liệu)

**Terminal 2 — UI:**
```powershell
cd D:\...\CS-313-Data-Mining\webapp
python run_ui.py
```

Mở trình duyệt: **http://127.0.0.1:8501**

> ⚠️ Dùng **127.0.0.1**, không dùng `localhost` (tránh lỗi WebSocket trên Windows)

---

### Cách 2 — Thủ công (nếu cách 1 lỗi)

```powershell
# Terminal 1
cd webapp
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 (mở sau khi API ready)
cd webapp
python -m streamlit run ui/app.py --server.port 8501 --server.address 127.0.0.1
```

---

## Các trang trong demo

| Trang | Mô tả |
|-------|-------|
| 📈 **Risk Timeline** | Biểu đồ xác suất At-Risk theo tuần (4→24) cho cả 4 model |
| 🔍 **Week Detail** | Prediction của từng model tại 1 tuần cụ thể |
| 📊 **SHAP Analysis** | Giải thích feature importance cho XGBoost/LightGBM |
| 📋 **Model Metrics** | Thông tin model đã load, danh sách features |

---

## Cấu trúc thư mục

```
webapp/
├── config.json                  # Cấu hình đường dẫn model + data
├── run_ui.py                    # Launcher UI (fix asyncio Windows)
├── run.bat                      # Khởi động cả 2 service (batch)
├── requirements.txt
│
├── data/                        # Dữ liệu test (sinh ra từ Sequence FE notebook)
│   ├── df_test_long.csv         # Long-format, 281500 rows × 63 cols
│   ├── df_test.csv              # Unscaled, dùng để hiển thị
│   ├── X_seq_test.npy           # (11260, 25, 33) — input DL
│   ├── X_static_test.npy        # (11260, 25) — static features
│   ├── mask_test.npy            # (11260, 25) — padding mask
│   ├── y_test.npy               # (11260,) — ground truth
│   └── feature_cols_detail.json # Tên 58 features (dynamic + static)
│
├── api/
│   ├── main.py                  # FastAPI app
│   ├── model_loader.py          # Load XGBoost/LightGBM/LSTM/Transformer
│   ├── data_manager.py          # Load + cache dữ liệu test
│   ├── shap_explainer.py        # SHAP TreeExplainer
│   └── routes/
│       ├── predict.py           # POST /predict, POST /shap
│       ├── students.py          # GET /students, GET /students/{idx}
│       ├── metadata.py          # GET /metadata
│       └── timeline.py          # GET /timeline/{student_idx}
│
├── ui/
│   ├── app.py                   # Streamlit main (sidebar + routing)
│   ├── utils/
│   │   └── api_client.py        # HTTP wrapper gọi FastAPI
│   └── views/                   # Các trang
│       ├── risk_timeline.py
│       ├── week_detail.py
│       ├── shap_analysis.py
│       └── model_metrics.py
│
└── .streamlit/
    └── config.toml              # Streamlit config (port, CORS, toolbar)
```

---

## Model cần có

Các file model được đọc từ `config.json`. Đường dẫn mặc định:

| Model | Đường dẫn |
|-------|-----------|
| XGBoost week W | `../ketqua/OULAD ML/models/xgb_model_week_{W}.pkl` |
| LightGBM week W | `../ketqua/OULAD ML/models/lgbm_model_week_{W}.pkl` |
| LSTM | `../ketqua/OULAD sequence/models/best_lstm_model.pth` |
| Transformer | `../ketqua/OULAD sequence/models/best_transformer_model.pth` |

Các tuần W = 4, 8, 12, 16, 20, 24 → tổng **12 ML + 2 DL = 14 file model**.

> Để thay đường dẫn: chỉnh `config.json`, **không cần sửa code**.

---

## Sinh dữ liệu test (nếu chưa có)

Dữ liệu trong `data/` được sinh từ notebook Sequence FE:
```
ketqua/OULAD sequence/OULAD-01-FE/
```

Chạy notebook đó với output path trỏ vào `webapp/data/`. Chi tiết hỏi người làm FE (hướng sequence).

---

## API docs

Khi FastAPI đang chạy, truy cập:
```
http://127.0.0.1:8000/docs
```

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Fix |
|-----|-------------|-----|
| `WebSocket failed` trên Chrome | Extension chặn WS | Dùng Incognito hoặc `127.0.0.1` thay `localhost` |
| `503 Data not loaded` | API chưa ready | Chờ thêm, API cần ~30–40s để load |
| `Model X not loaded` | File pkl/pth thiếu | Kiểm tra đường dẫn trong `config.json` |
| `ModuleNotFoundError` | Chưa install deps | Chạy `pip install -r requirements.txt` |
| Sidebar text vô hình | Dark mode conflict | Đã fix, restart Streamlit |
