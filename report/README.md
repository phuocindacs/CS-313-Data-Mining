# Báo cáo LaTeX — CS313 OULAD

## Cấu trúc

```
report/
├── main.tex          # File chính (trang bìa + \input section_*)
├── section_I.tex     # Giới thiệu đề tài
├── section_II.tex    # EDA — Khám phá dữ liệu
├── .latexmkrc        # Cấu hình build (pdflatex + latexmk)
├── figures/          # Hình ảnh (đặt UIT_Logo.jpg ở đây)
└── README.md
```

## Yêu cầu

1. **TeX distribution**: MiKTeX (Windows) hoặc TeX Live.
   - MiKTeX sẽ tự cài thêm gói khi build lần đầu (`vntex`, `tikz`, `fancyhdr`, `tocloft`, `hyperref`, `scrextend`, …).
2. **VS Code extension**: `James-Yu.latex-workshop` (đã cài).
3. **Logo UIT**: bỏ file `UIT_Logo.jpg` vào `report/figures/`, sau đó bỏ comment dòng `\includegraphics` trong `main.tex`.

## Preview như Overleaf trong VS Code

1. Mở thư mục dự án trong VS Code.
2. Mở `report/main.tex`.
3. Build:
   - Phím tắt: `Ctrl + Alt + B` → LaTeX Workshop sẽ chạy `latexmk` theo `.latexmkrc`.
   - Hoặc: TeX sidebar (biểu tượng TeX bên trái) → **Build LaTeX project**.
4. Preview PDF:
   - `Ctrl + Alt + V` → mở tab PDF cạnh bên (live reload).
   - SyncTeX hai chiều: Ctrl+Click vào PDF nhảy về source, ngược lại từ source ra PDF.

## Build bằng command line

```powershell
cd report
latexmk -pdf main.tex     # build
latexmk -c                # clean file phụ (.aux, .log, .toc, …)
latexmk -C                # clean toàn bộ kể cả PDF
```

## Ghi chú

- File dùng `\usepackage[utf8]{vietnam}` → **phải build bằng `pdflatex`**, không phải `xelatex`/`lualatex`. `.latexmkrc` đã set sẵn.
- Nếu gặp lỗi font hoặc thiếu gói: chạy `mpm --admin` (MiKTeX Console) → **Update / Install missing packages on the fly: Yes**.
- Encoding source: **UTF-8 (no BOM)** để gói `vietnam` đọc đúng tiếng Việt.
