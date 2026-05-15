# latexmk config for LaTeX Workshop / Overleaf-style local build.
# Vietnamese support qua \usepackage[utf8]{vietnam} -> dùng pdflatex.

$pdf_mode      = 1;            # build PDF
$pdflatex      = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use    = 2;            # chạy bibtex khi cần
$max_repeat    = 4;            # đủ vòng cho TOC + hyperref
@default_files = ('main.tex');
