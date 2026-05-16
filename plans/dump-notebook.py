import json, sys
nb_path = r"d:/Projects/Personal/UIT/CS-313-Data-Mining/Notebook/OULAD_EDA_Consolidated_thucln.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", [])).strip()
    if not src:
        continue
    print(f"\n--- CELL {i} [{c['cell_type']}] ---")
    print(src[:600])
