import json, sys
nb = json.load(open(r"d:/Projects/Personal/UIT/CS-313-Data-Mining/Notebook/OULAD_EDA_Consolidated_thucln.ipynb", encoding="utf-8"))
idx = int(sys.argv[1])
cell = nb["cells"][idx]
print(f"--- CELL {idx} [{cell['cell_type']}] ---")
print("".join(cell.get("source", [])))
print("\n=== OUTPUTS ===")
for out in cell.get("outputs", []):
    if "text" in out:
        print("".join(out["text"]))
    elif "data" in out:
        for mime, content in out["data"].items():
            if mime == "text/plain":
                print("".join(content) if isinstance(content, list) else content)
