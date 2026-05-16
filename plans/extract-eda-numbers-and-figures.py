"""
Extract exact EDA numbers and export 2 figures for the report.
Reads OULAD CSVs directly from asset/ (the same files the notebook uses).
No fabricated data — every number comes from real groupby/value_counts.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(r"d:/Projects/Personal/UIT/CS-313-Data-Mining/asset")
FIG_DIR  = Path(r"d:/Projects/Personal/UIT/CS-313-Data-Mining/report/figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)

CUTOFF_DAY = 135
RESULT_ORDER = ['Distinction', 'Pass', 'Fail', 'Withdrawn']
COLORS = {
    'Distinction': '#2980b9',
    'Pass':        '#27ae60',
    'Fail':        '#e67e22',
    'Withdrawn':   '#e74c3c',
}

plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
})

# ── Load core tables ─────────────────────────────────────────────────────────
df_info = pd.read_csv(DATA_DIR / 'studentInfo.csv')
df_sa   = pd.read_csv(DATA_DIR / 'studentAssessment.csv')
df_ass  = pd.read_csv(DATA_DIR / 'assessments.csv')
df_reg  = pd.read_csv(DATA_DIR / 'studentRegistration.csv')
df_crs  = pd.read_csv(DATA_DIR / 'courses.csv')

print("="*60)
print("1. TARGET DISTRIBUTION")
print("="*60)
counts = df_info['final_result'].value_counts().reindex(RESULT_ORDER)
total = counts.sum()
for label, n in counts.items():
    print(f"  {label:12s}: {n:6,} ({n/total*100:5.2f}%)")
print(f"  TOTAL       : {total:,}")

print("\n" + "="*60)
print("2. WITHDRAWAL PHASES (calculated from date_unregistration)")
print("="*60)
merged_reg = df_reg.merge(
    df_info[['code_module','code_presentation','id_student','final_result']],
    on=['code_module','code_presentation','id_student'], how='inner'
)
withdrawn = merged_reg[merged_reg['final_result'] == 'Withdrawn'].copy()
withdrawn = withdrawn.dropna(subset=['date_unregistration'])

# Phases: A=before start (<0), B=early (0-30d), C=mid (30-90d), D=late (>90d)
# Plus very-early (<0) which captures students who withdrew before module began
def phase(d):
    if d < 0:    return 'A: Trước khai giảng (<0)'
    if d < 30:   return 'B: Sớm (0–30 ngày)'
    if d < 90:   return 'C: Giữa (30–90 ngày)'
    return 'D: Muộn (>90 ngày)'

withdrawn['phase'] = withdrawn['date_unregistration'].apply(phase)
phase_order = ['A: Trước khai giảng (<0)', 'B: Sớm (0–30 ngày)',
               'C: Giữa (30–90 ngày)', 'D: Muộn (>90 ngày)']
phase_counts = withdrawn['phase'].value_counts().reindex(phase_order)
phase_total = phase_counts.sum()
for ph, n in phase_counts.items():
    print(f"  {ph:35s}: {n:5,} ({n/phase_total*100:5.2f}%)")
print(f"  Total Withdrawn (có ngày rút)     : {phase_total:,}")

print("\n" + "="*60)
print("3. WEIGHTED SCORE BY final_result (median per class)")
print("="*60)
# Build weighted_score per student at cutoff 135
merged = df_sa.merge(
    df_ass[['id_assessment','assessment_type','weight','date','code_module','code_presentation']],
    on='id_assessment', how='left'
).merge(
    df_info[['id_student','code_module','code_presentation','final_result']],
    on=['id_student','code_module','code_presentation'], how='inner'
)
merged = merged[merged['assessment_type'].isin(['TMA','CMA'])]
merged = merged[merged['date'] <= CUTOFF_DAY]
merged['score'] = pd.to_numeric(merged['score'], errors='coerce').fillna(0)
merged['weight'] = merged['weight'].fillna(1)

def weighted_mean(g):
    denom = g['weight'].sum()
    return (g['score'] * g['weight']).sum() / denom if denom > 0 else np.nan

ws = (merged.groupby(['id_student','code_module','code_presentation'])
            .apply(weighted_mean)
            .reset_index(name='weighted_score')
            .merge(df_info[['id_student','code_module','code_presentation','final_result']],
                   on=['id_student','code_module','code_presentation'], how='left')
            .dropna(subset=['weighted_score','final_result']))

for label in RESULT_ORDER:
    sub = ws[ws['final_result']==label]['weighted_score']
    print(f"  {label:12s}: median={sub.median():5.2f}, mean={sub.mean():5.2f}, n={len(sub):,}")

print("\n" + "="*60)
print("4. MODULE STRUCTURE (CMA count, Exam weight, TMA weight)")
print("="*60)
pivot_count  = df_ass.groupby(['code_module','assessment_type'])['id_assessment'].count().unstack(fill_value=0)
pivot_weight = df_ass.groupby(['code_module','assessment_type'])['weight'].sum().unstack(fill_value=0)
mod_table = pd.DataFrame({
    'CMA count':   pivot_count.get('CMA', 0).astype(int),
    'Exam weight': pivot_weight.get('Exam', 0).astype(int),
    'TMA weight':  pivot_weight.get('TMA', 0).astype(int),
})
print(mod_table)

print("\n" + "="*60)
print("5. PRESENTATION (B vs J) PASS RATE")
print("="*60)
df_info['semester'] = df_info['code_presentation'].str[-1]
df_info['is_pass']  = df_info['final_result'].isin(['Pass','Distinction']).astype(int)
sem = df_info.groupby('semester')['is_pass'].agg(['mean','count'])
sem['pass_rate_pct'] = sem['mean']*100
print(sem[['pass_rate_pct','count']].round(2))

print("\n" + "="*60)
print("6. VLE & DATASET TOTALS")
print("="*60)
print(f"  studentInfo rows: {len(df_info):,}")
print(f"  studentAssessment rows: {len(df_sa):,}")
print(f"  assessments rows: {len(df_ass):,}")
print("  (studentVle rows: too large to load here — see notebook)")

# ============================================================================
# FIGURES
# ============================================================================
print("\n" + "="*60)
print("EXPORTING FIGURES")
print("="*60)

# --- Figure 1: Boxplot weighted_score by final_result ---
fig, ax = plt.subplots(figsize=(7, 4))
order = RESULT_ORDER
box_data = [ws[ws['final_result']==r]['weighted_score'].values for r in order]
bp = ax.boxplot(box_data, tick_labels=order, patch_artist=True, showfliers=False,
                widths=0.55, medianprops={'color':'black','linewidth':1.5})
for patch, r in zip(bp['boxes'], order):
    patch.set_facecolor(COLORS[r])
    patch.set_alpha(0.85)
    patch.set_edgecolor('black')
ax.set_xlabel('Kết quả cuối kỳ (final_result)', fontsize=11)
ax.set_ylabel('Điểm tổng có trọng số (cut-off 135 ngày)', fontsize=11)
ax.set_title('Phân phối weighted_score theo final_result', fontsize=12, fontweight='bold')
ax.set_ylim(-5, 105)
plt.tight_layout()
out1 = FIG_DIR / 'boxplot-weighted-score-by-result.png'
plt.savefig(out1, dpi=160, bbox_inches='tight')
plt.close()
print(f"  Saved: {out1}")

# --- Figure 2: Withdrawal phases bar chart ---
fig, ax = plt.subplots(figsize=(7, 4))
labels = [p.split(':')[1].strip() for p in phase_order]
vals = (phase_counts / phase_total * 100).values
bars = ax.bar(['A','B','C','D'], vals,
              color=['#c0392b','#e67e22','#f39c12','#95a5a6'],
              edgecolor='black', linewidth=0.5)
for bar, v, lab in zip(bars, vals, labels):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.8, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar.get_x()+bar.get_width()/2, -3, lab,
            ha='center', va='top', fontsize=8, color='#444')
ax.set_ylabel('Tỉ lệ trong nhóm Withdrawn (%)', fontsize=11)
ax.set_xlabel('Giai đoạn rút môn (Phase)', fontsize=11)
ax.set_title('Phân bố thời điểm rút môn của sinh viên Withdrawn',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, max(vals)*1.2)
plt.tight_layout()
out2 = FIG_DIR / 'withdrawal-phases.png'
plt.savefig(out2, dpi=160, bbox_inches='tight')
plt.close()
print(f"  Saved: {out2}")

print("\nDONE.")
