"""
Verify demographic signal strength for the EDA report.
- Compute Chi-square + Cramer's V + Mutual Information for all 6 demographic
  features the report mentions (3 kept + 3 dropped).
- Output a comparable table so we can say "mạnh = Cramer's V > X" with real numbers.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

DATA = Path(r"d:/Projects/Personal/UIT/CS-313-Data-Mining/asset")
df = pd.read_csv(DATA / 'studentInfo.csv')

features = ['num_of_prev_attempts', 'highest_education', 'imd_band',
            'gender', 'disability', 'studied_credits',
            'age_band', 'region', 'code_module']

y = df['final_result']

print(f"{'Feature':22s}  {'Chi2':>10s}  {'p-value':>10s}  {'CramersV':>9s}  {'MI':>7s}")
print("-" * 70)
rows = []
for col in features:
    sub = df.dropna(subset=[col])
    ct = pd.crosstab(sub[col], sub['final_result'])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    # Cramer's V (bias-corrected version of Bergsma 2013)
    cv = np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0
    # Mutual information (numeric: label-encode categorical)
    x_enc = LabelEncoder().fit_transform(sub[col].astype(str))
    mi = mutual_info_classif(x_enc.reshape(-1, 1),
                              LabelEncoder().fit_transform(sub['final_result']),
                              discrete_features=True, random_state=42)[0]
    print(f"{col:22s}  {chi2:10.1f}  {p:10.2e}  {cv:9.4f}  {mi:7.4f}")
    rows.append((col, chi2, p, cv, mi))

print("\n=== Reference thresholds (Cohen, Cramer's V for df=3) ===")
print("  V < 0.10  : negligible")
print("  V 0.10-0.30: weak")
print("  V 0.30-0.50: medium")
print("  V > 0.50  : strong")

print("\n=== Sorted by Cramer's V (high to low) ===")
for r in sorted(rows, key=lambda x: -x[3]):
    print(f"  {r[0]:22s}  V={r[3]:.4f}  MI={r[4]:.4f}")
