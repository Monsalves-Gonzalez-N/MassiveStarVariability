"""
Build unified LS+ACF catalog (log normalization).

Step 1 — Per TIC × sector_list (combining all peaks from LS and ACF):
  - If ALL peaks are Rndm/No_periodo → "Non-periodic"
  - If, after removing Rndm/No_periodo, ALL remaining are LPV → "Irregular"
  - Discard Rndm/No_periodo and LPV peaks; from the rest, keep the peak
    with highest probability
  - Tiebreaker: LS vs LS → lowest power; ACF vs ACF → highest power;
    LS vs ACF → keep BOTH in ties subset

Step 2 — Probability threshold:
  - prob < 0.8 → "Unconstrained"

Step 3 — Group by TIC across sectors:
  - Same class in all sectors → unified entry (one TIC, all periods)
  - Different classes across sectors → separate CSV for review
"""

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

# ── 1. Load & tag source ─────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ls = pd.read_csv(os.path.join(REPO_ROOT, "data/catalogos/CNN_RF_prediction_LS_log.csv"))
acf = pd.read_csv(os.path.join(REPO_ROOT, "data/catalogos/CNN_RF_prediction_ACF_log.csv"))

ls["source"] = "LS"
ls["cube_idx"] = ls.index

acf["source"] = "ACF"
acf["cube_idx"] = acf.index

combined = pd.concat([ls, acf], ignore_index=True)

# ── 2. Best peak per TIC × sector_list ───────────────────────────────
groups = combined.groupby(["TIC", "sector_list"])

non_periodic_rows = []
irregular_rows = []
best_rows = []
tie_rows = []

for (tic, sector), grp in groups:
    # 1) All peaks are Rndm/No_periodo → Non-periodic
    no_rndm = grp[~grp["final_class"].isin(["Rndm", "No_periodo"])]
    if no_rndm.empty:
        row = grp.iloc[0].copy()
        row["final_class"] = "Non-periodic"
        row["CNN+RF_max_prob"] = 0.0
        row["power"] = 0.0
        row["period"] = 0.0
        non_periodic_rows.append(row)
        continue

    # 2) Without Rndm/No_periodo, if all remaining are LPV → Irregular
    no_lpv = no_rndm[no_rndm["final_class"] != "LPV"]
    if no_lpv.empty:
        row = no_rndm.iloc[0].copy()
        row["final_class"] = "Irregular"
        irregular_rows.append(row)
        continue

    # 3) Discard Rndm/No_periodo and LPV; select best from the rest
    valid_sorted = no_lpv.sort_values("CNN+RF_max_prob", ascending=False)
    max_prob = valid_sorted["CNN+RF_max_prob"].iloc[0]
    top = valid_sorted[valid_sorted["CNN+RF_max_prob"] == max_prob]

    if len(top) == 1:
        best_rows.append(top.iloc[0])
    else:
        top_ls = top[top["source"] == "LS"]
        top_acf = top[top["source"] == "ACF"]
        resolved = []
        if len(top_ls) > 0:
            resolved.append(top_ls.sort_values("power", ascending=True).iloc[0])
        if len(top_acf) > 0:
            resolved.append(top_acf.sort_values("power", ascending=False).iloc[0])
        if len(resolved) == 1:
            best_rows.append(resolved[0])
        else:
            # LS vs ACF tie: same class & periods within 10% → random pick
            r0, r1 = resolved[0], resolved[1]
            same_class = r0["final_class"] == r1["final_class"]
            p0, p1 = r0["period"], r1["period"]
            close_period = abs(p0 - p1) / max(p0, p1) < 0.10 if max(p0, p1) > 0 else True
            if same_class and close_period:
                best_rows.append(resolved[rng.integers(2)])
            else:
                for r in resolved:
                    tie_rows.append(r)

# ── 3. Build per-sector catalog & apply threshold ────────────────────
catalog_sector = pd.DataFrame(best_rows + non_periodic_rows + irregular_rows)
catalog_sector = catalog_sector.sort_values(["TIC", "sector_list"]).reset_index(drop=True)

catalog_ties = pd.DataFrame(tie_rows)
if not catalog_ties.empty:
    catalog_ties = catalog_ties.sort_values(
        ["TIC", "sector_list", "CNN+RF_max_prob"], ascending=[True, True, False]
    ).reset_index(drop=True)

# Probability threshold: < 0.8 → Unconstrained
mask_low = (
    (catalog_sector["CNN+RF_max_prob"] < 0.8) &
    (~catalog_sector["final_class"].isin(["Non-periodic", "Irregular"]))
)
catalog_sector.loc[mask_low, "final_class"] = "Unconstrained"

# ── 4. Group by TIC across sectors ───────────────────────────────────
consistent_rows = []
mixed_rows = []

for tic, grp in catalog_sector.groupby("TIC"):
    classes = grp["final_class"].unique()
    # Filter out Unconstrained and Non-periodic for consistency check
    real_classes = [c for c in classes if c not in ("Unconstrained", "Non-periodic", "Irregular")]

    if len(real_classes) <= 1:
        # Consistent: all sectors agree (or all are Unconstrained/Non-periodic)
        for _, row in grp.iterrows():
            consistent_rows.append(row)
    else:
        # Mixed: different classes across sectors → review
        for _, row in grp.iterrows():
            mixed_rows.append(row)

catalog_main = pd.DataFrame(consistent_rows).reset_index(drop=True)
catalog_mixed = pd.DataFrame(mixed_rows).reset_index(drop=True)

# ── 5. Save ──────────────────────────────────────────────────────────
catalog_main.to_csv(os.path.join(REPO_ROOT, "data/catalogos/catalog_unified_LS_ACF_log.csv"), index=False)

if not catalog_ties.empty:
    catalog_ties.to_csv(os.path.join(REPO_ROOT, "data/catalogos/catalog_unified_ties.csv"), index=False)

if not catalog_mixed.empty:
    catalog_mixed.to_csv(os.path.join(REPO_ROOT, "data/catalogos/catalog_mixed_classes.csv"), index=False)

# Save one CSV per class
for cls, grp in catalog_main.groupby("final_class"):
    fname = os.path.join(REPO_ROOT, f"data/catalogos/catalog_{cls}.csv")
    grp.to_csv(fname, index=False)
    print(f"  → {fname} ({len(grp)} filas, {grp.TIC.nunique()} TICs)")

# ── 6. Summary ───────────────────────────────────────────────────────
print(f"\n{'═'*50}")
print(f"Catálogo principal: {len(catalog_main)} filas ({catalog_main.TIC.nunique()} TICs)")
print(f"\nClases:")
print(catalog_main["final_class"].value_counts().to_string())
print(f"\nEmpates LS vs ACF: {len(catalog_ties)} filas "
      f"({catalog_ties[['TIC','sector_list']].drop_duplicates().shape[0] if not catalog_ties.empty else 0} "
      f"pares TIC-sector)")
print(f"\nClases mixtas (para revisar): {len(catalog_mixed)} filas "
      f"({catalog_mixed.TIC.nunique() if not catalog_mixed.empty else 0} TICs)")
