"""
Build unified LS+ACF catalog — path 2 (hierarchical cascade).

Step 1 — Per TIC: classify non-periodic cases (No_periodo, Rndm, Irregular).
Step 2 — Per (TIC, sector): vote by sector for remaining TICs.
Step 3 — Per TIC: final class by mode of sector votes.
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

THRESHOLD = 0.8
NON_PERIODIC = {"No_periodo", "Rndm", "LPV"}
rng = np.random.default_rng(seed=42)

# ── Load & tag ────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ls = pd.read_csv(os.path.join(REPO_ROOT, "data/catalogos/CNN_RF_prediction_LS_log.csv"))
acf = pd.read_csv(os.path.join(REPO_ROOT, "data/catalogos/CNN_RF_prediction_ACF_log.csv"))

ls["source"] = "LS"
ls["cube_idx"] = ls.index
acf["source"] = "ACF"
acf["cube_idx"] = acf.index

combined = pd.concat([ls, acf], ignore_index=True)

# ── Step 1: Hierarchical non-periodic TIC classification ──────────────
# Rules applied in order; first match wins (TIC exits pipeline).

step1_results = {}

for tic, grp in combined.groupby("TIC"):
    classes = set(grp["final_class"].unique())
    has_above = (grp["CNN+RF_max_prob"] >= THRESHOLD).any()

    # 1.1 — No_periodo: ALL peaks are No_periodo
    if classes <= {"No_periodo"}:
        step1_results[tic] = "No_periodo"

    # 1.2 — Rndm: ALL peaks ∈ {No_periodo, Rndm} — no periodic class at all
    elif classes <= {"No_periodo", "Rndm"}:
        step1_results[tic] = "Rndm"

    # 1.3 — Irregular: ALL peaks ∈ {Rndm, LPV, No_periodo}, at least one LPV, at least one >= threshold
    elif classes <= {"No_periodo", "Rndm", "LPV"} and "LPV" in classes and has_above:
        step1_results[tic] = "Irregular"

    # 1.4 — Fall-through: TIC passes to Step 2

step1_tic_set = set(step1_results)
step2_combined = combined[~combined["TIC"].isin(step1_tic_set)]

# Step 1 output: all original rows tagged with final_class_tic
step1_output = combined[combined["TIC"].isin(step1_tic_set)].copy()
step1_output["final_class_tic"] = step1_output["TIC"].map(step1_results)
step1_output["unconstrained_sector"] = False

# ── Step 2: Sector votes for remaining TICs ───────────────────────────
# Pre-filter: only periodic peaks participate in voting.
step2_periodic = step2_combined[~step2_combined["final_class"].isin(NON_PERIODIC)].copy()
step2_tic_set = set(step2_combined["TIC"].unique())

tic_votes = {tic: [] for tic in step2_tic_set}
sector_output_rows = []


def _best_peak(subgrp, tiebreak_ascending):
    """Peak with highest prob; tiebreak by power (asc or desc)."""
    if subgrp.empty:
        return None
    max_prob = subgrp["CNN+RF_max_prob"].max()
    candidates = subgrp[subgrp["CNN+RF_max_prob"] == max_prob]
    return candidates.sort_values("power", ascending=tiebreak_ascending).iloc[0].copy()


for (tic, sector), grp in step2_periodic.groupby(["TIC", "sector_list"]):
    acf_peaks = grp[grp["source"] == "ACF"]
    ls_peaks = grp[grp["source"] == "LS"]

    if (grp["CNN+RF_max_prob"] < THRESHOLD).all():
        # Case A: Unconstrained sector — 0 votes, keep top peaks for output.
        # ACF tiebreak: highest power; LS tiebreak: lowest power.
        top_acf = _best_peak(acf_peaks, tiebreak_ascending=False)
        top_ls = _best_peak(ls_peaks, tiebreak_ascending=True)

        if top_acf is not None and top_ls is not None:
            if top_acf["CNN+RF_max_prob"] == top_ls["CNN+RF_max_prob"]:
                row = [top_acf, top_ls][rng.integers(2)]
                row["unconstrained_sector"] = True
                sector_output_rows.append(row)
            else:
                for row in [top_acf, top_ls]:
                    row["unconstrained_sector"] = True
                sector_output_rows.extend([top_acf, top_ls])
        elif top_acf is not None:
            top_acf["unconstrained_sector"] = True
            sector_output_rows.append(top_acf)
        elif top_ls is not None:
            top_ls["unconstrained_sector"] = True
            sector_output_rows.append(top_ls)
        # No vote appended: Unconstrained contributes 0 to mode.

    else:
        # Case B: at least one periodic peak above threshold.
        # Winning ACF: highest prob, tiebreak highest power.
        # Winning LS:  highest prob, tiebreak lowest power.
        winning_acf = _best_peak(acf_peaks, tiebreak_ascending=False)
        winning_ls = _best_peak(ls_peaks, tiebreak_ascending=True)

        if winning_acf is not None and winning_ls is not None:
            p_acf = winning_acf["CNN+RF_max_prob"]
            p_ls = winning_ls["CNN+RF_max_prob"]
            if p_acf > p_ls:
                winning_acf["unconstrained_sector"] = False
                tic_votes[tic].append(winning_acf["final_class"])
                sector_output_rows.append(winning_acf)
            elif p_ls > p_acf:
                winning_ls["unconstrained_sector"] = False
                tic_votes[tic].append(winning_ls["final_class"])
                sector_output_rows.append(winning_ls)
            else:
                # Equal prob: 2 votes (one per source)
                winning_acf["unconstrained_sector"] = False
                winning_ls["unconstrained_sector"] = False
                tic_votes[tic].extend([winning_acf["final_class"], winning_ls["final_class"]])
                sector_output_rows.extend([winning_acf, winning_ls])
        elif winning_acf is not None:
            winning_acf["unconstrained_sector"] = False
            tic_votes[tic].append(winning_acf["final_class"])
            sector_output_rows.append(winning_acf)
        elif winning_ls is not None:
            winning_ls["unconstrained_sector"] = False
            tic_votes[tic].append(winning_ls["final_class"])
            sector_output_rows.append(winning_ls)

# ── Step 3: Mode per TIC ──────────────────────────────────────────────
tic_final_class = {}

for tic in step2_tic_set:
    votes = tic_votes[tic]
    if not votes:
        # No contributing sectors → Unconstrained
        tic_final_class[tic] = "Unconstrained"
    else:
        counts = Counter(votes)
        max_n = max(counts.values())
        mode_cls = [c for c, n in counts.items() if n == max_n]
        tic_final_class[tic] = mode_cls[0] if len(mode_cls) == 1 else "Mixed"

if sector_output_rows:
    step2_output = pd.DataFrame(sector_output_rows).reset_index(drop=True)
    step2_output["final_class_tic"] = step2_output["TIC"].map(tic_final_class)
else:
    step2_output = pd.DataFrame()

# ── Step 4: Build main catalog and save ──────────────────────────────
catalog_main = pd.concat(
    [df for df in [step1_output, step2_output] if not df.empty],
    ignore_index=True,
)

catalog_main.to_csv(os.path.join(REPO_ROOT, "data/catalogos/catalog_path2_main.csv"), index=False)

catalog_mixed = catalog_main[catalog_main["final_class_tic"] == "Mixed"]
if not catalog_mixed.empty:
    catalog_mixed.to_csv(os.path.join(REPO_ROOT, "data/catalogos/catalog_path2_mixed.csv"), index=False)

for cls, grp in catalog_main[catalog_main["final_class_tic"] != "Mixed"].groupby("final_class_tic"):
    fname = os.path.join(REPO_ROOT, f"data/catalogos/catalog_path2_{cls}.csv")
    grp.to_csv(fname, index=False)

# ── Summary ───────────────────────────────────────────────────────────
tic_cls = catalog_main.drop_duplicates("TIC").set_index("TIC")["final_class_tic"]
print(f"\n{'═'*50}")
print(f"catalog_path2_main: {len(catalog_main)} filas, {catalog_main['TIC'].nunique()} TICs")
print(f"\nfinal_class_tic (por TIC):")
print(tic_cls.value_counts().to_string())
print(f"\nTICs en Mixed:         {(tic_cls == 'Mixed').sum()}")
print(f"TICs en Unconstrained: {(tic_cls == 'Unconstrained').sum()}")
