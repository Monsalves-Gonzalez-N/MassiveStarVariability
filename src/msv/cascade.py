"""Path 2 — catálogo por cascada jerárquica (selección por peaks).

Port de las celdas Path-2 del notebook 2_Github_TESS_variability_OGLE:
  Paso 1: candidatos NO periódicos a granularidad TIC (Rndm / Irregular).
  Paso 2: candidatos periódicos por voto por sector (ganador ACF vs LS).
  Paso 3: ensamble del catálogo (+ No_periodo para TICs sin peaks).

Entrada: DataFrame de predicciones por peak (schema de classify_peaks /
run_brf_snr.py: TIC, sector, source, per, power, brf_class, brf_prob, ...).
No requiere TensorFlow.
"""
import numpy as np
import pandas as pd

CLASS_COL, PROB_COL = "brf_class", "brf_prob"
PER_COL, POWER_COL, SECTOR_COL = "per", "power", "sector"
NONPER = {"Rndm", "LPV"}
PERIODIC = {"ELL", "M", "CEP", "DST", "E", "RR"}


def _acf_winner(sub):   # desempate: mayor power
    return sub.sort_values([PROB_COL, POWER_COL], ascending=[False, False]).iloc[0]


def _ls_winner(sub):    # desempate: menor power
    return sub.sort_values([PROB_COL, POWER_COL], ascending=[False, True]).iloc[0]


def _classify_tic_step1(g, threshold):
    """Reglas jerárquicas no periódicas: la primera que matchea gana."""
    classes = set(g[CLASS_COL])
    has_over = bool((g[PROB_COL] >= threshold).any())
    if classes == {"Rndm"} and has_over:                              # 1.2 Rndm
        return "Rndm"
    if classes <= {"Rndm", "LPV"} and "LPV" in classes and has_over:  # 1.3 Irregular
        return "Irregular"
    return None                                                       # 1.4 -> Paso 2


def path2_cascade(df_pred, cat_tics=None, threshold=0.8, seed=42):
    """Colapsa predicciones por peak a una clase por TIC.

    Args:
        df_pred: predicciones por peak (LS+ACF combinados vía `source`).
        cat_tics: TICs del catálogo completo; los que no tengan ningún peak
            salen como `No_periodo`. None -> solo TICs presentes en df_pred.
        threshold: umbral sobre brf_prob para votar.
        seed: solo desempates aleatorios ACF/LS en sectores Unconstrained.

    Returns:
        (main, tic_class): DataFrame por peak con `sector_class`,
        `final_class_tic` y `decision_step`; y Series clase final por TIC.
    """
    rng = np.random.default_rng(seed=seed)

    df_p2 = df_pred.copy()
    if "cube_idx" not in df_p2.columns:
        df_p2["cube_idx"] = np.arange(len(df_p2))
    df_p2 = df_p2[df_p2[CLASS_COL].notna()].reset_index(drop=True)

    peak_tics = set(df_p2["TIC"].astype(int).unique())
    cat_tics = peak_tics if cat_tics is None else set(int(t) for t in cat_tics)
    no_periodo_tics = sorted(cat_tics - peak_tics)

    # --- Paso 1: no periódicos a nivel TIC --------------------------------
    step1, fallthrough = {}, []
    for tic, g in df_p2.groupby("TIC"):
        lab = _classify_tic_step1(g, threshold)
        if lab is None:
            fallthrough.append(tic)
        else:
            step1[tic] = lab

    # --- Paso 2: periódicos, voto por sector ------------------------------
    out_rows, step2 = [], {}
    for tic in fallthrough:
        g = df_p2[(df_p2["TIC"] == tic) & (df_p2[CLASS_COL].isin(PERIODIC))]
        votes, tic_rows = [], []
        for sector, sg in g.groupby(SECTOR_COL):
            acf = sg[sg["source"] == "ACF"]
            ls = sg[sg["source"] == "LS"]
            a = _acf_winner(acf) if len(acf) else None
            l = _ls_winner(ls) if len(ls) else None
            if not bool((sg[PROB_COL] >= threshold).any()):   # Caso A: 0 votos
                if a is not None and l is not None and a[PROB_COL] == l[PROB_COL]:
                    keep = [a] if rng.choice(["ACF", "LS"]) == "ACF" else [l]
                else:
                    keep = [r for r in (a, l) if r is not None]
                for r in keep:
                    r = r.copy()
                    r["sector_class"] = "Unconstrained"
                    tic_rows.append(r)
            else:                                             # Caso B: vota
                if a is not None and l is not None:
                    if a[PROB_COL] > l[PROB_COL]:
                        win = [a]
                    elif l[PROB_COL] > a[PROB_COL]:
                        win = [l]
                    else:
                        win = [a, l]                          # empate -> 2 votos
                else:
                    win = [r for r in (a, l) if r is not None]
                for r in win:
                    votes.append(r[CLASS_COL])
                    r = r.copy()
                    r["sector_class"] = r[CLASS_COL]
                    tic_rows.append(r)

        if len(votes) == 0:
            final = "Unconstrained"
        else:
            vc = pd.Series(votes).value_counts()
            modes = vc[vc == vc.max()].index.tolist()
            final = modes[0] if len(modes) == 1 else "Mixed"
        step2[tic] = final
        if not tic_rows:                    # garantizar presencia en main
            r = pd.Series({c: np.nan for c in df_p2.columns})
            r["TIC"] = tic
            r["sector_class"] = "Unconstrained"
            tic_rows = [r]
        for r in tic_rows:
            r = r.copy()
            r["final_class_tic"] = final
            r["decision_step"] = 2
            out_rows.append(r)

    # --- Paso 3: ensamble ---------------------------------------------------
    for tic, lab in step1.items():
        for _, r in df_p2[df_p2["TIC"] == tic].iterrows():
            r = r.copy()
            r["final_class_tic"] = lab
            r["sector_class"] = lab
            r["decision_step"] = 1
            out_rows.append(r)
    for tic in no_periodo_tics:
        r = pd.Series({c: np.nan for c in df_p2.columns})
        r["TIC"] = tic
        r["final_class_tic"] = "No_periodo"
        r["sector_class"] = "No_periodo"
        r["decision_step"] = 1
        out_rows.append(r)

    main = pd.DataFrame(out_rows).reset_index(drop=True)
    main["TIC"] = main["TIC"].astype(int)
    tic_class = pd.Series({**step1, **step2,
                           **{t: "No_periodo" for t in no_periodo_tics}},
                          name="final_class_tic")
    return main, tic_class


def write_path2_catalogs(main, out_dir, suffix=""):
    """catalog_path2_main + un CSV por clase (Mixed separado), como el notebook."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main.to_csv(out_dir / f"catalog_path2_main{suffix}.csv", index=False)
    main[main["final_class_tic"] == "Mixed"].to_csv(
        out_dir / f"catalog_path2_mixed{suffix}.csv", index=False)
    for clase, sub in main.groupby("final_class_tic"):
        if clase == "Mixed":
            continue
        sub.to_csv(out_dir / f"catalog_path2_{clase}{suffix}.csv", index=False)
    return out_dir
