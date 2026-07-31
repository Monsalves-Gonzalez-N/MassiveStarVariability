# Validación con benchmark OGLE + comparación de 7 modelos + gate por σ (MC-dropout)

Instrucciones para añadir **5 secciones nuevas** al final de
`2_Github_TESS_variability_OGLE.ipynb`. El objetivo es, usando el cross-match OGLE como
**benchmark** (`types`, `per`):

1. Auditar **falsos positivos** en las estrellas no detectables (`label=0`).
2. Verificar **período + clase** en las detectables (`label=1`).
3. Elegir el **mejor de 7 sets de pesos** (`Paper_OGLE/Weights/`).
4. Probar si el **σ de MC-dropout** sirve como *gate* para matar los FP.

> **Kernel:** `MSV` (sklearn 1.0.2 + tf 2.13.1; antes `CNN_TESS_min`), igual que la sección CNN+BRF.

---

## 1. Contexto y problema

El pipeline CNN→BRF clasifica cada *peak* en 8 clases
`['ELL','M','CEP','DST','E','LPV','RR','Rndm']` y **Path 2** colapsa a una clase por TIC
(`final_class_tic`). El problema: estrellas que **no pueden** tener período detectable en TESS
(sector ≈ 27 d) salen como E/ELL/DST/RR.

`df_1` (celda `4e035dc4`) ya define el `label` con el benchmark OGLE:

| label | criterio | significado | composición (147 TICs) |
|-------|----------|-------------|------------------------|
| `1` | `per < 13 d` | período corto, **detectable** | 34 TICs: `dsct×18`, `cep×11`, `rrlyr×5` |
| `0` | `per ≥ 13 d` (o NaN) | **no detectable** (Miras ~367 d, etc.) | 113 TICs: `Mira×105`, `cep×5`, `lpv×3` |

**FP medidos hoy** (`batchBalanced_Number_M`, peaks de `label=0`):
`Rndm 4723` (ok), `LPV 42` (ok), **`E 2087`, `ELL 579`, `DST 318`, `CEP 170`, `RR 38` (FP)**
→ ~3200/7800 peaks de `label=0` son falsos positivos.

---

## 2. Decisiones de diseño

- **Ubicación:** secciones nuevas en `2_Github_TESS_variability_OGLE.ipynb`.
- **Ranking:** sobre el **pipeline completo CNN→BRF** (no el CNN aislado).
  - ⚠️ **Caveat:** el BRF (`balanced_random_forest_model.joblib`) se entrenó sobre las probs de
    **un** CNN. Pasar los 7 CNN por el mismo BRF mide el **pipeline**, no el CNN puro: los CNN
    ≠ M ven un BRF parcialmente fuera de distribución. Tenerlo presente al leer la tabla.
- **σ (MC-dropout) → gate:** relabelar a `Rndm` los peaks inciertos **antes** de Path-2. El BRF
  se alimenta con la pasada **determinista** (`training=False`); el σ sale de N pasadas con
  `training=True` (el `Dropout(0.3)` ya existe en `make_model`).

---

## 3. Requisitos previos

Ejecutar el notebook hasta el final de la sección Path 2. Deben existir en memoria/disco:

- `df_1` con columnas `TIC, types, per, label` (celda `4e035dc4`).
- `make_model`, `CLASS_NAMES`, `brf` y las rutas `PEAKS_OGLE`, `LC_OGLE`, `BRF_PATH`,
  `CATALOG` (celdas `ceddacd2` y `eaa94c43`).
- `peaks_statsmodels.parquet` y `lightcurves_all_OGLE.parquet`.

Las funciones `phase_fold_hist2d_log` y `amplitude_of` ya están en la celda `efac30ca`; aquí se
re-empaquetan dentro de `build_cube` para que el bloque sea autocontenido.

---

## 4. Reglas de validación (a nivel TIC, sobre `final_class_tic`)

**`label=0` — no debe reclamarse período corto:**

| resultado | clases | veredicto |
|-----------|--------|-----------|
| aceptable | `Rndm`, `Irregular`, `No_periodo`, `Unconstrained` | OK (sin período espurio) |
| falso positivo | `ELL`, `E`, `CEP`, `DST`, `RR`, `M`, `Mixed` | **FP** |

(`Irregular` es el nombre Path-2 de LPV; `No_periodo`/`Unconstrained` = no se reclamó período.)

**`label=1` — debe acertar período y clase:**

- `class_correct = final_class_tic == expected_class` (`dsct→DST`, `cep→CEP`, `rrlyr→RR`).
- `period_match(P_det, P_ogle)` ∈ {`exact`, `harmonic`, `none`} con `rtol=5%` y armónicos
  `×{2, ½, 3, ⅓}`. `P_det` = `per` del peak ganador (mayor `brf_prob` entre los peaks cuya
  `sector_class` == `final_class_tic`).
- `fully_correct = class_correct and period_match != "none"`.

> ⚠️ **Dos `per` distintos — no confundirlos:**
> - **`per_ogle`** = período OGLE (benchmark), vive en `df_1`/`gt`. Rango real: 0.06–716.7 d.
> - **`per`** (detectado) = período del peak del periodograma, vive en `peaks`/`predictions`/`main_df`.
>   Rango ≤ ~13.5 d (limitado por `baseline/2` de un sector TESS).
>
> No se sobrescriben (están en DataFrames distintos), pero comparten nombre `per`. Por eso `gt`
> renombra el suyo a **`per_ogle`**: así un `merge(..., on="TIC")` nunca produce un `per_x/per_y`
> silencioso. `evaluate` compara `p_det` (de `main_df`) contra `per_ogle` (de `gt`).

---

## 5. Sección A — Ground truth + helpers de validación

> **Celda markdown:** `## Validación benchmark — ground truth y reglas`

```python
import numpy as np
import pandas as pd

TYPE2CLASS = {"Mira":"M", "lpv":"LPV", "rrlyr":"RR", "ELL":"ELL",
              "ecl":"E", "dsct":"DST", "cep":"CEP"}
PERIODIC      = {"ELL", "M", "CEP", "DST", "E", "RR"}          # clases con período corto
ACCEPT_LABEL0 = {"Rndm", "Irregular", "No_periodo", "Unconstrained"}  # OK para label=0

def build_ground_truth(df_1):
    """Una fila por TIC con types/per/label de OGLE + clase esperada (solo label==1)."""
    gt = df_1[["TIC", "types", "per", "label"]].drop_duplicates("TIC").reset_index(drop=True).copy()
    # OJO: el `per` de df_1 es el período OGLE (benchmark). Se renombra a `per_ogle` para que
    # NUNCA se confunda ni colisione con el `per` DETECTADO (periodograma) de peaks/predictions.
    gt = gt.rename(columns={"per": "per_ogle"})
    gt["TIC"] = gt["TIC"].astype(int)
    gt["label"] = gt["label"].astype(int)
    gt["expected_class"] = gt["types"].map(TYPE2CLASS)
    return gt

def period_match(p_det, p_ogle, rtol=0.05, harmonics=(2, 0.5, 3, 1/3)):
    """exact si |p_det-p_ogle|/p_ogle<=rtol; harmonic si p_det~k*p_ogle; si no, none."""
    if not (np.isfinite(p_det) and np.isfinite(p_ogle)) or p_ogle <= 0:
        return "none"
    if abs(p_det - p_ogle) / p_ogle <= rtol:
        return "exact"
    for k in harmonics:
        if abs(p_det - k * p_ogle) / (k * p_ogle) <= rtol:
            return "harmonic"
    return "none"

def winning_period(main_df, tic, final_class):
    """per del peak ganador (mayor brf_prob) cuya sector_class == final_class."""
    sub = main_df[(main_df["TIC"] == tic) & (main_df["sector_class"] == final_class)]
    sub = sub.dropna(subset=["per"])
    if sub.empty:
        return np.nan
    return float(sub.sort_values("brf_prob", ascending=False).iloc[0]["per"])

gt = build_ground_truth(df_1)
print(f"Ground truth: {len(gt)} TICs | label=1: {(gt['label']==1).sum()} | label=0: {(gt['label']==0).sum()}")
print(gt.loc[gt['label']==1, 'expected_class'].value_counts().to_string())
```

---

## 6. Sección B — Funciones reutilizables (cube, MC, clasificación, Path-2, evaluación)

> **Celda markdown:** `## Funciones: cube, MC-dropout, clasificación por modelo, Path-2, evaluación`

```python
import tensorflow as tf

N_EDGES = 33

def phase_fold_hist2d_log(time, flux, period, n_edges=N_EDGES):
    fase = np.mod(time, period) / period
    bins_x = np.linspace(0.0, 1.0, n_edges)
    bins_y = np.linspace(flux.min(), flux.max(), n_edges)
    h, _, _ = np.histogram2d(fase, flux, bins=(bins_x, bins_y))
    hmax = h.max()
    if hmax > 0:
        h = np.log1p(h) / np.log1p(hmax)
    return h.T[::-1].astype(np.float32)

def amplitude_of(f):
    if f.min() > 0:
        return float(np.absolute(-2.5 * np.log10(f.max() / f.min())))
    return float("nan")

def build_cube(peaks_ogle, lc_groups):
    """hist2d (N,32,32,1) + amplitud por peak. INDEPENDIENTE del modelo -> se calcula 1 vez."""
    N = len(peaks_ogle)
    X = np.zeros((N, 32, 32, 1), dtype=np.float32)
    amp = np.full(N, np.nan, dtype=np.float64)
    amp_cache, n_skip = {}, 0
    for i, row in enumerate(peaks_ogle.itertuples(index=False)):
        key = (row.TIC, row.sector)
        lc = lc_groups.get(key)
        if lc is None:
            n_skip += 1; continue
        t = lc["Time"].to_numpy(); f = lc["flux"].to_numpy()
        m = np.isfinite(t) & np.isfinite(f); t, f = t[m], f[m]
        if len(t) < 20:
            n_skip += 1; continue
        if key not in amp_cache:
            amp_cache[key] = amplitude_of(f)
        amp[i] = amp_cache[key]
        X[i, ..., 0] = phase_fold_hist2d_log(t, f, float(row.per))
    print(f"Cube: {X.shape} | peaks sin curva válida: {n_skip}")
    return X, amp

def cnn_mc_predict(model, X, n_iter=50):
    """N pasadas con Dropout activo (training=True) -> mean,std (N_peaks,8)."""
    preds = np.stack([model(X, training=True).numpy() for _ in range(n_iter)])  # (n_iter,N,8)
    return preds.mean(0), preds.std(0)

def classify_with_model(weights_dir, X, amp, peaks_ogle, brf, n_mc=50):
    """Carga pesos -> probs deterministas (->BRF) + MC sigma. Devuelve tabla por peak."""
    tf.keras.backend.clear_session()
    m = make_model()
    m.load_weights(f"{weights_dir}/cp.ckpt")

    probs_det = m.predict(X, batch_size=512, verbose=0)          # determinista -> BRF
    mc_mean, mc_std = cnn_mc_predict(m, X, n_mc)                  # incertidumbre
    top = mc_mean.argmax(1)
    sigma_top = mc_std[np.arange(len(mc_mean)), top]
    mc_conf = mc_mean[np.arange(len(mc_mean)), top]

    df = peaks_ogle[["TIC", "sector", "source", "per", "power", "prominence", "width"]].copy()
    df["amplitude"] = amp
    for j, c in enumerate(CLASS_NAMES):
        df[f"cnn_{c}"] = probs_det[:, j]

    feat = pd.DataFrame(probs_det, columns=CLASS_NAMES)
    feat["per"] = peaks_ogle["per"].to_numpy()
    feat["amplitud"] = amp
    feat = feat[list(brf.feature_names_in_)]                     # orden EXACTO del BRF
    mask = feat.notna().all(axis=1).to_numpy()                  # descarta flujo<=0 (amp NaN)

    pred = np.full(len(df), -1, dtype=int)
    proba = np.full(len(df), np.nan, dtype=float)
    if mask.any():
        pred[mask] = brf.predict(feat.loc[mask])
        pr = brf.predict_proba(feat.loc[mask])
        proba[mask] = pr[np.arange(mask.sum()), pred[mask]]

    df["brf_class"] = [CLASS_NAMES[c] if c >= 0 else None for c in pred]
    df["brf_prob"] = proba
    df["sigma_top"] = sigma_top
    df["mc_conf"] = mc_conf
    return df

def run_path2(df_pred, catalog_tics, threshold=0.8, seed=42):
    """Cascada Path-2 (idéntica a celdas eaa94c43->fdb33036) empaquetada. En memoria."""
    rng = np.random.default_rng(seed)
    CLASS_COL, PROB_COL = "brf_class", "brf_prob"
    PER_COL, POWER_COL, SECTOR_COL = "per", "power", "sector"

    df = df_pred.copy()
    df["cube_idx"] = np.arange(len(df))
    df = df[df[CLASS_COL].notna()].reset_index(drop=True)
    peak_tics = set(df["TIC"].astype(int).unique())
    no_periodo_tics = sorted(set(int(t) for t in catalog_tics) - peak_tics)

    def classify_tic_step1(g):
        classes = set(g[CLASS_COL]); has_over = bool((g[PROB_COL] >= threshold).any())
        if classes == {"Rndm"} and has_over:
            return "Rndm"
        if classes <= {"Rndm", "LPV"} and "LPV" in classes and has_over:
            return "Irregular"
        return None

    step1, fallthrough = {}, []
    for tic, g in df.groupby("TIC"):
        lab = classify_tic_step1(g)
        if lab is None:
            fallthrough.append(tic)
        else:
            step1[tic] = lab

    def _acf_winner(s): return s.sort_values([PROB_COL, POWER_COL], ascending=[False, False]).iloc[0]
    def _ls_winner(s):  return s.sort_values([PROB_COL, POWER_COL], ascending=[False, True]).iloc[0]

    out_rows, step2 = [], {}
    for tic in fallthrough:
        g = df[(df["TIC"] == tic) & (df[CLASS_COL].isin(PERIODIC))]
        votes, tic_rows = [], []
        for sector, sg in g.groupby(SECTOR_COL):
            acf = sg[sg["source"] == "ACF"]; ls = sg[sg["source"] == "LS"]
            a = _acf_winner(acf) if len(acf) else None
            l = _ls_winner(ls) if len(ls) else None
            if not bool((sg[PROB_COL] >= threshold).any()):           # Unconstrained
                if a is not None and l is not None and a[PROB_COL] == l[PROB_COL]:
                    keep = [a] if rng.choice(["ACF", "LS"]) == "ACF" else [l]
                else:
                    keep = [r for r in (a, l) if r is not None]
                for r in keep:
                    r = r.copy(); r["sector_class"] = "Unconstrained"; tic_rows.append(r)
            else:                                                      # aporta voto(s)
                if a is not None and l is not None:
                    if   a[PROB_COL] > l[PROB_COL]: win = [a]
                    elif l[PROB_COL] > a[PROB_COL]: win = [l]
                    else:                           win = [a, l]
                else:
                    win = [r for r in (a, l) if r is not None]
                for r in win:
                    votes.append(r[CLASS_COL])
                    r = r.copy(); r["sector_class"] = r[CLASS_COL]; tic_rows.append(r)

        if len(votes) == 0:
            final = "Unconstrained"
        else:
            vc = pd.Series(votes).value_counts(); modes = vc[vc == vc.max()].index.tolist()
            final = modes[0] if len(modes) == 1 else "Mixed"
        step2[tic] = final
        if not tic_rows:
            r = pd.Series({c: np.nan for c in df.columns}); r["TIC"] = tic
            r["sector_class"] = "Unconstrained"; tic_rows = [r]
        for r in tic_rows:
            r = r.copy(); r["final_class_tic"] = final; r["decision_step"] = 2; out_rows.append(r)

    for tic, lab in step1.items():
        for _, r in df[df["TIC"] == tic].iterrows():
            r = r.copy(); r["final_class_tic"] = lab; r["sector_class"] = lab; r["decision_step"] = 1
            out_rows.append(r)
    for tic in no_periodo_tics:
        r = pd.Series({c: np.nan for c in df.columns})
        r["TIC"] = tic; r["final_class_tic"] = "No_periodo"; r["sector_class"] = "No_periodo"; r["decision_step"] = 1
        out_rows.append(r)

    main = pd.DataFrame(out_rows).reset_index(drop=True)
    main["TIC"] = main["TIC"].astype(int)
    tic_class = {**step1, **step2, **{t: "No_periodo" for t in no_periodo_tics}}
    return main, tic_class

def evaluate(tic_class, main_df, gt, w=(0.5, 0.25, 0.25)):
    """Aplica reglas Sección 4 y devuelve métricas + score combinado."""
    g = gt.set_index("TIC")
    n0 = n0_fp = 0; fp_by_class = {}
    n1 = n1_class = n1_exact = n1_harm = n1_full = 0
    detail = []
    for tic, lab in tic_class.items():
        if tic not in g.index:
            continue
        row = g.loc[tic]
        if int(row["label"]) == 0:
            n0 += 1
            if (lab in PERIODIC) or (lab == "Mixed"):
                n0_fp += 1; fp_by_class[lab] = fp_by_class.get(lab, 0) + 1
        else:
            n1 += 1
            exp = row["expected_class"]; class_ok = (lab == exp)
            # p_det = período DETECTADO (main_df, periodograma); per_ogle = benchmark OGLE
            p_det = winning_period(main_df, tic, lab) if lab in PERIODIC else np.nan
            pm = period_match(p_det, float(row["per_ogle"]))
            n1_class += int(class_ok)
            n1_exact += int(pm == "exact"); n1_harm += int(pm == "harmonic")
            n1_full += int(class_ok and pm != "none")
            detail.append((tic, exp, lab, float(row["per_ogle"]), p_det, pm, class_ok))
    FP_rate_0 = n0_fp / n0 if n0 else np.nan
    class_acc_1 = n1_class / n1 if n1 else np.nan
    period_recovery_1 = (n1_exact + n1_harm) / n1 if n1 else np.nan
    fully_correct_1 = n1_full / n1 if n1 else np.nan
    score = w[0]*(1 - FP_rate_0) + w[1]*class_acc_1 + w[2]*period_recovery_1
    return {"FP_rate_0": FP_rate_0, "fp_by_class": fp_by_class,
            "class_acc_1": class_acc_1, "period_recovery_1": period_recovery_1,
            "fully_correct_1": fully_correct_1, "score": score,
            "n0": n0, "n1": n1, "detail": pd.DataFrame(
                detail, columns=["TIC","expected","final","per_ogle","per_det","period_match","class_ok"])}
```

---

## 7. Sección C — Loop sobre los 7 modelos (baseline, sin gate)

> **Celda markdown:** `## Comparación de los 7 modelos (baseline CNN→BRF)`

```python
WEIGHTS_DIR = "/home/nicolas/nico/git/Paper_OGLE/Weights"
MODELS = ["Number_CEP", "Number_DST", "Number_ELL", "Number_M",
          "batchBalanced_Number_DST", "batchBalanced_Number_ELL", "batchBalanced_Number_M"]
N_MC = 30   # subir a 50-100 para el reporte final

# Datos comunes (se leen 1 vez)
peaks_ogle = pd.read_parquet(PEAKS_OGLE).reset_index(drop=True)
lc_all = pd.read_parquet(LC_OGLE, columns=["Time", "flux", "TIC", "sector"])
lc_groups = {k: v for k, v in lc_all.groupby(["TIC", "sector"], sort=False)}
catalog_tics = set(pd.read_csv(CATALOG)["TIC"].astype(int).unique())

X, amp = build_cube(peaks_ogle, lc_groups)     # INDEPENDIENTE del modelo

preds_by_model, rows = {}, []
for name in MODELS:
    print(f"\n=== {name} ===")
    dfp = classify_with_model(f"{WEIGHTS_DIR}/{name}", X, amp, peaks_ogle, brf, n_mc=N_MC)
    preds_by_model[name] = dfp
    main, tic_class = run_path2(dfp, catalog_tics, threshold=0.8, seed=42)
    met = evaluate(tic_class, main, gt)
    rows.append({"model": name, "FP_rate_0": met["FP_rate_0"],
                 "class_acc_1": met["class_acc_1"], "period_recovery_1": met["period_recovery_1"],
                 "fully_correct_1": met["fully_correct_1"], "score": met["score"]})
    print(f"FP_rate_0={met['FP_rate_0']:.3f} | class_acc_1={met['class_acc_1']:.3f} | "
          f"period_rec_1={met['period_recovery_1']:.3f} | score={met['score']:.3f}")
    print(f"FP por clase (label=0): {met['fp_by_class']}")

comp = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
comp.to_csv("model_comparison.csv", index=False)
print("\n=== Ranking (sin gate) ===")
print(comp.to_string(index=False))
```

Graficar (opcional): barras de `FP_rate_0` y `fully_correct_1` por modelo; matriz de confusión
sobre `label=1` con `met["detail"]` del modelo top (columnas `expected` vs `final`).

---

## 8. Sección D — MC-dropout σ + gate

> **Celda markdown:** `## ¿Sirve el σ de MC-dropout como gate contra los FP?`

```python
import matplotlib.pyplot as plt

# --- Diagnóstico: ¿los FP (label=0) tienen σ más alto que los aciertos (label=1)? ---
diag_model = "batchBalanced_Number_M"        # o el mejor de la Sección C
dfp = preds_by_model[diag_model]
mp = dfp.merge(gt[["TIC", "label", "expected_class"]], on="TIC", how="left")

sig_fp0 = mp[(mp["label"] == 0) & (mp["brf_class"].isin(PERIODIC))]["sigma_top"].dropna()
sig_ok1 = mp[(mp["label"] == 1) & (mp["brf_class"] == mp["expected_class"])]["sigma_top"].dropna()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(sig_fp0, bins=40, alpha=0.6, density=True, label=f"FP label=0 ({len(sig_fp0)})")
ax.hist(sig_ok1, bins=40, alpha=0.6, density=True, label=f"OK label=1 ({len(sig_ok1)})")
ax.set_xlabel("sigma_top (σ MC-dropout de la clase top)"); ax.set_ylabel("densidad")
ax.legend(); ax.set_title(f"{diag_model}: σ de FP vs aciertos"); plt.tight_layout(); plt.show()
print(f"σ medio  FP label=0: {sig_fp0.mean():.4f} | OK label=1: {sig_ok1.mean():.4f}")

# --- Gate + barrido de umbral τ ---
def apply_gate(df_pred, tau, mode="relabel"):
    """Peaks con sigma_top>tau -> 'Rndm' (relabel) o descartados (drop), antes de Path-2."""
    d = df_pred.copy()
    bad = d["sigma_top"] > tau
    if mode == "relabel":
        d.loc[bad, "brf_class"] = "Rndm"
    else:
        d = d[~bad].reset_index(drop=True)
    return d

taus = np.quantile(dfp["sigma_top"].dropna(), np.linspace(0.10, 0.95, 18))
sweep = []
for tau in taus:
    main_g, tc_g = run_path2(apply_gate(dfp, tau), catalog_tics, threshold=0.8, seed=42)
    m = evaluate(tc_g, main_g, gt)
    sweep.append({"tau": tau, "FP_rate_0": m["FP_rate_0"], "recall_1": m["fully_correct_1"]})
sweep = pd.DataFrame(sweep)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(sweep["tau"], sweep["FP_rate_0"], "o-", label="FP_rate_0 (↓ mejor)")
ax.plot(sweep["tau"], sweep["recall_1"], "s-", label="recall_1 = fully_correct_1 (↑ mejor)")
ax.set_xlabel("τ (umbral de σ)"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title(f"{diag_model}: trade-off del gate"); plt.tight_layout(); plt.show()
print(sweep.to_string(index=False))

# Elegir τ*: p.ej. el τ que minimiza FP_rate_0 manteniendo recall_1 >= recall_sin_gate - margen
TAU_STAR = float(sweep.sort_values("FP_rate_0").iloc[0]["tau"])   # ajustar a criterio
print("τ* sugerido:", TAU_STAR)
```

> Si los histogramas NO se separan, el σ **no** discrimina FP y el gate no ayudará — reportarlo
> igual (resultado negativo válido). Alternativas a explorar: usar `mc_conf` (confianza media) o
> entropía predictiva en vez de `sigma_top`.

---

## 9. Sección E — Selección final (con gate) y escritura del ganador

> **Celda markdown:** `## Modelo ganador: re-evaluación con gate y catálogos finales`

```python
import os

# Re-evaluar los 7 modelos CON gate(τ*) y comparar contra el baseline
rows_gate = []
for name in MODELS:
    main_g, tc_g = run_path2(apply_gate(preds_by_model[name], TAU_STAR), catalog_tics)
    m = evaluate(tc_g, main_g, gt)
    rows_gate.append({"model": name, "FP_rate_0": m["FP_rate_0"],
                      "class_acc_1": m["class_acc_1"], "period_recovery_1": m["period_recovery_1"],
                      "fully_correct_1": m["fully_correct_1"], "score": m["score"]})
comp_gate = pd.DataFrame(rows_gate).sort_values("score", ascending=False).reset_index(drop=True)
comp_gate.to_csv("model_comparison_gated.csv", index=False)
print("=== Ranking (con gate τ*) ===")
print(comp_gate.to_string(index=False))

# Comparación con vs sin gate
cmp = comp.merge(comp_gate, on="model", suffixes=("_base", "_gate"))
print("\n=== FP_rate_0: base vs gate ===")
print(cmp[["model", "FP_rate_0_base", "FP_rate_0_gate",
           "fully_correct_1_base", "fully_correct_1_gate"]].to_string(index=False))

# Escribir catálogos SOLO del modelo ganador (con gate)
BEST = comp_gate.iloc[0]["model"]
OUT_DIR = "catalogs"
best_pred = apply_gate(preds_by_model[BEST], TAU_STAR)
best_pred.to_csv(f"OGLE_CNN_BRF_predictions_{BEST}.csv", index=False)
main_best, tic_best = run_path2(best_pred, catalog_tics)
main_best.to_csv(os.path.join(OUT_DIR, f"catalog_path2_main_{BEST}.csv"), index=False)
print(f"\nGanador: {BEST} | TICs: {main_best['TIC'].nunique()}")
print(pd.Series(tic_best).value_counts().to_string())
```

---

## 10. Verificación

1. Ejecutar el notebook top-to-bottom en kernel **`MSV`** (antes `CNN_TESS_min`).
2. `gt` tiene **147** TICs (34 `label=1`, 113 `label=0`); `build_cube` reproduce el `X` actual
   (mismo `shape` y nº de peaks sin curva que la celda `fc8e35c7`).
3. **Baseline:** `FP_rate_0` de `batchBalanced_Number_M` debe ser alto (consistente con
   `E=2087`/`ELL=579` de peaks); `model_comparison.csv` tiene 7 filas.
4. **Gate:** al bajar `τ`, `FP_rate_0` baja; confirmar que `recall_1` no colapsa.
5. Empezar con `N_MC=30` (rapidez) y subir a 50–100 para el reporte final.
6. `tf.keras.backend.clear_session()` ya está dentro de `classify_with_model` (evita acumular
   memoria entre los 7 modelos).

## 11. Notas y extensiones

- **Caveat BRF** (Sección 2): la tabla mide el pipeline CNN→BRF, no el CNN aislado. Para una
  comparación CNN-pura, añadir métricas con `argmax(probs_det)` por peak (sin BRF) en paralelo.
- **σ alternativos:** además de `sigma_top`, probar entropía predictiva
  `-(mc_mean*np.log(mc_mean+1e-9)).sum(1)` o `1 - mc_conf` como señal de gate.
- **Variante "media MC → BRF":** si más adelante se quiere, alimentar `mc_mean` (en vez de
  `probs_det`) al BRF; aquí se mantuvo determinista por decisión de diseño.
- **Score:** los pesos `w=(0.5,0.25,0.25)` priorizan reducir FP; ajustables en `evaluate`.
```
