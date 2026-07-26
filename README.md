# MassiveStarVariability

Pipeline de variabilidad TESS para estrellas masivas, con clasificación CNN+BRF
entrenada sobre OGLE. Detecta períodos con Lomb-Scargle (LS) y autocorrelación
(ACF), construye histogramas 2D phase-folded y clasifica cada peak en 8 clases:
`ELL, M, CEP, DST, E, LPV, RR, Rndm`.

## Pipeline

```
descarga FITS (TESS-SPOC)                      notebooks 1_*
        │
clean_lightcurve()          rampas de scattered light + sigma-clip en gaps/bordes
        │
LS + ACF periodogramas      FAP Baluev (LS) / Bartlett (ACF)
        │
select_peaks()              find_peaks sobre power crudo, ventana min_peak_sep_days
        │                   → scripts/run_peaks.py → results/peaks.parquet
hist2d 32×32 + amplitude
        │
CNN (MC-dropout) → BRF      incertidumbre: sigma_top, sigma_brf, instability, entropy
        │                   → scripts/run_brf_snr.py → results/brf_mc_peaks_<model>.csv
gate (PROB_MIN, SIGMA_MAX)  y cascada Path-2 → catalogs/catalog_path2_*.csv
```

## Entornos

Dos entornos (ver `environment.txt`):

- **CNN_TESS** (python 3.8, ray, astropy 5.2.2, statsmodels): todo el pipeline
  hasta peaks. `msv.classify_brf` importa TF de forma lazy, así el resto del
  paquete funciona aquí.
- **CNN_TESS_min** (TF 2.13.1, sklearn 1.0.2): predicción CNN + BRF
  (`scripts/run_brf_snr.py`). Ojo: el joblib del BRF requiere sklearn 1.0.x —
  versiones más nuevas no lo cargan.

Instalación (editable, en ambos envs):

```bash
pip install -e .
```

## Uso rápido

1. Peaks LS+ACF (env base):

```bash
python scripts/run_peaks.py --lc lightcurves_all_OGLE.parquet \
    --out results/peaks.parquet --min-peak-sep-days 0.5 --sources ls acf
```

2. Clasificación CNN MC-dropout + BRF + gate (env TF):

```bash
python scripts/run_brf_snr.py --peaks results/peaks.parquet \
    --model Number_DST --n-iter 50 --out results/brf_mc_peaks_Number_DST.csv
```

Los paths a los pesos de la CNN (`Paper_OGLE/Weights/`) y al modelo BRF
(`balanced_random_forest_model.joblib`) se configuran en `src/msv/config.py` o
por variables de entorno `MSV_WEIGHTS` / `MSV_BRF`.

## Mapa de notebooks

En orden de pipeline (el número indica la etapa):

| Notebook | Qué muestra |
|---|---|
| `0_Validate_Preprocessing.ipynb` | **Validación visual paso a paso** de UNA curva: cruda → rampas → sigma-clip en bordes → LS/ACF → phase-fold de todos los peaks. Punto de entrada para entender el pipeline. |
| `1_TESS_variability_MassiveStarG12_preparate_data_OGLE.ipynb` | Cross-match Vizier/TIC, descarga MAST, chequeo de apertura (masivas y OGLE). |
| `1_preparate_data_VSX.ipynb` | Cross-match masivas × VSX (benchmark de validación). |
| `2_Github_TESS_variability_MassiveStarG12.ipynb` | Periodogramas + peaks + clasificación sobre las masivas. |
| `2_Github_TESS_variability_OGLE.ipynb` | Benchmark OGLE: peaks, CNN+BRF, cascada Path-2, gate. |
| `2_Github_TESS_variability_VSX.ipynb` | Benchmark VSX: métricas, barrido de τ, predicción final. |
| `3_HowToUse_CNN_BRF.ipynb` | Demo de la CNN+BRF y el gate de MC-dropout. |
| `4_Visual_Review_Pipeline.ipynb` | Auditoría end-to-end con `viz.show_sample` sobre samples aleatorios, relabelados por el gate y periódicos sobrevivientes. |

## Layout

- `src/msv/` — paquete: `config.py`, `cleaning.py`, `periodograms.py`,
  `peaks.py`, `features.py`, `classify_brf.py`, `viz.py`.
- `scripts/` — CLIs `run_peaks.py` y `run_brf_snr.py`.
- `catalogs/` — catálogos de cross-match y salidas Path-2 (versionados).
- `test_data/` — subsample de 10 TICs para smoke tests (versionado).
- `results/` — salidas derivadas (ignorado por git).
- `VALIDACION_Y_COMPARACION_MODELOS.md` — especificación del benchmark
  OGLE×TESS y del gate de incertidumbre.

## Nota sobre columnas de período

`per` es el período **detectado** (LS/ACF); `per_ogle` es el período de
referencia del catálogo OGLE. En validación usar siempre `per_ogle` — nunca
sobreescribir una con la otra.
