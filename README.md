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

Un solo entorno **MSV** (ver `environment.txt`): python 3.8, ray, astropy
5.2.2, TF 2.13.1, sklearn 1.0.2, lightkurve. Cubre todo el pipeline: peaks,
predicción CNN y BRF (`scripts/run_brf_snr.py`). Ojo: el joblib del BRF
requiere sklearn 1.0.x — versiones más nuevas no lo cargan, por eso el env
fija 1.0.2.

Instalación (editable):

```bash
pip install -e .
```

## Datos (setup en una máquina nueva)

Los ~36 GB de datos **no están en git**: viven en Dropbox, en
`~/Dropbox/MassiveStarVariability/`, para poder trabajar desde varias máquinas.

```
~/Dropbox/MassiveStarVariability/
├── raw/       27 GB  cubos/, download_paralell/, ogle_download/,
│                     CheckAperture/, lightcurves/   (re-descargables de MAST/OGLE)
└── derived/  8.4 GB  lightcurves_all{,_OGLE}.parquet, peaks*.parquet,
                      periodograms_{ls,acf}.parquet, train_number_M.csv
```

Para dejar una máquina lista: clonar el repo, esperar a que Dropbox termine de
sincronizar y correr

```bash
python scripts/link_data.py      # --check para solo diagnosticar
```

Eso crea en la raíz del repo un symlink por dataset, así las rutas relativas de
los notebooks (`peaks.parquet`, `cubos/...`) funcionan sin cambios. Es
idempotente y nunca pisa un archivo real.

`config.DATA_DIR` localiza la carpeta buscando, en orden: `$MSV_DATA_DIR`, los
candidatos de `_DATA_DIR_CANDIDATES` en `src/msv/config.py` (añadir ahí la ruta
de cada máquina nueva) y, como fallback, la raíz del repo (layout antiguo).
Cada dataset admite además su propio override `MSV_*` — útil para correr en un
cluster sin Dropbox:

```bash
MSV_DATA_DIR=/scratch/$USER/msv python scripts/run_peaks.py ...
```

Importante: si usás *selective sync* / "online-only" sobre esta carpeta,
Dropbox reemplaza los archivos por placeholders y los symlinks apuntan a nada.
Mantenela disponible offline.

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
| `2_Github_TESS_variability_VSX.ipynb` | Benchmark VSX: métricas, barrido de τ, predicción final y **revisión visual** de las 567 estrellas con período VSX. |
| `3_HowToUse_CNN_BRF.ipynb` | Demo de la CNN+BRF y el gate de MC-dropout. |
| `4_Visual_Review_Pipeline.ipynb` | Auditoría end-to-end con `viz.show_sample` sobre samples aleatorios, relabelados por el gate y periódicos sobrevivientes. |

## Layout

- `src/msv/` — paquete: `config.py`, `cleaning.py`, `periodograms.py`,
  `peaks.py`, `features.py`, `classify_brf.py`, `viz.py`, `review.py`.
- `scripts/` — CLIs `run_peaks.py`, `run_brf_snr.py`, `build_vsx_review.py` y
  `link_data.py` (setup de datos por máquina).
- `catalogs/` — catálogos de cross-match y salidas Path-2 (versionados).
- `test_data/` — subsample de 10 TICs para smoke tests (versionado).
- `results/` — salidas derivadas (ignorado por git).
- `VALIDACION_Y_COMPARACION_MODELOS.md` — especificación del benchmark
  OGLE×TESS y del gate de incertidumbre.

## Revisión visual del benchmark VSX

Los `Type`/`Period` de VSX no son ground truth ciego (surveys heterogéneos,
tipos cajón como `MISC`/`VAR`, períodos que no dominan la curva TESS). Las
567 masivas con período se marcan a mano en `catalogs/vsx_visual_review.csv`
(versionado): `vis_verdict` valida el período — y con él el `label` —, y
`vis_shape` valida la morfología — y con ella el `expected_class`.

```
python scripts/build_vsx_review.py --model Number_DST   # tabla + caché de curvas
```

y el revisor (`msv.review.VisualReviewer`) se usa desde
`2_Github_TESS_variability_VSX.ipynb`. Las métricas del benchmark se reportan
restringidas a `vis_verdict == "ok"`.

## Nota sobre columnas de período

`per` es el período **detectado** (LS/ACF); `per_ogle` es el período de
referencia del catálogo OGLE. En validación usar siempre `per_ogle` — nunca
sobreescribir una con la otra.
