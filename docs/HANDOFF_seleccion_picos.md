# Handoff — refactor de la selección de picos del ACF (2026-09-02)

Estado al cerrar la sesión: **todo el trabajo está sin commitear**. Este documento
es el punto de retomada para una sesión nueva.

---

## 1. Qué se hizo

Se auditó `select_peaks_acf` con un PDF de revisión sobre 50 pares (TIC, sector)
al azar y se encontraron **ocho defectos**. Todos corregidos, con las constantes
documentadas en `src/msv/config.py`.

| # | Síntoma | Causa | Corrección |
|---|---|---|---|
| 1 | 25/50 estrellas con 0 picos, algunas con oscilación de amplitud 0.5 | `bartlett_confint=True` usa `Var(r_k) ~ (1/N)(1+2·Σr_j²)`, suma acumulativa: la señal infla la banda a lags mayores y se esconde a sí misma | `ACF_BARTLETT_CONFINT = False` → banda de ruido blanco constante `z/√N` |
| 2 | Estrella de ruido puro pasaba 26 picos | `prominence = 0.20·max(power)` se normaliza por la SEÑAL: menos señal → umbral más permisivo | `ACF_PROMINENCE_K_FAP = 2.0` → `prominence ≥ 2·fap[lag]` |
| 3 | ~16 excursiones esperadas por azar y por estrella | `FAP_ALPHA` es 3σ **por lag**, sin corrección por multiplicidad sobre ~6000 lags | `ACF_TRIALS_CORRECTION = True` → `α/n_lags` → banda efectiva ~5.05σ |
| 4 | Spikes de 1-2 muestras pasando el filtro | En ruido denso la prominencia no discrimina (pico a +3.4σ con valle vecino en −3σ → prominencia ~6σ automática) | `ACF_WIDTH_FRAC = 0.05` como array a `width` de `find_peaks`, con techo `ACF_WIDTH_MAX_SAMPLES = 20` |
| 5 | TIC 316947536 reportaba el **5º armónico** como período | `ΔP > k·P²/T` como radio de exclusión crece con P²; en 10 d valía 12.2 d, más que todo el rango | NMS en FRECUENCIA, `|Δf| > k/T`, y `ACF_RAYLEIGH_K` de 3.0 a **1.0** |
| 6 | Cadenas armónicas adelgazadas, sobrevivía uno suelto que parecía ruido | La separación entre armónicos es `f₀/(n(n+1))`, cae bajo `1/T` desde n≈5 y el NMS los arrasaba | La serie se identifica **antes** del NMS y queda exenta |
| 7 | TIC 316187504 marcaba 14 de 21 armónicos | `HARMONIC_MAX_ORDER = 12` fijo; sin etiqueta no hay exención del NMS | `HARMONIC_MAX_ORDER = None` → se deriva de `ceil(max(per)/P0)` |
| 8 | Impares del peine tratados como picos sueltos | El peine del ACF tiene paso `P₀/2`, no `P₀` (eclipsante con eclipses de distinta profundidad) | Columna `comb_order`; la exención cubre todo el peine |

**Extra:** el período de la serie ya no es el del pico 1× sino la pendiente
`P = Σ(n·Pₙ)/Σ(n²)` sobre todos los armónicos. En TIC 13785212 el pico 1× está
corrido y el ajuste da 1.807 d en vez de 1.875.

### Regla operativa acordada

> Si es evidente que son armónicos, se presenta uno solo. Ante cualquier duda se
> presentan todos y decide la red sobre los phase-folds.

Codificada en `label_harmonics`: la serie se declara **solo si es inequívoca** —
`HARMONIC_MIN_RUN = 3` armónicos consecutivos, residuo mediano ≤ `HARMONIC_FIT_TOL = 0.02`
contra el período ajustado, y el pico más prominente dentro de la serie. Si falla
cualquiera, `harmonic_order` queda todo `<NA>` y no se pliega nada.

`candidate_periods` emite además **dos** candidatos (`P₀` y `P₀/2`) cuando el peine
tiene miembros impares, en vez de resolver la ambigüedad de medio período a ciegas.

### Resultado sobre las 50 (`--seed 7`)

```
picos          275  = armónicos 156 + impares del peine 28 + fuera de la serie 92
sin picos        8/50
con serie       18/50
phase-folds    113      (18/50 colapsan a 1 candidato, 3/50 llevan P0/2)
```

### Lo que se revisó y resultó estar BIEN (no tocar)

- **La banda de ruido blanco está bien calibrada.** En las 8 estrellas sin picos
  (donde la cola del ACF es ruido medible) la dispersión real da ×0.84 mediana
  contra `z/√N`. No hace falta modelar un nulo de ruido rojo.
- **La corrección `1/√(N−k)` no cambia nada** (167 → 167 picos, 0/50 afectadas).
- **El LS no tiene el problema que se le atribuyó.** La FAP de Baluev ya es
  global (incluye look-elsewhere por construcción) y el `max(FP, 0.01)` la hace
  más conservadora todavía. La condición `power - window > fap` sí filtra:
  solo 4 de 579 picos tienen contaminación de ventana apreciable.

---

## 2. Archivos tocados (sin commitear)

```
M  src/msv/config.py         constantes nuevas + PULSATING/CLASS_GROUPS/MC_ITER
M  src/msv/peaks.py          _rayleigh_nms en frecuencia, width_frac,
                             label_harmonics, candidate_periods
M  src/msv/periodograms.py   bartlett_confint, trials_correction,
                             white_noise_band, trials_corrected_band
M  src/msv/classify_brf.py   _load_tf_checkpoint (fallback Keras 3)
?? scripts/build_peak_review_pdfs.py
?? scripts/build_phasefold_review.py
?? scripts/build_cleaning_review.py
```

`white_noise_band` y `trials_corrected_band` des-inflan las bandas ya guardadas
en `periodograms_acf.parquet` sin reprocesar 40M de filas: como
`fap[k]² = c²(1 + 2Σ_{j<k} r_j²)`, diferenciando en k se despeja `c²` (recuperación
exacta, dispersión ~1e-14).

---

## 3. Entorno en este Mac

Ningún env cumple `environment.txt` completo. El pipeline corre en **tres pasos**
entre dos envs:

| | CNN_TESS | tf_env |
|---|---|---|
| Python | 3.8.20 arm64 | 3.13 |
| sklearn | **1.0.2** (bajado en esta sesión desde 1.3.2) | — |
| imblearn | 0.10.1 (instalado en esta sesión) | — |
| TensorFlow | — | 2.20 / Keras 3.13 |
| astropy, statsmodels, pyarrow, lightkurve | sí | no |

Siempre con `PYTHONPATH=src`.

- **Pesos CNN:** `~/ViT_VariableStars/pretrained/keras_checkpoints/` (7 checkpoints).
  Pasar `MSV_WEIGHTS=` — el default de `config.py` apunta a un path de Linux.
  Keras 3 no lee los `cp.ckpt`: lo resuelve `_load_tf_checkpoint` (ya en el repo).
- **BRF:** `~/Dropbox/MassiveStarVariability/models/balanced_random_forest_model.joblib`
  (1.35 GB, bajado con gdown). Pasar `MSV_BRF=`. Requiere sklearn < 1.3.
- `tf_env` no puede `import msv` (falta astropy): hay un shim que carga
  `msv.classify_brf` sin ejecutar el `__init__`.

### Dropbox

Casi todo `derived/` y **los 13483 FITS de `raw/download_paralell/`** son
placeholders online-only de 0 bytes. Leerlos desde la terminal NO los hidrata;
hay que hacerlo desde Finder (click derecho → "Hacer disponible sin conexión").

Locales: `periodograms_acf.parquet` (1.1 GB) y `periodograms_ls.parquet` (5.6 GB).

**Pendiente de revertir:** se movieron 50 FITS de `raw/download_paralell/` a
`raw/review_50/` para poder hidratarlos de una. Para deshacerlo:

```bash
mv ~/Dropbox/MassiveStarVariability/raw/review_50/*.fits \
   ~/Dropbox/MassiveStarVariability/raw/download_paralell/ && \
rmdir ~/Dropbox/MassiveStarVariability/raw/review_50
```

---

## 4. Plan para la sesión nueva — ejecutado 2026-09-03

### Tarea 1 — `open_LC` como lectura única  ✅

`build_cleaning_review.py` leía los FITS con `astropy.io.fits` y sin máscara de
calidad. Ahora todo pasa por `msv.io.read_lc_fits`
(`quality_bitmask="hardest"`, `pdcsap_flux`, `remove_nans`).

Impacto medido de nuevo sobre las 50 de `raw/review_50/`, idéntico al de ayer:
**298 puntos (0.070%)**, concentrados en **TIC 331664814 s11 (21.22%)**, luego
TIC 466283988 s10 (3.88%); el resto por debajo del 0.2%.

### Tarea 2 — `open_LC` movida a `src/msv/`  ✅

Nuevo módulo `src/msv/io.py`:

- `read_lc_fits(path)` — la lectura canónica, devuelve `Time/flux/flux_err`.
- `open_lc(tic, sector, path_download=config.TESS_DOWNLOAD_DIR)` — por par.
- `fits_path(tic, sector, dir)` y `parse_fits_name(path)` — el nombre
  `hlsp_tess-spoc_..._lc.fits` en un solo sitio.

Las dos celdas que la definían (celda 66 del notebook 1, celda 68 del de OGLE)
son ahora un `from msv.io import open_lc`, y sus celdas de ray llaman `open_lc`.

### Tarea 3 — duplicación resuelta  ✅

Decisión: **el notebook se queda como "una estrella, todos los pasos" y los
scripts como "muchas estrellas en un PDF"**, pero ambos sobre las mismas
funciones. Lo que se movió a la librería:

- `viz.plot_cleaning_curve(time, flux, ...)` — el panel de limpieza sobre
  arrays (rojo lo que quita el sigma-clip, líneas de gap). `viz.plot_cleaning`
  —la versión por (TIC, sector)— delega en él, así que el notebook y el PDF
  dibujan lo mismo.
- `viz.plot_phase_fold(..., phase_bins=N)` — la mediana por bin de fase que
  tenía suelta `build_phasefold_review.py`.
- `scripts/_preview.py` — cerrar Preview antes de abrir el PDF. Estaba en dos
  de los tres scripts; `build_peak_review_pdfs.py` no lo hacía y por eso
  mostraba la versión vieja.

Los tres scripts quedan solo con su parte propia: elegir la muestra, paginar y
titular.

### Tarea 4 — validación contra verdad conocida  ⚠️ escrita, bloqueada por datos

`scripts/validate_peak_selection.py`, dos sets y dos números:

- `--set fp` — los 24 `config.FP_PAIRS` (OGLE). Sin período real: cuenta cuántos
  pares emiten algún candidato. Objetivo 0.
- `--set vsx` — el cruce masivas×VSX con `Period < 13 d` (`review.load_vsx_xmatch`,
  label=1): **1427 pares (TIC, sector) de 528 TICs**. Reporta recovery top-N
  exacto+armónico con `review.HARMONICS` y `config.TOL`.

Lee siempre con `msv.io`, limpia con `clean_lightcurve` y calcula LS/ACF **en
vivo** — los parquets y los cubos vienen de la selección vieja (§5.5).

Bloqueo: los FITS de `ogle_download/` (los 24 FP incluidos) y los de
`download_paralell/` son placeholders de 0 bytes. Confirmado otra vez que
leerlos por terminal NO los hidrata. Único set medible hoy son los 9 pares de
`raw/review_50/` que caen en VSX label=1: **recovery top-3 = 6/9**, sin valor
estadístico, solo prueba de que la cadena corre.

Para desbloquear hay que hidratar desde Finder los 24 FP + una submuestra de
VSX (~100 pares basta), o volver a bajarlos de MAST a un directorio local.

## 5. Cosas encontradas que quedaron abiertas

1. **`clean_ramps_binned` borraba señal — RESUELTO 2026-09-03: eliminada.**
   TIC 179639066 s28 perdía el **49.7%** de la curva y en **TIC 339568213 s12**
   el criterio se comía un eclipse completo (un bin con un eclipse tiene std
   alta y se marca como rampa). La limpieza de producción es ahora **solo
   `sigma_clip_gap_edges`**: `clean_lightcurve` ya no tiene la etapa de rampas,
   se borraron la función, su plot diagnóstico y las celdas DEV/TEST que la
   usaban (notebooks 0 y 2-OGLE). Está en el historial de git.
   **Pendiente:** todo lo procesado con rampas (parquets de periodogramas,
   cubos, CSV de clasificación) quedó con una limpieza distinta a la actual.
2. **Eclipsante de libro clasificada `Rndm`.** TIC 12675729 s82 con
   `P₀ = 1.4405 d` dobla en un eclipse limpísimo de 25000 counts y el BRF da
   `Rndm` 0.222 con instability 0.60. Sospecha: el `hist2d` la degrada porque el
   eclipse ocupa pocos bins en el eje de flujo (el rango lo domina la profundidad).
3. **Eje del PDF de revisión.** El lineal en lag es netamente mejor que el log
   para verificar peines armónicos (el log los apelmaza a la derecha; el de
   frecuencia es peor todavía, apiña como `1/n`). Propuesto y no implementado:
   eje lineal por defecto + barra de resolución `±P²/T` sobre cada pico, y
   reportar el período con esa incerteza en `candidate_periods` — hoy se emite
   `4.190 d` cuando la resolución ahí es `±0.6 d`.
4. **Constantes hardcodeadas del LS.** `prominence_frac=0.01` está en la firma de
   `select_peaks_ls` en vez de `config.py`, y el `max(FP, 0.01)` de
   `ls_periodogram` gobierna el umbral en 44/50 estrellas sin estar documentado.
5. **Los cubos `raw/cubos/*.npy`** (ACF 2.49 GB ≈ 607k picos, LS 1.38 GB ≈ 337k,
   en `log` / `min_max` / `min_max_undersampling`) se calcularon con la selección
   de picos **vieja**. No sirven para validar el criterio nuevo.

---

## 6. Clasificación — cómo correrla

Tres pasos, porque ningún env tiene TF y sklearn 1.0.2 a la vez.

```bash
# 1. candidatos + cubo (CNN_TESS)
PYTHONPATH=src python  → candidate_periods + phase_fold_hist2d_log + amplitude_of
                          sobre flujo YA LIMPIO, guardar .npz

# 2. CNN, 20 pasadas con dropout activo (tf_env)
MSV_WEIGHTS=~/ViT_VariableStars/pretrained/keras_checkpoints \
  python step_cnn.py in.npz out.npz Number_DST     # → p_mc (20, N, 8)

# 3. BRF sobre cada pasada (CNN_TESS)
MSV_BRF=~/Dropbox/.../balanced_random_forest_model.joblib \
  PYTHONPATH=src python step_brf.py in.npz mc.npz clasificacion.csv
```

`MC_ITER = 20`. El BRF corre sobre **cada pasada MC por separado** (input fijo de
8 clases + `per` + `amplitud`); la clase sale de la **mediana** de las 20, con
`sigma` (desviación de la probabilidad ganadora) e `instability` (fracción de
pasadas donde cambia la clase ganadora).

**Agrupamiento Pulsating:** `PULSATING = ["M", "CEP", "RR", "DST"]`. Se aplica
sumando columnas de la **salida** del BRF, nunca de la entrada — `feature_names_in_`
es fijo con las 8 clases. Sumar antes del argmax importa: una pasada con
`M=0.30, DST=0.25, RR=0.10` contra `ELL=0.32` da Pulsating 0.65 agrupando, y se
perdería tomando el argmax primero.

`amplitude_of` es `|−2.5·log10(fmax/fmin)|`, o sea magnitudes — es la feature
`amplitud` del BRF y hay que calcularla sobre la curva **limpia**.

---

## 7. Sesión 2026-09-03 — dos bugs de raíz y las decisiones que salieron

Se corrió el pipeline completo sobre las 50 de `raw/review_50/` (curvas →
LS+ACF → picos → CNN → BRF → cascada) y eso destapó dos bugs, ninguno de los
cuales estaba en la lista de §5.

### Bug 1 — los períodos del ACF salían sistemáticamente cortos

`statsmodels.acf` indexa por MUESTRA y `periodograms.py` convertía con
`lag_days = lag_idx * mediana(diff(t))`. Con cadencias faltantes el índice `k`
abarca más tiempo real que `k * cadencia`, así que **todo el peine sale
comprimido**, en una fracción del orden de las cadencias perdidas de cada curva.

En TIC 12675729 s82 (3 gaps > 0.1 d, 2.6% de cadencias perdidas) el fundamental
implicado por el peine era 1.3926 d contra 1.4405 d real: **−3.3%**. Sobre una
grilla temporal uniforme el mismo ACF da 1.4398 (−0.05%) y su segundo diente
cae en **2.8819 d**, que es el período orbital verdadero medido de las
efemérides de los 18 eclipses.

Arreglo: `config.ACF_FILL_GAPS = "noiselevel"`, ahora el default de
`acf_periodogram`. Las otras opciones de astrobase no sirven: `'nan'` deja el
ACF sin picos y `'last_value'` revienta con `UnboundLocalError: gapfiller`.

El LS nunca tuvo este problema — trabaja con los tiempos reales.

### Bug 2 — el `hist2d` se normalizaba distinto a como se entrenó

`features.py` solo implementaba `log1p`, y el comentario de la celda 27 del
notebook 2-OGLE **afirma** que la CNN se entrenó así. Es falso: se entrenó con
min-max. `log1p` no es invariante al número de puntos (si `h → αh` la imagen se
satura), así que una curva TESS de 2 min con ~10 000 puntos entra como un gris
uniforme. La fracción de picos `Rndm` era 0.78 a 2 min, 0.67 a 10 min y 0.44 a
30 min: la clasificación dependía de la cadencia.

Se agregaron `phase_fold_hist2d_minmax` y el dispatcher `phase_fold_hist2d`,
con `config.HIST_NORM`.

### Decisión — `HIST_NORM = "log"` a pesar de todo

El criterio no es cuántos picos se salvan de `Rndm` sino que **la clase quede
atada al período**. Doblar una eclipsante en P/2 apila primario y secundario y
sigue pareciendo un eclipse:

| TIC 12675729 s82 | P/2 = 1.4398 | P = 2.8819 |
|---|---|---|
| `min_max` | E 0.883 | E 0.996 |
| `log` | E 0.016 (Rndm) | E 0.958 |

`min_max` no distingue (factor 1.13) y produciría un catálogo con clases
correctas y períodos a la mitad; `log` separa por factor 60. El precio es que
`log` es más conservadora y su llamada positiva tiene más dispersión.

### Consecuencia — el ensemble reemplaza al MC-dropout

Con `log`, la llamada positiva de un solo modelo con dropout es inestable: en
TIC 12675729 las pasadas MC son **bimodales** (9 de 20 sobre 0.9, 6 bajo 0.02),
y con 20 pasadas la mediana es un sorteo. Subir a 200 no arregla la
distribución, solo la mide mejor — y de hecho la σ mediana SUBE de 0.121 a
0.137, o sea 20 pasadas subestimaban la incertidumbre.

Los 7 checkpoints de `config.MODELS` son el mismo paper, el mismo test y
validation set, y difieren solo en la estrategia de balanceo. Usarlos como
ensemble determinístico:

| | MC-dropout 20 | MC-dropout 200 | Ensemble 7 |
|---|---|---|---|
| σ mediana | 0.121 | 0.137 | **0.093** |
| picos no-Rndm | 232 | 229 | **288** |
| TIC 12675729 en P | Rndm 0.73 ± 0.43 | E 0.96 ± 0.37 | **E 0.990 ± 0.012** |
| pasadas de CNN | 20 | 200 | **7** |

`scripts/step_cnn.py --ensemble`. Ojo: `sigma` e `instability` pasan a
significar desacuerdo entre modelos, no dispersión MC, así que el gate
`SIGMA_MAX = 0.12` habría que recalibrarlo.

### Qué quedó obsoleto

Todo lo calculado antes de hoy: `periodograms_acf.parquet`, `peaks.parquet`,
los cubos `raw/cubos/*.npy` y los CSV de clasificación. También la validación
de la ventana de Rayleigh anotada en `config.py` (los "24 FP_PAIRS → 0
espurios, recovery 77.6% top-3" del 2026-07-26), medida sobre períodos
sesgados.

### Scripts nuevos

```
scripts/build_lc_parquet.py       directorio de FITS -> parquet de curvas
scripts/step_cnn_export.py        peaks parquet -> npz de entrada a la CNN
scripts/step_cnn.py               npz -> p_mc (tf_env), con --ensemble
scripts/step_brf.py               npz + p_mc -> CSV con p_/s_ por clase
scripts/build_lc_prob_review.py   PDF: curva + clase por período candidato
scripts/build_phasefold_input.py  cruce con candidate_periods para el PDF de folds
scripts/validate_peak_selection.py  FP_PAIRS y VSX (bloqueado por datos)
scripts/compare_hist_norms.py     las 7 normalizaciones + sonda de invariancia
scripts/build_norm_review.py      PDF de decisión de normalización
scripts/_preview.py               cerrar Preview antes de abrir el PDF
```
