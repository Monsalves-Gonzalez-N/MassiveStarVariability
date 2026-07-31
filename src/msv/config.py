"""Configuración central: paths, clases y umbrales del pipeline.

Todos los paths absolutos del pipeline viven aquí y se pueden sobreescribir
con variables de entorno MSV_*. Ningún otro módulo debe hardcodear paths.

Los datos pesados (36 GB) NO están en el repo: viven en Dropbox para poder
trabajar desde varias máquinas. `DATA_DIR` los localiza, en este orden:

  1. $MSV_DATA_DIR                     — override explícito (útil en cluster)
  2. el primer candidato de _DATA_DIR_CANDIDATES que exista
  3. REPO_ROOT                         — layout antiguo (todo suelto en el repo)

En una máquina nueva: clonar el repo, sincronizar Dropbox y correr
`python scripts/link_data.py` para dejar los symlinks en su sitio.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Ubicaciones conocidas de la carpeta de datos, en orden de preferencia.
# Añadir aquí la ruta de cada máquina nueva (Linux, macOS, etc.).
_DATA_DIR_CANDIDATES = [
    Path.home() / "Dropbox" / "MassiveStarVariability",
    Path.home() / "Dropbox (Personal)" / "MassiveStarVariability",
]


def _resolve_data_dir():
    env = os.environ.get("MSV_DATA_DIR")
    if env:
        return Path(env).expanduser()
    for cand in _DATA_DIR_CANDIDATES:
        if cand.is_dir():
            return cand
    return REPO_ROOT


DATA_DIR = _resolve_data_dir()
RAW_DIR = DATA_DIR / "raw" if DATA_DIR != REPO_ROOT else REPO_ROOT
DERIVED_DIR = DATA_DIR / "derived" if DATA_DIR != REPO_ROOT else REPO_ROOT

# --- Datos derivados -------------------------------------------------------
LC_PARQUET_OGLE = Path(os.environ.get("MSV_LC_OGLE", DERIVED_DIR / "lightcurves_all_OGLE.parquet"))
LC_PARQUET_MASSIVE = Path(os.environ.get("MSV_LC_MASSIVE", DERIVED_DIR / "lightcurves_all.parquet"))
TRAIN_NUMBER_M = Path(os.environ.get("MSV_TRAIN_M", DERIVED_DIR / "train_number_M.csv"))
RESULTS_DIR = Path(os.environ.get("MSV_RESULTS", REPO_ROOT / "results"))
CATALOGS_DIR = REPO_ROOT / "catalogs"

# Grillas completas de periodogramas y peaks precalculados (masivas)
PERIODOGRAMS_LS = Path(os.environ.get("MSV_PGRAM_LS", DERIVED_DIR / "periodograms_ls.parquet"))
PERIODOGRAMS_ACF = Path(os.environ.get("MSV_PGRAM_ACF", DERIVED_DIR / "periodograms_acf.parquet"))
PEAKS_PARQUET = Path(os.environ.get("MSV_PEAKS", DERIVED_DIR / "peaks.parquet"))

# --- Datos crudos (descargables de MAST/OGLE si se pierden) -----------------
CUBES_DIR = RAW_DIR / "cubos"
TESS_DOWNLOAD_DIR = RAW_DIR / "download_paralell"
OGLE_DOWNLOAD_DIR = RAW_DIR / "ogle_download"

# --- Modelos externos (repo Paper_OGLE) -------------------------------------
WEIGHTS_DIR = Path(os.environ.get("MSV_WEIGHTS", "/home/nicolas/nico/git/Paper_OGLE/Weights"))
BRF_MODEL = Path(os.environ.get("MSV_BRF", "/home/nicolas/nico/git/balanced_random_forest_model.joblib"))

# --- Clasificación ----------------------------------------------------------
CLASS_NAMES = ["ELL", "M", "CEP", "DST", "E", "LPV", "RR", "Rndm"]
PERIODIC = ["ELL", "M", "CEP", "DST", "E", "RR"]
MODELS = ["Number_CEP", "Number_DST", "Number_ELL", "Number_M",
          "batchBalanced_Number_DST", "batchBalanced_Number_ELL", "batchBalanced_Number_M"]
DEFAULT_MODEL = "Number_DST"

# Gate de incertidumbre (MC-dropout) y selección Path-2
PROB_MIN = 0.90
SIGMA_MAX = 0.12
TOL = 0.05          # tolerancia relativa para match de período vs OGLE

# --- Periodogramas ----------------------------------------------------------
FAP_ALPHA = 0.0027
MIN_PEAK_SEP_DAYS = None     # ventana temporal FIJA entre peaks del ACF [d].
                             # None → usar la ventana adaptativa de Rayleigh
                             # (ACF_RAYLEIGH_K). La fija de 0.5 d suprimía el
                             # fundamental de estrellas con per < 0.5 d y
                             # reportaba el armónico 2x.
ACF_RAYLEIGH_K = 3.0         # ventana adaptativa: k * P^2 / T (T = baseline);
                             # dos picos más cercanos que eso no son
                             # distinguibles físicamente (límite de Rayleigh)
ACF_PROMINENCE_FRAC = 0.20   # prominencia mínima (frac. del max) contra wiggles
                             # de ruido del ACF. Validado 2026-07-26 con ventana
                             # Rayleigh: 24 FP_PAIRS → 0 espurios; 67 pares
                             # label=1 → recovery 77.6% top-3 (la fija 0.5d +
                             # prom 0.05 daba 68.7%); el recovery no cambia en
                             # todo el barrido de prominencia 0.05→0.20

# Falsos positivos conocidos del corte Number_DST (la misma lista de
# FP_OGLE.pdf): se usan para revisar la limpieza de rampas/telemetría.
FP_PAIRS = [
    (11819849, 61), (11819849, 62), (14655089, 80), (15121710, 80),
    (151220588, 61), (167691127, 80), (180618091, 27), (180618091, 28),
    (182735022, 27), (188166816, 66), (208151281, 66), (267547804, 28),
    (279102962, 65), (293718765, 65), (294665029, 61), (299374140, 65),
    (30643326, 64), (30722286, 64), (31150648, 63), (315181712, 65),
    (362368622, 35), (400652459, 81), (45589186, 66), (47028374, 66),
]


def find_bad_flux_pairs(lc_parquet):
    """Pares (TIC, sector) con flux <= 0 (error de cleaning): se excluyen
    de todo el pipeline — la amplitud |−2.5·log10(fmax/fmin)| queda indefinida."""
    import pandas as pd
    fmin = pd.read_parquet(lc_parquet, columns=["TIC", "sector", "flux"]).groupby(
        ["TIC", "sector"])["flux"].min()
    return {(int(t), int(s)) for (t, s) in fmin[fmin <= 0].index}
