"""Configuración central: paths, clases y umbrales del pipeline.

Todos los paths absolutos del pipeline viven aquí y se pueden sobreescribir
con variables de entorno MSV_*. Ningún otro módulo debe hardcodear paths.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Datos derivados -------------------------------------------------------
LC_PARQUET_OGLE = Path(os.environ.get("MSV_LC_OGLE", REPO_ROOT / "lightcurves_all_OGLE.parquet"))
LC_PARQUET_MASSIVE = Path(os.environ.get("MSV_LC_MASSIVE", REPO_ROOT / "lightcurves_all.parquet"))
RESULTS_DIR = Path(os.environ.get("MSV_RESULTS", REPO_ROOT / "results"))
CATALOGS_DIR = REPO_ROOT / "catalogs"

# Grillas completas de periodogramas y peaks precalculados (masivas)
PERIODOGRAMS_LS = Path(os.environ.get("MSV_PGRAM_LS", REPO_ROOT / "periodograms_ls.parquet"))
PERIODOGRAMS_ACF = Path(os.environ.get("MSV_PGRAM_ACF", REPO_ROOT / "periodograms_acf.parquet"))
PEAKS_PARQUET = Path(os.environ.get("MSV_PEAKS", REPO_ROOT / "peaks.parquet"))

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
MIN_PEAK_SEP_DAYS = 0.5      # ventana temporal mínima entre peaks del ACF
ACF_PROMINENCE_FRAC = 0.05   # prominencia mínima (frac. del max) contra wiggles
                             # de ruido del ACF; validado en los 24 FP_PAIRS
                             # (0 espurios) y 68 pares label=1 (0 pérdida)

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
