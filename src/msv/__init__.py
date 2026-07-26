"""msv — pipeline de variabilidad TESS para estrellas masivas.

Módulos:
  config       paths y constantes (override por env MSV_*)
  cleaning     saltos de telemetría: rampas + sigma-clip en gaps/bordes
  periodograms LS (Baluev) y ACF (Bartlett) con grilla compartida
  peaks        find_peaks sobre power crudo con ventana temporal mínima
  features     hist2d 32x32 phase-folded + amplitud (input CNN)
  classify_brf CNN MC-dropout → BRF → gate (TensorFlow lazy)
"""
from . import config  # noqa: F401
from .cleaning import clean_lightcurve, clean_ramps_binned, sigma_clip_gap_edges  # noqa: F401
from .periodograms import acf_periodogram, ls_periodogram  # noqa: F401
from .peaks import select_peaks, select_peaks_acf, select_peaks_ls  # noqa: F401
from .features import amplitude_of, build_cube, phase_fold_hist2d_log  # noqa: F401

__version__ = "0.1.0"
