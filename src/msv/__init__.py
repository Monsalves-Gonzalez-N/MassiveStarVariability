"""msv — pipeline de variabilidad TESS para estrellas masivas.

Módulos:
  config       paths y constantes (override por env MSV_*)
  io           lectura canónica de los FITS SPOC (quality_bitmask="hardest")
  cleaning     saltos de telemetría: sigma-clip en gaps/bordes
  periodograms LS (Baluev) y ACF (Bartlett) con grilla compartida
  peaks        find_peaks sobre power crudo con ventana temporal mínima
  features     hist2d 32x32 phase-folded + amplitud (input CNN)
  classify_brf CNN MC-dropout → BRF → gate (TensorFlow lazy)
"""
from . import config  # noqa: F401
from .io import fits_path, open_lc, parse_fits_name, read_lc_fits  # noqa: F401
from .cleaning import clean_lightcurve, sigma_clip_gap_edges  # noqa: F401
from .periodograms import acf_periodogram, ls_periodogram  # noqa: F401
from .peaks import select_peaks, select_peaks_acf, select_peaks_ls  # noqa: F401
from .features import (amplitude_of, build_cube, phase_fold_hist2d,  # noqa: F401
                       phase_fold_hist2d_log, phase_fold_hist2d_minmax)

__version__ = "0.1.0"
