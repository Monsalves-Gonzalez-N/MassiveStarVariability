"""Single entry point for TESS-SPOC light curves.

Every read of a `hlsp_tess-spoc_*_lc.fits` file in the pipeline goes through
`read_lc_fits`. It is the canonical read: PDCSAP flux with
`quality_bitmask="hardest"`, NaNs dropped. Reading the FITS by hand with
astropy skips the quality mask and gives a different light curve (up to 21% of
the points on the worst star of the 2026-09-02 sample of 50).
"""
import sys

import numpy as np
import pandas as pd

from . import config

FITS_TEMPLATE = "hlsp_tess-spoc_tess_phot_{tic:016d}-s{sector:04d}_tess_v1_lc.fits"
QUALITY_BITMASK = "hardest"
FLUX_COLUMN = "pdcsap_flux"


def _to_native_endian(array):
    # lightkurve hands back big-endian columns from the FITS; pandas/pyarrow
    # refuse them on write.
    if array.dtype.byteorder == ">":
        return array.astype(array.dtype.newbyteorder("="))
    if array.dtype.byteorder == "=" and sys.byteorder != "little":
        return array.astype("<" + array.dtype.kind + str(array.dtype.itemsize))
    return array


def fits_path(tic, sector, path_download=None):
    directory = config.TESS_DOWNLOAD_DIR if path_download is None else path_download
    from pathlib import Path
    return Path(directory) / FITS_TEMPLATE.format(tic=int(tic), sector=int(sector))


def parse_fits_name(path):
    """(TIC, sector) from a `hlsp_tess-spoc_..._lc.fits` filename."""
    from pathlib import Path
    stem = Path(path).name.split("_phot_")[1].replace("_tess_v1_lc.fits", "")
    tic, sector = stem.split("-s")
    return int(tic), int(sector)


def read_lc_fits(path):
    """Canonical read of one SPOC FITS: (Time, flux, flux_err) as a DataFrame."""
    import lightkurve as lk

    light_curve = lk.read(str(path), quality_bitmask=QUALITY_BITMASK,
                          flux_column=FLUX_COLUMN).remove_nans()
    frame = pd.DataFrame({
        "Time": _to_native_endian(np.asarray(light_curve.time.value)),
        "flux": _to_native_endian(np.asarray(light_curve.flux.value)),
        "flux_err": _to_native_endian(np.asarray(light_curve.flux_err.value)),
    })
    return frame.sort_values("Time").reset_index(drop=True)


def open_lc(tic, sector, path_download=None):
    """Canonical read by (TIC, sector). Replaces the notebook `open_LC`."""
    return read_lc_fits(fits_path(tic, sector, path_download))
