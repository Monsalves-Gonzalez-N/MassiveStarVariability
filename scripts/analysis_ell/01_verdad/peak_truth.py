"""What the peak-level truth changes about the ELL contamination."""
import numpy as np
import pandas as pd

from _common import (argmax_class, brf_grouped, load_brf, load_passes,
                     load_peak_truth, load_peaks, load_truth)

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

brf_mean = np.nanmean(brf_grouped(load_passes("log"), peaks, brf)[0], axis=0)
names = ["ELL", "Pulsating", "E", "LPV", "Rndm"]
klass = argmax_class(brf_mean, names)

print("picos por estado y clase del pipeline:")
print(pd.crosstab(pd.Series(status, name="estado"),
                  pd.Series(klass, name="pipeline")).to_string())

print("\nlos picos MATCHED: clase humana contra clase del pipeline")
matched = status == "matched"
print(pd.crosstab(pd.Series(human_class[matched], name="humano"),
                  pd.Series(klass[matched], name="pipeline")).to_string())

print("\nlos picos que el pipeline llama ELL, desglosados:")
is_ell = klass == "ELL"
detail = pd.DataFrame({
    "TIC": peaks.TIC[is_ell].values, "sector": peaks.sector[is_ell].values,
    "per": peaks.period[is_ell].values.round(3),
    "power": peaks.power[is_ell].values.round(3),
    "estado": status[is_ell], "humano": human_class[is_ell],
    "estrella_periodica": has_period[is_ell],
})
print(detail[detail.estado != "spurious"].to_string(index=False))
print(f"\ntotal ELL: {int(is_ell.sum())}   "
      f"matched-y-humano-ELL: {int((is_ell & matched & (human_class == 'ELL')).sum())}   "
      f"matched-pero-otra-clase: {int((is_ell & matched & (human_class != 'ELL')).sum())}   "
      f"spurious: {int((is_ell & (status == 'spurious')).sum())}   "
      f"unknown: {int((is_ell & (status == 'unknown')).sum())}")
