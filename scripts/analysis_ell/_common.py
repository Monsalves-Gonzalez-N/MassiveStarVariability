"""Shared loading and metrics for the ELL-contamination analysis.

Human labels live at star level, keyed by (TIC, sector), and are broadcast to
every peak of that star. A peak of a periodic star is therefore NOT itself a
confirmed period: `has_period` at peak level means "belongs to a star for
which the human found some coherent period".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from msv.classify_brf import brf_mc_probs, group_probs, load_brf
from msv.config import REPO_ROOT

# Stars whose note "n" could not be parsed as "ninguno" vs "the panel n".
AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
RESULTS = REPO_ROOT / "results"
NORMALIZATIONS = ["log", "min_max", "power_0.5", "power_0.33", "column",
                  "quantized", "poisson", "rank", "binary"]


def load_peaks():
    data = np.load(RESULTS / "cnn_input.npz", allow_pickle=True)
    peaks = pd.DataFrame({
        "TIC": data["TIC"].astype(int),
        "sector": data["sector"].astype(int),
        "period": data["per"].astype(float),
        "amplitude": data["amplitude"].astype(float),
        "power": data["power"].astype(float),
        "prominence": data["prominence"].astype(float),
        "width": data["width"].astype(float),
        "source": data["source"],
    })
    peaks["key"] = [f"{tic}_{sector}"
                    for tic, sector in zip(peaks.TIC, peaks.sector)]
    return peaks


def load_truth(peaks):
    """Star-level human truth plus the two peak-level boolean masks."""
    notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
    truth = {}
    for _, row in notes.iterrows():
        if int(row.TIC) in AMBIGUOUS:
            continue
        chosen = row.per_elegido
        truth[f"{int(row.TIC)}_{int(row.sector)}"] = bool(
            isinstance(chosen, str) or np.isfinite(chosen))
    star_truth = pd.Series(truth)
    has_period = np.array([truth.get(key, None) is True for key in peaks.key])
    no_period = np.array([truth.get(key, None) is False for key in peaks.key])
    return star_truth, has_period, no_period


def cnn_grouped(probabilities_per_pass):
    """(n_pass, N, 8) CNN softmax -> (n_pass, N, 5) grouped, plus names."""
    return group_probs(np.asarray(probabilities_per_pass, dtype=float))


def brf_grouped(probabilities_per_pass, peaks, brf):
    """CNN passes -> BRF passes, grouped. NaN where amplitude/period invalid."""
    per_pass, _ = brf_mc_probs(probabilities_per_pass, peaks.period.values,
                               peaks.amplitude.values, brf)
    return group_probs(per_pass)


def argmax_class(mean_probabilities, names):
    filled = np.where(np.isnan(mean_probabilities), -np.inf, mean_probabilities)
    return np.array([names[i] for i in filled.argmax(1)])


def auc(score, positive, negative):
    """Rank AUC; ties count half. NaN scores are pushed to the bottom."""
    # NaN goes to the bottom of the ranking, but -inf minus -inf is NaN again,
    # so the comparison is done on ranks instead of on differences.
    finite = np.isfinite(score)
    ranked = np.empty(len(score), dtype=float)
    ranked[finite] = pd.Series(score[finite]).rank().values
    ranked[~finite] = 0.0
    a, b = ranked[positive], ranked[negative]
    if not len(a) or not len(b):
        return np.nan
    difference = np.subtract.outer(a, b)
    return float((difference > 0).mean() + 0.5 * (difference == 0).mean())


def load_passes(name):
    """Per-pass CNN probabilities for one normalization (7-checkpoint ensemble)."""
    if name == "log":
        path = RESULTS / "norm_compare" / "cnn_mc_log.npz"
    else:
        path = RESULTS / "norm_compare" / f"cnn_mc_{name}.npz"
    return np.load(path)["p_mc"].astype(float)


def load_peak_truth(peaks, tolerance=0.05):
    """Peak-level truth, not star-level.

    A star-level label marks every peak of a periodic star as "real", so a
    peak sitting on P/2 of an eclipsing binary counts as a correct detection.
    That is the very alias the pipeline is supposed to reject, so the match is
    made against the periods the human actually chose:

      matched  the peak reproduces one of the chosen periods within `tolerance`
               and carries that period's human class
      spurious the peak belongs to a labelled star and matches no chosen period
      unknown  the star was never reviewed, or is one of the AMBIGUOUS six
    """
    notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
    chosen = {}
    reviewed = set()
    for _, row in notes.iterrows():
        if int(row.TIC) in AMBIGUOUS:
            continue
        key = f"{int(row.TIC)}_{int(row.sector)}"
        reviewed.add(key)
        if not isinstance(row.per_elegido, str) and not np.isfinite(row.per_elegido):
            continue
        periods = str(row.per_elegido).split(",")
        classes = str(row.clase_elegida).split(",")
        chosen[key] = [(float(period), klass.strip())
                       for period, klass in zip(periods, classes)]

    status = np.array(["unknown"] * len(peaks), dtype=object)
    human_class = np.array([""] * len(peaks), dtype=object)
    for position, (key, period) in enumerate(zip(peaks.key, peaks.period)):
        if key not in reviewed:
            continue
        status[position] = "spurious"
        for reference_period, klass in chosen.get(key, []):
            if abs(period - reference_period) <= tolerance * reference_period:
                status[position] = "matched"
                human_class[position] = klass
                break
    return status, human_class
