"""Periodogramas LS (Lomb-Scargle) y ACF con la misma grilla de períodos.

Grilla compartida:
  cadence  = mediana(diff(time))            # robusta a gaps de TESS
  per_min  = 2 * cadence                    # período de Nyquist
  per_max  = baseline / 2                   # al menos 2 ciclos
"""
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from statsmodels.tsa.stattools import acf as _sm_acf

from .config import FAP_ALPHA


def ls_periodogram(time, flux, err, step=None, oversample=5,
                   fap_alpha=FAP_ALPHA, window=True):
    """Lomb-Scargle periodogram (astropy), FAP analítica de Baluev.

    El paso de frecuencia por defecto es 1/(oversample*baseline): ~oversample
    puntos por resolución natural de Rayleigh; muestrear más fino solo
    sobre-resuelve los mismos picos.

    Con `window=True` agrega la columna `window`: el LS de flux=1 en las
    mismas frecuencias (respuesta de la ventana de observación), usado como
    piso local en la selección de picos (power - window > FAP).

    Devuelve DataFrame con columnas `per`, `power` (+ `window`) y attrs:
    fap_level, cadence_days, baseline_days, per_min, per_max, step_cpd.
    """
    x = np.ascontiguousarray(time, dtype=np.float64)
    y = np.ascontiguousarray(flux, dtype=np.float64)
    yerr = np.ascontiguousarray(err, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    x, y, yerr = x[mask], y[mask], yerr[mask]

    mu = np.mean(y)
    y = (y / mu - 1) * 1e3     # ppt
    yerr = yerr * 1e3 / mu

    baseline = x.max() - x.min()
    cadence = float(np.median(np.diff(x)))
    per_min = 2.0 * cadence
    per_max = baseline / 2.0
    min_freq = 1.0 / per_max
    max_freq = 1.0 / per_min
    if step is None:
        step = 1.0 / (oversample * baseline)
    freq = np.arange(min_freq, max_freq + step, step)

    LS = LombScargle(x, y, yerr, normalization="standard")
    power = LS.power(freq)
    FP = LS.false_alarm_level(
        fap_alpha,
        minimum_frequency=freq[0],
        maximum_frequency=freq[-1],
        method="baluev",
    )
    level = max(float(FP), 0.01)

    df_per = pd.DataFrame({"per": 1.0 / freq, "power": power})
    if window:
        df_per["window"] = LombScargle(
            x, np.ones_like(x), center_data=False, fit_mean=False,
            normalization="standard",
        ).power(freq)
    df_per.attrs["fap_level"] = level
    df_per.attrs["fap_alpha"] = fap_alpha
    df_per.attrs["fap_method"] = "baluev"
    df_per.attrs["cadence_days"] = cadence
    df_per.attrs["baseline_days"] = baseline
    df_per.attrs["per_min"] = per_min
    df_per.attrs["per_max"] = per_max
    df_per.attrs["step_cpd"] = step
    df_per.attrs["oversample"] = oversample
    return df_per


def acf_periodogram(time, flux, err, maxlag_days=None, fap_alpha=FAP_ALPHA,
                    fill_gaps=False):
    """ACF periodogram: statsmodels.tsa.stattools.acf con confint de Bartlett.

    Pipeline:
      1. Normaliza flux a ppt: y -> (y/mean - 1) * 1e3.
      2. Con `fill_gaps` (str de astrobase, p.ej. 'noiselevel', o False):
         astrobase.autocorr_magseries se usa SOLO por sus utilidades de
         binning a cadencia uniforme y relleno del gap orbital de TESS;
         descartamos su ACF y nos quedamos con la serie regularizada.
         Sin fill_gaps la ACF corre directo sobre la serie (asume cadencia
         ~uniforme; los gaps no se rellenan).
      3. statsmodels acf(bartlett_confint=True): FAP dependiente del lag
         según Bartlett 1946 extendido:
            Var(r_k) ~ (1/N) * [1 + 2 * sum_{j=1..k-1} r_j^2]
         Detalle: una señal real a P infla r_P^2 -> infla el threshold a
         lags k > P -> puede esconder sus propios armónicos.
      4. Filtra lags a [3*cadence, maxlag_days].

    Devuelve DataFrame `per, power, fap` con `per = lag * cadence` (grilla
    UNIFORME en período: esto hace bien definida la conversión de una ventana
    temporal a `distance` de find_peaks). attrs incluye cadence_days.
    """
    x = np.ascontiguousarray(time, dtype=np.float64)
    y = np.ascontiguousarray(flux, dtype=np.float64)
    e = np.ascontiguousarray(err, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    x, y, e = x[mask], y[mask], e[mask]

    mu = np.mean(y)
    y_ppt = (y / mu - 1) * 1e3
    e_ppt = e * 1e3 / mu

    cadence = float(np.median(np.diff(x)))
    baseline = x.max() - x.min()
    if maxlag_days is None:
        maxlag_days = baseline / 2.0
    maxlags = int(np.ceil(maxlag_days / cadence))

    if fill_gaps:
        from astrobase.varbase.autocorr import autocorr_magseries
        res = autocorr_magseries(
            x, y_ppt, e_ppt,
            maxlags=maxlags,
            fillgaps=fill_gaps,
            forcetimebin=cadence,
            sigclip=None,
            magsarefluxes=True,
            verbose=False,
        )
        y_uniform = np.asarray(res["imags"], dtype=float)
        cadence_used = float(res.get("cadence", cadence))
    else:
        y_uniform = y_ppt
        cadence_used = cadence

    n_eff = len(y_uniform)
    nlags = min(maxlags, n_eff - 1)
    acf_vals, confint = _sm_acf(
        y_uniform,
        nlags=nlags,
        alpha=fap_alpha,
        bartlett_confint=True,
        fft=True,
    )
    fap_arr = confint[:, 1] - acf_vals   # threshold por lag (half-width z*se_k)

    lag_idx = np.arange(len(acf_vals))
    lag_days = lag_idx * cadence_used

    keep = (lag_days >= 3 * cadence_used) & (lag_days <= maxlag_days)

    df = pd.DataFrame({
        "per": lag_days[keep],
        "power": acf_vals[keep],
        "fap": fap_arr[keep],
    })
    df.attrs["fap_alpha"] = fap_alpha
    df.attrs["fap_method"] = "bartlett-statsmodels"
    df.attrs["cadence_days"] = cadence_used
    df.attrs["baseline_days"] = baseline
    df.attrs["per_min"] = 2.0 * cadence_used
    df.attrs["per_max"] = maxlag_days
    df.attrs["low_cut"] = 3 * cadence_used
    df.attrs["n_eff"] = n_eff
    df.attrs["n_search"] = int(keep.sum())
    df.attrs["fill_gaps"] = str(fill_gaps)
    return df
