"""Limpieza de saltos de telemetría de TESS.

Una sola etapa, `sigma_clip_gap_edges`: sigma-clipping global (mediana/std
calculadas EXCLUYENDO las zonas vecinas a gaps/bordes) aplicado SOLO a los
puntos dentro de esas zonas; el resto de la curva queda intacto.

La limpieza de rampas por bins (`clean_ramps_binned`) se eliminó el 2026-09-03:
descartaba señal real —un bin que contiene un eclipse tiene std alta y se
marcaba como rampa: TIC 339568213 s12 perdía un eclipse entero, TIC 179639066
s28 el 49.7% de la curva—. Está en el historial de git si hace falta.
"""
import numpy as np


def sigma_clip_gap_edges(time, flux, seg_days=0.5, gap_thresh=1.0, sigma=3.0,
                         n_edge_blocks=2, min_ref_points=20, return_mask=False):
    """Sigma-clipping global aplicado SOLO cerca de gaps/bordes.

    La mediana/std se calculan con la curva completa EXCLUYENDO las "zonas
    vecinas" (primeros/últimos `n_edge_blocks` bloques de `seg_days` días y
    los bloques adyacentes a cada gap > `gap_thresh` días) — una referencia
    global limpia. El rechazo a `sigma` desviaciones se aplica únicamente a
    los puntos DENTRO de esas zonas; el resto queda intacto.
    """
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    keep = np.ones(len(t), dtype=bool)
    if len(t) < min_ref_points:
        return (t, f, keep) if return_mask else (t, f)

    block = np.floor((t - t.min()) / seg_days).astype(int)
    n_blocks = block.max() + 1

    gap_pos = np.where(np.diff(t) > gap_thresh)[0]
    near_blocks = set(range(n_edge_blocks)) | set(range(n_blocks - n_edge_blocks, n_blocks))
    for g in gap_pos:
        near_blocks.add(int(block[g]))       # último punto antes del gap
        near_blocks.add(int(block[g + 1]))   # primer punto después del gap
    in_near_zone = np.isin(block, list(near_blocks))

    ref = f[~in_near_zone]
    if len(ref) < min_ref_points:            # curva demasiado corta: no tocar
        return (t, f, keep) if return_mask else (t, f)

    med, std = np.median(ref), np.std(ref)
    drop = (np.abs(f - med) > sigma * std) & in_near_zone
    keep = ~drop
    return (t, f, keep) if return_mask else (t[keep], f[keep])


def clean_lightcurve(time, flux, err=None,
                     seg_days=0.5, gap_thresh=1.0, sigma=3.0,
                     return_mask=False):
    """Entry point de producción: sigma-clip en gaps/bordes.

    Ordena por tiempo, descarta no-finitos y aplica `sigma_clip_gap_edges`.
    Devuelve (time, flux) o (time, flux, err), o (time, flux, err, keep) sobre
    la curva ordenada si `return_mask=True`.
    """
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    e = np.asarray(err, float) if err is not None else None

    m = np.isfinite(t) & np.isfinite(f)
    if e is not None:
        m &= np.isfinite(e)
    order = np.argsort(t[m])
    t, f = t[m][order], f[m][order]
    if e is not None:
        e = e[m][order]

    _, _, keep = sigma_clip_gap_edges(t, f, seg_days=seg_days,
                                      gap_thresh=gap_thresh, sigma=sigma,
                                      return_mask=True)

    if return_mask:
        return t, f, e, keep
    if e is not None:
        return t[keep], f[keep], e[keep]
    return t[keep], f[keep]
