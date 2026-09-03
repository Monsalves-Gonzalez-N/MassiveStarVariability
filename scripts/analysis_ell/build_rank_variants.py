"""rank con distintos desempates: si la senal sobrevive al azar, es real."""
import numpy as np, pandas as pd
from msv.features import _phase_fold_counts

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float)
curves = pd.read_pickle("results/phasefold_curves.pkl")

raw = np.empty((len(per), 32, 32))
for index, (t, s, period) in enumerate(zip(tic, sector, per)):
    time, flux = curves[(int(t), int(s))]
    good = np.isfinite(time) & np.isfinite(flux)
    raw[index] = _phase_fold_counts(time[good], flux[good], period)

def rank_index(counts):                      # el original: empates por indice
    flat = counts.ravel()
    order = flat.argsort().argsort().astype(float)
    return (order / max(order.max(), 1.0)).reshape(counts.shape)

def rank_random(counts, rng):                # empates al azar
    flat = counts.ravel()
    jitter = rng.permutation(flat.size)
    order = np.lexsort((jitter, flat)).argsort().astype(float)
    return (order / max(order.max(), 1.0)).reshape(counts.shape)

def rank_average(counts):                    # empates promediados (rango medio)
    flat = counts.ravel()
    return (pd.Series(flat).rank(method="average").to_numpy() / flat.size
            ).reshape(counts.shape)

rng = np.random.default_rng(20260903)
for name, function in [("rank_random1", lambda c: rank_random(c, rng)),
                       ("rank_random2", lambda c: rank_random(c, rng)),
                       ("rank_average", rank_average)]:
    cube = np.empty((len(per), 32, 32, 1), dtype=np.float32)
    for index in range(len(per)):
        cube[index, :, :, 0] = function(raw[index]).T[::-1]
    out = {key: data[key] for key in data.files if key != "X"}
    np.savez_compressed(f"results/norm_compare/cnn_input_{name}.npz", X=cube, **out)
    print(f"{name}: media {cube.mean():.4f}")
