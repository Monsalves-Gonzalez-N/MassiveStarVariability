import sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from compare_hist_norms import normalize_counts
from msv.features import _phase_fold_counts

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float)
curves = pd.read_pickle("results/phasefold_curves.pkl")

raw = np.empty((len(per), 32, 32), dtype=np.float64)
n_points = np.empty(len(per), dtype=int)
for index, (t, s, period) in enumerate(zip(tic, sector, per)):
    time, flux = curves[(int(t), int(s))]
    good = np.isfinite(time) & np.isfinite(flux)
    raw[index] = _phase_fold_counts(time[good], flux[good], period)
    n_points[index] = int(good.sum())

def rank_norm(counts):
    """Ecualizacion: reemplaza cada valor por su rango. Invariante a cualquier
    transformacion monotona de las cuentas, o sea el limite de la familia."""
    flat = counts.ravel()
    order = flat.argsort().argsort().astype(np.float64)
    return (order / max(order.max(), 1.0)).reshape(counts.shape)

def binary_norm(counts):
    """Solo ocupacion: borra la densidad, deja el contorno del fold."""
    return (counts > 0).astype(np.float64)

NORMS = ["log", "min_max", "power_0.5", "power_0.33", "column", "quantized",
         "poisson", "rank", "binary"]
for name in NORMS:
    cube = np.empty((len(per), 32, 32, 1), dtype=np.float32)
    for index in range(len(per)):
        if name == "rank":
            image = rank_norm(raw[index])
        elif name == "binary":
            image = binary_norm(raw[index])
        else:
            image = normalize_counts(raw[index], name, n_points=n_points[index])
        # msv.features orienta la imagen con .T[::-1] DESPUES de normalizar;
        # saltearlo le entrega a la CNN el histograma transpuesto.
        cube[index, :, :, 0] = image.T[::-1]
    if name == "log":
        delta = np.abs(cube - data["X"]).max()
        print(f"CONTROL log vs X guardado: max|diff| = {delta:.2e}")
    out = {key: data[key] for key in data.files if key != "X"}
    np.savez_compressed(f"results/norm_compare/cnn_input_{name}.npz", X=cube, **out)
    print(f"{name:11s} rango [{cube.min():.3f}, {cube.max():.3f}]  "
          f"media {cube.mean():.4f}")
