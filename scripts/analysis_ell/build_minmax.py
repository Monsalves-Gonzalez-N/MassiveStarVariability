"""Reconstruye el cubo de entrada a la CNN con min_max, sobre los MISMOS picos."""
import numpy as np, pandas as pd
from msv.features import phase_fold_hist2d_minmax, phase_fold_hist2d_log

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float)
curves = pd.read_pickle("results/phasefold_curves.pkl")

cube_minmax = np.empty((len(per), 32, 32, 1), dtype=np.float32)
cube_log = np.empty_like(cube_minmax)
for index, (t, s, period) in enumerate(zip(tic, sector, per)):
    time, flux = curves[(int(t), int(s))]
    good = np.isfinite(time) & np.isfinite(flux)
    cube_minmax[index, :, :, 0] = phase_fold_hist2d_minmax(time[good], flux[good], period)
    cube_log[index, :, :, 0] = phase_fold_hist2d_log(time[good], flux[good], period)

# control: el log reconstruido debe coincidir con el X guardado
delta = np.abs(cube_log - data["X"]).max()
print(f"max|log reconstruido - X guardado| = {delta:.2e}  (control de consistencia)")

out = {key: data[key] for key in data.files if key != "X"}
np.savez_compressed("results/cnn_input_minmax.npz", X=cube_minmax, **out)
print("-> results/cnn_input_minmax.npz", cube_minmax.shape)
