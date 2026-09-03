import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf, vote_fractions

data = np.load("results/cnn_input.npz", allow_pickle=True)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
tic = data["TIC"]; sector = data["sector"]; source = data["source"]
rng = np.random.default_rng(0)
p_mc = np.load("results/cnn_mc200.npz")["p_mc"][rng.choice(200, 20, replace=False)]
brf = load_brf()

pp, _ = brf_mc_probs(p_mc, per, amp, brf)
gb, gnames = group_probs(pp)
gc, _ = group_probs(p_mc)
iell = gnames.index("ELL")
mean_brf = np.nanmean(gb, 0); mean_cnn = gc.mean(0)
votes = vote_fractions(gb)
winner = np.where(np.isnan(mean_brf), -np.inf, mean_brf).argmax(1)
ell = winner == iell

print("=== donde nace la saturacion: prob ELL de los picos ELL ===")
print("umbral   CNN sola   tras BRF")
for threshold in [0.9, 0.99, 0.999, 0.9999]:
    print(f">{threshold:<8} {(mean_cnn[ell, iell] > threshold).sum():>6d}"
          f"     {(mean_brf[ell, iell] > threshold).sum():>6d}")
print(f"max prob ELL: CNN {mean_cnn[ell, iell].max():.6f}   BRF {mean_brf[ell, iell].max():.6f}")
print(f"(el BRF tiene {brf.n_estimators} arboles -> resolucion minima {1/brf.n_estimators:.4f})")

sat = ell & (mean_brf[:, iell] > 0.99)
print(f"\n=== los {sat.sum()} picos con prob ELL > 0.99 tras el BRF ===")
second = np.array(mean_brf, copy=True); second[:, iell] = -np.inf
detail = pd.DataFrame({
    "TIC": tic[sat], "sec": sector[sat], "src": source[sat],
    "per": per[sat].round(3), "amp": amp[sat].round(4),
    "p_ELL": mean_brf[sat, iell].round(4),
    "sigma": np.nanstd(gb, 0)[sat, iell].round(4),
    "voto": votes[sat, iell].round(2),
    "2do": [gnames[i] for i in second[sat].argmax(1)],
    "p_2do": second[sat].max(1).round(4),
    "p_LPV": mean_brf[sat, gnames.index("LPV")].round(4),
    "p_Rndm": mean_brf[sat, gnames.index("Rndm")].round(4),
}).sort_values("p_ELL", ascending=False)
print(detail.to_string(index=False))

print("\n=== a los saturados los agarra alguna penalizacion? ===")
print(f"voto < 0.5 (gate 0.5): {(votes[sat, iell] < 0.5).sum()} de {sat.sum()}")
print(f"voto < 0.7 (gate 0.7): {(votes[sat, iell] < 0.7).sum()} de {sat.sum()}")
print(f"sigma mediana de los saturados: {np.nanmedian(np.nanstd(gb, 0)[sat, iell]):.4f}"
      f"   vs resto de ELL: {np.nanmedian(np.nanstd(gb, 0)[ell & ~sat, iell]):.4f}")
