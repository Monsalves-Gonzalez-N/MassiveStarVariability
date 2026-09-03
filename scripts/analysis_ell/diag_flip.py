import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf, vote_fractions

data = np.load("results/cnn_input.npz", allow_pickle=True)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
rng = np.random.default_rng(0)
p_mc = np.load("results/cnn_mc200.npz")["p_mc"][rng.choice(200, 20, replace=False)]
brf = load_brf()

pp, _ = brf_mc_probs(p_mc, per, amp, brf)
gb, gnames = group_probs(pp)
mean_pp = np.nanmean(gb, 0)
votes = vote_fractions(gb)
winner = np.where(np.isnan(mean_pp), -np.inf, mean_pp).argmax(1)
per_pass_argmax = np.where(np.isnan(gb), -np.inf, gb).argmax(2)   # (20, N)

print("Para cada clase ganadora: en las 20 pasadas, que fraccion del tiempo gana cada clase")
table = []
for index, name in enumerate(gnames):
    which = winner == index
    shares = [(per_pass_argmax[:, which] == other).mean() for other in range(len(gnames))]
    table.append([name, int(which.sum())] + [round(share, 3) for share in shares])
print(pd.DataFrame(table, columns=["ganadora", "N"] + gnames).to_string(index=False))

iell = gnames.index("ELL")
ell = winner == iell
print(f"\nDe los {ell.sum()} picos ELL, distribucion del voto ELL:")
print(pd.Series(votes[ell, iell]).describe().round(3).to_string())
print("\nvotos ELL ordenados:", np.round(np.sort(votes[ell, iell]), 2))
print(f"picos con voto ELL == 1.0 (nunca cambian): {(votes[ell, iell] == 1.0).sum()}")
print(f"picos con voto ELL < 0.5 (ELL pierde la mayoria): {(votes[ell, iell] < 0.5).sum()}")

print("\nCuando ELL NO gana una pasada, quien gana:")
sub = per_pass_argmax[:, ell]
other = sub[sub != iell]
print(pd.Series([gnames[i] for i in other]).value_counts(normalize=True).round(3).to_string())
