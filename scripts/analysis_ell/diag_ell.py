import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf, brf_features, vote_fractions
from msv.config import CLASS_NAMES

data = np.load("results/cnn_input.npz", allow_pickle=True)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
p_mc = np.load("results/cnn_mc200.npz")["p_mc"]          # (200, 857, 8)
brf = load_brf()
n_iter, N, _ = p_mc.shape

# ---------- 1. CNN alone: mean vs median -------------------------------
gc, gnames = group_probs(p_mc)                            # (n_iter,N,G)
print("grupos:", gnames)
for label, est in [("mean", gc.mean(0)), ("median", np.median(gc, 0))]:
    w = est.argmax(1)
    counts = pd.Series([gnames[i] for i in w]).value_counts()
    print(f"CNN-only {label:6s} sum={est.sum(1).mean():.3f} min_sum={est.sum(1).min():.3f} | "
          + " ".join(f"{k}={v}" for k, v in counts.items()))

# ---------- 2. BRF per-pass then mean vs BRF(mean CNN) ----------------
pp, valid = brf_mc_probs(p_mc, per, amp, brf)
gb, _ = group_probs(pp)
mean_pp = np.nanmean(gb, 0); med_pp = np.nanmedian(gb, 0)
for label, est in [("mean", mean_pp), ("median", med_pp)]:
    w = np.where(np.isnan(est), -np.inf, est).argmax(1)
    counts = pd.Series([gnames[i] for i in w]).value_counts()
    print(f"BRF/pass {label:6s} sum={np.nanmean(est.sum(1)):.3f} min_sum={np.nanmin(est.sum(1)):.3f} | "
          + " ".join(f"{k}={v}" for k, v in counts.items()))

cnn_mean = p_mc.mean(0)
feat = brf_features(cnn_mean, per, amp, brf)
ok = np.isfinite(per) & np.isfinite(amp)
pp_single = np.full((N, 8), np.nan)
pp_single[ok] = brf.predict_proba(feat[ok])
gs, _ = group_probs(pp_single)
w = np.where(np.isnan(gs), -np.inf, gs).argmax(1)
counts = pd.Series([gnames[i] for i in w]).value_counts()
print("BRF(mean CNN)        | " + " ".join(f"{k}={v}" for k, v in counts.items()))

# agreement between the two orderings
wa = np.where(np.isnan(mean_pp), -np.inf, mean_pp).argmax(1)
print(f"acuerdo mean(BRF) vs BRF(mean): {(wa == w).mean():.3f}")

# ---------- 3. asimetria por clase -----------------------------------
print("\nclase      media  mediana   sigma   p16    p84   (semiancho lo/hi)")
for i, name in enumerate(gnames):
    v = gb[:, :, i]
    m, md, sd = np.nanmean(v), np.nanmedian(v), np.nanstd(v)
    lo, hi = np.nanpercentile(v, [16, 84])
    print(f"{name:10s} {m:.3f}  {md:.3f}   {sd:.3f}  {lo:.3f} {hi:.3f}   {md-lo:.3f}/{hi-md:.3f}")

# ---------- 4. cuanto ELL agrega el BRF -------------------------------
cnn_w = gc.mean(0).argmax(1)
brf_w = wa
iell = gnames.index("ELL")
print(f"\nELL CNN-only: {(cnn_w==iell).sum()}   ELL tras BRF: {(brf_w==iell).sum()}")
flow = pd.crosstab(pd.Series([gnames[i] for i in cnn_w], name="CNN"),
                   pd.Series([gnames[i] for i in brf_w], name="BRF"))
print(flow.to_string())

# ---------- 5. regimen constante del BRF ------------------------------
print("\nBRF con probs CNN FIJAS (pico medio), barriendo amplitud:")
base = cnn_mean.mean(0)[None, :].repeat(6, 0)
amps = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
pers = np.full(6, np.median(per[np.isfinite(per)]))
out = brf.predict_proba(brf_features(base, pers, amps, brf))
print(pd.DataFrame(out, columns=CLASS_NAMES, index=[f"amp={a}" for a in amps]).round(3).to_string())
print("\nmisma cosa barriendo per (amp=mediana):")
pers2 = np.array([0.1, 0.5, 1.0, 3.0, 10.0, 30.0])
amps2 = np.full(6, np.nanmedian(amp))
out2 = brf.predict_proba(brf_features(base, pers2, amps2, brf))
print(pd.DataFrame(out2, columns=CLASS_NAMES, index=[f"per={p}" for p in pers2]).round(3).to_string())

print(f"\nrango observado: amp p1/p50/p99 = "
      f"{np.nanpercentile(amp,[1,50,99]).round(4)}  per = {np.nanpercentile(per[np.isfinite(per)],[1,50,99]).round(3)}")
