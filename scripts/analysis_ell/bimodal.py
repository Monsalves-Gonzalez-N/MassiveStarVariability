import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

data = np.load("results/cnn_input.npz", allow_pickle=True)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
p200 = np.load("results/cnn_mc200.npz")["p_mc"]
p20 = np.load("results/cnn_mc20.npz")["p_mc"]
brf = load_brf()

def bimodality(grouped, gnames, label):
    """Por pico: distribucion entre pasadas de la prob de la clase ganadora."""
    mean = np.nanmean(grouped, 0)
    winner = np.where(np.isnan(mean), -np.inf, mean).argmax(1)
    rows = np.arange(grouped.shape[1])
    values = grouped[:, rows, winner]                    # (n_iter, N)

    middle = ((values > 0.2) & (values < 0.8)).mean(0)    # masa en el medio
    edges = ((values < 0.1) | (values > 0.9)).mean(0)     # masa en los extremos
    both_ends = (((values < 0.1).any(0)) & ((values > 0.9).any(0)))
    sd = np.nanstd(values, 0)
    print(f"\n--- {label} ({grouped.shape[0]} pasadas) ---")
    print(f"picos con masa en LOS DOS extremos (<0.1 y >0.9): "
          f"{both_ends.sum()} de {len(both_ends)} ({both_ends.mean():.1%})")
    print(f"fraccion de pasadas en el medio [0.2,0.8]: mediana {np.median(middle):.3f}")
    print(f"fraccion de pasadas en los extremos:       mediana {np.median(edges):.3f}")
    print(f"sigma mediana: {np.median(sd):.3f}")
    frame = pd.DataFrame({"clase": [gnames[i] for i in winner],
                          "bimodal": both_ends, "medio": middle,
                          "extremos": edges, "sigma": sd})
    print(frame.groupby("clase").agg(N=("bimodal", "size"),
                                     bimodal=("bimodal", "sum"),
                                     frac=("bimodal", "mean"),
                                     medio=("medio", "median"),
                                     sigma=("sigma", "median")).round(3).to_string())
    return both_ends

for p_mc, tag in [(p20, "20"), (p200, "200")]:
    gc, gnames = group_probs(p_mc)
    bimodality(gc, gnames, f"CNN sola, {tag}")
for p_mc, tag in [(p20, "20"), (p200, "200")]:
    pp, _ = brf_mc_probs(p_mc, per, amp, brf)
    gb, gnames = group_probs(pp)
    bimodality(gb, gnames, f"tras BRF, {tag}")
