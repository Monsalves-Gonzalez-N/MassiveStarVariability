import pickle
import numpy as np

with open("results/phasefold_curves.pkl", "rb") as handle:
    curves = pickle.load(handle)
time, flux = curves[(12921082, 82)]
flux = flux / np.median(flux)

for period, tag in [(1.7703, "P (ELL, LS)"), (0.8866, "P/2 (Pulsating, ACF)")]:
    phase = (time / period) % 1.0
    nbins = 40
    which = np.clip((phase * nbins).astype(int), 0, nbins - 1)
    binned = np.array([np.median(flux[which == b]) for b in range(nbins)])
    counts = np.array([(which == b).sum() for b in range(nbins)])
    error = np.array([np.std(flux[which == b]) for b in range(nbins)]) / np.sqrt(counts)
    binned = np.roll(binned, -binned.argmin())
    error = np.roll(error, -binned.argmin() * 0)   # ya rolado arriba
    print(f"\n=== {tag}: P = {period} d ===")
    print("fase  flujo      err")
    for b in range(nbins):
        bar = "#" * int((binned[b] - binned.min()) / (binned.max() - binned.min()) * 50)
        print(f"{b / nbins:.3f} {binned[b]:.6f} {error[b]:.6f} {bar}")

# --- los dos minimos y los dos maximos, buscados en ventanas correctas ----
period = 1.7703
phase = (time / period) % 1.0
nbins = 40
which = np.clip((phase * nbins).astype(int), 0, nbins - 1)
binned = np.array([np.median(flux[which == b]) for b in range(nbins)])
counts = np.array([(which == b).sum() for b in range(nbins)])
error = np.array([np.std(flux[which == b]) for b in range(nbins)]) / np.sqrt(counts)
shift = -binned.argmin()
binned, error = np.roll(binned, shift), np.roll(error, shift)

def extremum(center, width, kind):
    window = [(center + offset) % nbins
              for offset in range(-width, width + 1)]
    values = binned[window]
    index = window[values.argmin() if kind == "min" else values.argmax()]
    return index, binned[index], error[index]

pairs = {"min A (fase 0.0)": extremum(0, 4, "min"),
         "min B (fase 0.5)": extremum(nbins // 2, 4, "min"),
         "max A (fase 0.25)": extremum(nbins // 4, 4, "max"),
         "max B (fase 0.75)": extremum(3 * nbins // 4, 4, "max")}
for name, (index, value, err) in pairs.items():
    print(f"{name:20s} fase {index / nbins:.3f}  {value:.6f} +/- {err:.6f}")

(_, va, ea), (_, vb, eb) = pairs["min A (fase 0.0)"], pairs["min B (fase 0.5)"]
depth = binned.max() - binned.min()
print(f"\nminimos: diferencia {(vb - va) * 1e3:+.3f} ppt  "
      f"({abs(vb - va) / np.hypot(ea, eb):.1f} sigma)  "
      f"= {abs(vb - va) / depth * 100:.2f}% de la amplitud")
(_, va, ea), (_, vb, eb) = pairs["max A (fase 0.25)"], pairs["max B (fase 0.75)"]
print(f"maximos: diferencia {(vb - va) * 1e3:+.3f} ppt  "
      f"({abs(vb - va) / np.hypot(ea, eb):.1f} sigma)  "
      f"= {abs(vb - va) / depth * 100:.2f}% de la amplitud")
print(f"amplitud pico a pico: {depth * 1e3:.2f} ppt")
