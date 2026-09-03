#!/usr/bin/env python
"""Two PDFs (one LS, one ACF) to visually review the selected peaks.

Both PDFs cover the SAME random sample of (TIC, sector) pairs, in the same
order: page N / panel M is the same star in both files, so the LS and the ACF
of a star can be compared side by side.

    python scripts/build_peak_review_pdfs.py
    python scripts/build_peak_review_pdfs.py --pages 10 --per-page 5 --seed 42
    python scripts/build_peak_review_pdfs.py --sample-from peaks
    python scripts/build_peak_review_pdfs.py --data-dir test_data   # while Dropbox syncs

Peak selection comes from msv.peaks (Rayleigh window + prominence_frac from
config), so what is drawn is exactly what the pipeline keeps.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview import open_in_preview  # noqa: E402

from msv import config
from msv.peaks import label_harmonics, select_peaks_acf, select_peaks_ls
from msv.periodograms import trials_corrected_band, white_noise_band

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)


def sample_pairs(source_path, n_pairs, seed=None):
    """Random (TIC, sector) pairs among those present in `source_path`."""
    pairs = pd.read_parquet(source_path, columns=["TIC", "sector"]).drop_duplicates()
    generator = np.random.default_rng(seed)
    n_pairs = min(n_pairs, len(pairs))
    chosen = generator.choice(len(pairs), size=n_pairs, replace=False)
    return [(int(tic), int(sector)) for tic, sector in pairs.to_numpy()[chosen]]


def load_periodograms(path, pairs):
    """One pushdown-filtered read for the whole sample instead of one per star.

    The parquets hold the full grid for every star, so filtering per pair would
    re-scan a multi-GB file `n_pairs` times.
    """
    tics = sorted({tic for tic, _ in pairs})
    sectors = {sector for _, sector in pairs}
    frame = pd.read_parquet(path, filters=[("TIC", "in", tics)])
    frame = frame[frame["sector"].isin(sectors)]
    return {key: group.sort_values("per").reset_index(drop=True)
            for key, group in frame.groupby(["TIC", "sector"], sort=False)}


def draw_ls_panel(axis, periodogram, tic, sector, annotate_top):
    """`stored_band` no aplica al LS: su FAP de Baluev ya es un escalar."""
    period = periodogram["per"].to_numpy()
    power = periodogram["power"].to_numpy()
    fap = float(periodogram["fap"].iloc[0])
    window = periodogram["window"].to_numpy() if "window" in periodogram else None
    frequency_step = float(np.median(np.abs(np.diff(1.0 / period))))

    peaks = select_peaks_ls(period, power, fap, window=window,
                            freq_step=frequency_step, top_n=None)

    if window is not None:
        axis.plot(period, window, color="0.6", lw=0.6, label="window (LS of flux=1)")
    axis.plot(period, power, "k-", lw=0.7, label="LS power")
    axis.axhline(fap, color="r", ls="--", lw=1.0, label=f"FAP Baluev = {fap:.3f}")
    annotate_peaks(axis, peaks, annotate_top)
    axis.set_ylabel("LS power", fontsize=9)
    finish_panel(axis, tic, sector, len(peaks))
    return peaks


def draw_acf_panel(axis, periodogram, tic, sector, annotate_top,
                   stored_band=False):
    period = periodogram["per"].to_numpy()
    power = periodogram["power"].to_numpy()
    fap = periodogram["fap"].to_numpy()
    cadence = float(np.median(np.diff(period)))

    band_label = "±FAP bartlett-statsmodels"
    if not stored_band:
        recovered = white_noise_band(period, power, fap)
        if np.isscalar(recovered):
            fap = np.full_like(period, recovered)
            band_label = "±FAP ruido blanco z/√N"
            if config.ACF_TRIALS_CORRECTION:
                fap = trials_corrected_band(fap, period.size)
                band_label += f" ({fap[0] / recovered * 3.0:.1f}σ, {period.size} trials)"

    peaks = label_harmonics(select_peaks_acf(period, power, fap, cadence,
                                             top_n=None))

    axis.plot(period, power, "k-", lw=0.7, label="ACF")
    axis.plot(period, fap, "r--", lw=1.0, label=band_label)
    axis.plot(period, -fap, "r--", lw=1.0)
    axis.axhline(0, color="0.85", lw=0.5)
    annotate_peaks(axis, peaks, annotate_top)
    axis.set_ylabel("ACF", fontsize=9)
    finish_panel(axis, tic, sector, len(peaks))
    return peaks


def annotate_peaks(axis, peaks, annotate_top):
    """Circle every selected peak, label only the `annotate_top` most prominent.

    Stars with dozens of peaks above the FAP turn into unreadable label soup
    if all of them are annotated.

    When `label_harmonics` has run, the harmonic series of the fundamental is
    drawn apart from the rest: the fundamental filled and labelled with its
    period, its harmonics open and labelled `n x`, and anything outside the
    series in a different colour. The fundamental is always labelled, however
    prominent it is — in a slowly decaying ACF the tallest peak is often a
    high-order harmonic, which is exactly the case worth seeing.
    """
    if peaks.empty:
        return

    has_series = "comb_order" in peaks and peaks["comb_order"].notna().any()
    if not has_series:
        axis.plot(peaks["per"], peaks["power"], "o", color="tab:orange",
                  ms=7, mfc="none", mew=1.5, label=f"{len(peaks)} peaks > FAP")
        labelled = peaks.nlargest(annotate_top, "prominence") if annotate_top else peaks
        for _, peak in labelled.iterrows():
            axis.annotate(f"{peak['per']:.3f} d", (peak["per"], peak["power"]),
                          xytext=(4, 4), textcoords="offset points", fontsize=7)
        return

    on_comb = peaks["comb_order"].notna()
    fundamental = peaks[peaks["is_fundamental"]]
    harmonics = peaks[peaks["harmonic_order"].notna() & ~peaks["is_fundamental"]]
    subharmonics = peaks[on_comb & peaks["harmonic_order"].isna()]
    unrelated = peaks[~on_comb]

    if not fundamental.empty:
        period_0 = float(fundamental["period_series"].iloc[0])
        axis.plot(fundamental["per"], fundamental["power"], "o", color="tab:orange",
                  ms=9, mew=1.5, label=f"P0 ajustado = {period_0:.3f} d")
    if not harmonics.empty:
        axis.plot(harmonics["per"], harmonics["power"], "o", color="tab:orange",
                  ms=7, mfc="none", mew=1.5,
                  label=f"{len(harmonics)} armonicos de la serie")
    if not subharmonics.empty:
        axis.plot(subharmonics["per"], subharmonics["power"], "^", color="tab:orange",
                  ms=6, mfc="none", mew=1.5,
                  label=f"{len(subharmonics)} impares del peine (P0/2)")
    if not unrelated.empty:
        axis.plot(unrelated["per"], unrelated["power"], "s", color="tab:blue",
                  ms=6, mfc="none", mew=1.5,
                  label=f"{len(unrelated)} picos fuera de la serie")

    labelled = peaks.nlargest(annotate_top, "prominence") if annotate_top else peaks
    labelled = pd.concat([fundamental, labelled]).drop_duplicates(subset="per")
    for _, peak in labelled.iterrows():
        order = peak["harmonic_order"]
        if pd.isna(order) and not pd.isna(peak.get("comb_order", pd.NA)):
            text = f"{int(peak['comb_order'])}/2"
        elif pd.isna(order):
            text = f"{peak['per']:.3f} d"
        elif order == 1:
            text = f"P0 = {float(peak['period_series']):.3f} d"
        else:
            text = f"{int(order)}x"
        axis.annotate(text, (peak["per"], peak["power"]),
                      xytext=(4, 4), textcoords="offset points", fontsize=7)


def finish_panel(axis, tic, sector, n_peaks):
    axis.set_xscale("log")
    axis.set_title(f"TIC {tic} — sector {sector}  ({n_peaks} peaks)", fontsize=10)
    axis.legend(loc="best", fontsize=7)
    axis.grid(alpha=0.3, which="both")


def build_pdf(outpath, pairs, periodograms, draw_panel, source_label, per_page,
              annotate_top, panel_kwargs=None):
    n_missing = 0
    with PdfPages(outpath) as pdf:
        for page_start in range(0, len(pairs), per_page):
            page_pairs = pairs[page_start:page_start + per_page]
            figure, axes = plt.subplots(len(page_pairs), 1,
                                        figsize=(11, 2.9 * len(page_pairs)),
                                        squeeze=False)
            for axis, (tic, sector) in zip(axes[:, 0], page_pairs):
                periodogram = periodograms.get((tic, sector))
                if periodogram is None or periodogram.empty:
                    axis.text(0.5, 0.5, f"TIC {tic} — sector {sector}\n"
                                        f"no {source_label} periodogram stored",
                              ha="center", va="center", fontsize=10, color="tab:red")
                    axis.set_axis_off()
                    n_missing += 1
                    continue
                draw_panel(axis, periodogram, tic, sector, annotate_top,
                           **(panel_kwargs or {}))
            axes[-1, 0].set_xlabel("Period / Lag [d]", fontsize=10)
            figure.suptitle(f"{source_label} peak review — page "
                            f"{page_start // per_page + 1}", fontsize=11)
            figure.tight_layout(rect=(0, 0, 1, 0.98))
            pdf.savefig(figure, dpi=150)
            plt.close(figure)
    return n_missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", type=int, default=10, help="pages per PDF")
    parser.add_argument("--per-page", type=int, default=5, help="panels per page")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed; omit for a different sample every run")
    parser.add_argument("--annotate-top", type=int, default=5,
                        help="label only the N most prominent peaks (0 = all)")
    parser.add_argument("--sample-from", choices=["acf", "ls", "peaks"], default="acf",
                        help="parquet the random (TIC, sector) pairs are drawn "
                             "from; 'peaks' restricts the sample to stars that "
                             "already have peaks stored")
    parser.add_argument("--data-dir", default=None,
                        help="directory holding the parquets, overriding "
                             "config (e.g. test_data while Dropbox syncs)")
    parser.add_argument("--outdir", default=config.RESULTS_DIR / "figures")
    parser.add_argument("--stored-band", action="store_true",
                        help="usar la banda de Bartlett extendido tal como "
                             "está guardada en el parquet, en vez de "
                             "des-inflarla a la banda de ruido blanco")
    parser.add_argument("--no-open", action="store_true",
                        help="do not open the PDFs when finished")
    args = parser.parse_args()

    if args.data_dir is None:
        ls_path, acf_path = config.PERIODOGRAMS_LS, config.PERIODOGRAMS_ACF
        peaks_path = config.PEAKS_PARQUET
    else:
        data_dir = config.REPO_ROOT / args.data_dir
        ls_path = data_dir / "periodograms_ls.parquet"
        acf_path = data_dir / "periodograms_acf.parquet"
        peaks_path = data_dir / "peaks.parquet"

    sample_path = {"acf": acf_path, "ls": ls_path, "peaks": peaks_path}[args.sample_from]

    for path in (ls_path, acf_path, sample_path):
        # Dropbox leaves online-only placeholders as 0-byte files; pyarrow then
        # fails with an opaque "Parquet file size is 0 bytes".
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"{path} missing or 0 bytes (Dropbox not synced yet?)")

    pairs = sample_pairs(sample_path, args.pages * args.per_page, seed=args.seed)
    print(f"{len(pairs)} random (TIC, sector) pairs from {sample_path}")

    outdir = config.REPO_ROOT / Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for source_label, path, draw_panel, panel_kwargs in [
            ("LS", ls_path, draw_ls_panel, {}),
            ("ACF", acf_path, draw_acf_panel, {"stored_band": args.stored_band})]:
        periodograms = load_periodograms(path, pairs)
        outpath = outdir / f"peak_review_{source_label.lower()}.pdf"
        n_missing = build_pdf(outpath, pairs, periodograms, draw_panel,
                              source_label, args.per_page, args.annotate_top,
                              panel_kwargs)
        print(f"{outpath}  ({args.pages} pages"
              + (f", {n_missing} pairs without periodogram)" if n_missing else ")"))
        outputs.append(outpath)

    if not args.no_open:
        open_in_preview(*outputs)


if __name__ == "__main__":
    main()
