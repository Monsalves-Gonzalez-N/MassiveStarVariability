#!/usr/bin/env python
"""Prepara la revisión visual del benchmark VSX.

Arma `catalogs/vsx_visual_review.csv` (las 567 masivas con label 0/1, con el
período VSX, el detectado y la clase del BRF como contexto) y cachea sus
curvas limpias en `results/lc_vsx_review.parquet` para que el revisor del
notebook no filtre el parquet de 73M filas estrella por estrella.

    python scripts/build_vsx_review.py [--model Number_DST] [--no-cache]

Es idempotente: los veredictos `vis_*` ya marcados se preservan.
"""
import argparse

import pandas as pd

from msv import config, review


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Number_DST",
                    help="modelo del CSV results/brf_mc_vsx_<model>.csv")
    ap.add_argument("--no-cache", action="store_true",
                    help="no reconstruir el parquet de curvas")
    args = ap.parse_args()

    xm = review.load_vsx_xmatch()
    preds = pd.read_csv(config.RESULTS_DIR / f"brf_mc_vsx_{args.model}.csv")
    preds = preds.merge(xm[["TIC", "per_vsx"]], on="TIC", how="left")

    tab = review.build_review_table(xm, preds)
    print(f"{review.REVIEW_CSV}: {len(tab)} estrellas")
    print(tab["grupo"].value_counts().to_string())
    marcadas = (tab["vis_verdict"].fillna("") != "").sum()
    print(f"veredictos preservados: {marcadas}")

    if not args.no_cache:
        review.build_lc_cache(tab["TIC"])


if __name__ == "__main__":
    main()
