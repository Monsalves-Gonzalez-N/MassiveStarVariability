#!/usr/bin/env python
"""Enlaza los datos pesados de Dropbox dentro del repo (setup por máquina).

Los 36 GB de datos viven en Dropbox (ver config.DATA_DIR), no en git. Este
script deja en la raíz del repo un symlink por cada dataset, de modo que las
rutas relativas de los notebooks ("peaks.parquet", "cubos/...") sigan
funcionando igual que cuando los archivos estaban sueltos en el repo.

Uso:
  python scripts/link_data.py            # enlaza lo que exista, reporta lo que falte
  python scripts/link_data.py --check    # solo diagnostica, no escribe nada
  MSV_DATA_DIR=/otra/ruta python scripts/link_data.py

Es idempotente: correrlo dos veces no cambia nada. Nunca sobreescribe un
archivo real — si en el repo hay un parquet de verdad con ese nombre, avisa y
lo deja intacto para que decidas tú.
"""
import argparse
import sys

from msv import config

# (nombre en el repo, ruta real dentro de DATA_DIR)
LINKS = [
    ("cubos", config.RAW_DIR / "cubos"),
    ("download_paralell", config.RAW_DIR / "download_paralell"),
    ("ogle_download", config.RAW_DIR / "ogle_download"),
    ("CheckAperture", config.RAW_DIR / "CheckAperture"),
    ("lightcurves", config.RAW_DIR / "lightcurves"),
    ("lightcurves_all.parquet", config.LC_PARQUET_MASSIVE),
    ("lightcurves_all_OGLE.parquet", config.LC_PARQUET_OGLE),
    ("peaks.parquet", config.PEAKS_PARQUET),
    ("peaks_ls.parquet", config.DERIVED_DIR / "peaks_ls.parquet"),
    ("peaks_single.parquet", config.DERIVED_DIR / "peaks_single.parquet"),
    ("peaks_statsmodels.parquet", config.DERIVED_DIR / "peaks_statsmodels.parquet"),
    ("periodograms_ls.parquet", config.PERIODOGRAMS_LS),
    ("periodograms_acf.parquet", config.PERIODOGRAMS_ACF),
    ("train_number_M.csv", config.TRAIN_NUMBER_M),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="solo diagnostica; no crea ni borra symlinks")
    args = ap.parse_args()

    print(f"DATA_DIR = {config.DATA_DIR}")
    if config.DATA_DIR == config.REPO_ROOT:
        print("  ! No se encontró la carpeta de datos (¿Dropbox sin sincronizar?).")
        print("  ! Definí MSV_DATA_DIR o añadí la ruta a _DATA_DIR_CANDIDATES "
              "en src/msv/config.py")
        return 1

    n_ok = n_new = n_missing = n_conflict = 0
    for name, target in LINKS:
        link = config.REPO_ROOT / name

        if not target.exists():
            print(f"  FALTA     {name:32s} -> {target} (no existe en Dropbox)")
            n_missing += 1
            continue

        if link.is_symlink():
            if link.resolve() == target.resolve():
                n_ok += 1
                continue
            print(f"  REAPUNTA  {name:32s} -> {target}")
            if not args.check:
                link.unlink()
                link.symlink_to(target)
            n_new += 1
            continue

        if link.exists():
            print(f"  CONFLICTO {name:32s} ya existe como archivo real; se deja intacto")
            n_conflict += 1
            continue

        print(f"  ENLAZA    {name:32s} -> {target}")
        if not args.check:
            link.symlink_to(target)
        n_new += 1

    verb = "faltarían" if args.check else "creados/actualizados"
    print(f"\n{n_ok} ya correctos, {n_new} {verb}, "
          f"{n_missing} sin datos, {n_conflict} en conflicto")
    return 1 if (n_missing or n_conflict) else 0


if __name__ == "__main__":
    sys.exit(main())
