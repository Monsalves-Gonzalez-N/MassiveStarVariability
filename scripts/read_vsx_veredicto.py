#!/usr/bin/env python
"""Leer el formulario de `build_vsx_pdf_review.py` y pegarlo a la revisión VSX.

`multiperiodica` es ORTOGONAL a `ok`/`maybe`/`bad`/`sin_lc`, no un veredicto
excluyente: marcar `ok` + `multiperiodica` significa "hay más de un período
pero el dominante (`per_vsx`) se ve claro en el fold" — sigue siendo un OK
utilizable, con la salvedad anotada. El veredicto final por estrella:

    exactamente uno de ok/maybe/bad/sin_lc marcado   -> ese, en `vis_verdict`
    multiperiodica también marcado                   -> además, `vis_multi`=True
    SOLO multiperiodica marcado (sin exclusivo)       -> `vis_verdict`="unconstrained"
    más de uno de ok/maybe/bad/sin_lc                 -> "conflicto" (revisar)

Se escribe en `catalogs/vsx_visual_review.csv` como `vis_verdict`, `vis_multi`
y `vis_notes`.

    PYTHONPATH=src python scripts/read_vsx_veredicto.py
"""
import argparse
import datetime as _dt

import fitz
import pandas as pd

from msv.config import CATALOGS_DIR, RESULTS_DIR

REVIEW_CSV = CATALOGS_DIR / "vsx_visual_review.csv"
EXCLUSIVE = ["ok", "maybe", "bad", "sin_lc"]
MULTI_FLAG = "multiperiodica"
UNCHECKED = {None, False, "", "Off"}


def is_checked(value):
    return value not in UNCHECKED


def read_widgets(pdf_path):
    document = fitz.open(pdf_path)
    answers = {}
    for page in document:
        for widget in page.widgets():
            name = widget.field_name
            if "__" not in name:
                continue
            tic, field = name.rsplit("__", 1)
            answers.setdefault(tic, {})[field] = widget.field_value
    document.close()
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", default=str(RESULTS_DIR / "figures" / "vsx_review.pdf"))
    parser.add_argument("--review", default=str(REVIEW_CSV))
    args = parser.parse_args()

    answers = read_widgets(args.pdf)
    tab = pd.read_csv(args.review)
    tab["TIC"] = tab["TIC"].astype(int)
    today = _dt.date.today().isoformat()

    updated = 0
    conflicts = []
    for tic_str, fields in answers.items():
        tic = int(tic_str)
        marked = [choice for choice in EXCLUSIVE if is_checked(fields.get(choice))]
        multi = is_checked(fields.get(MULTI_FLAG))
        note = (fields.get("nota") or "").strip().replace("\n", " ")
        if not marked and not multi and not note:
            continue
        if len(marked) == 1:
            verdict = marked[0]
        elif not marked:
            verdict = "unconstrained" if multi else ""
        else:
            verdict = "conflicto"
            conflicts.append((tic, marked))
        row_mask = tab["TIC"] == tic
        if not row_mask.any():
            continue
        tab.loc[row_mask, "vis_verdict"] = verdict
        tab.loc[row_mask, "vis_multi"] = multi
        tab.loc[row_mask, "vis_notes"] = note
        tab.loc[row_mask, "vis_date"] = today
        updated += 1

    tab.to_csv(args.review, index=False)
    print(f"{updated} estrellas actualizadas en {args.review}")
    if conflicts:
        print(f"\n{len(conflicts)} con más de un veredicto marcado:")
        for tic, marked in conflicts:
            print(f"  TIC {tic}: {marked}")

    reviewed = tab[tab["vis_verdict"].fillna("") != ""]
    if len(reviewed):
        print("\n=== veredicto sobre el fold en per_vsx ===")
        print(reviewed["vis_verdict"].value_counts().to_string())
        print("\n=== cruzado con expected_class (VSX) ===")
        print(pd.crosstab(reviewed["expected_class"].fillna("?"),
                          reviewed["vis_verdict"]).to_string())


if __name__ == "__main__":
    main()
