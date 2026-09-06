#!/usr/bin/env python
"""Leer el formulario del PDF de revisión y volcarlo a CSV.

Cada fila del PDF tiene cuatro checkboxes y un campo de notas. Tres de ellos
contestan quién tiene el período bueno y son excluyentes:

    periodico    -> golden    el período publicado cierra
    irregular    -> ninguno   no hay período que defender
    half period  -> nuestro   el publicado es la mitad del verdadero

`multi periodico` es **ortogonal** a esos tres y sale en su propia columna: una
estrella puede ser multiperiódica y además tener mal el período reportado. Se
marca junto con otro, no en su lugar.

Las estrellas cuyos sectores no coinciden quedan marcadas en `conflicto`, que
es información y no un error: un sector corto puede no mostrar lo que muestra
uno largo.

    PYTHONPATH=src python scripts/golden/read_veredicto.py
"""
import argparse
from collections import Counter
from pathlib import Path

import fitz
import pandas as pd

from msv.config import RESULTS_DIR

CHOICE_TO_VERDICT = {
    "periodico": "golden",
    "irregular": "ninguno",
    "half_period": "nuestro",
}
MULTI = "multi_periodico"
# Un checkbox sin marcar vuelve como el string "Off", que en Python es
# verdadero; hay que comparar contra el estado, no evaluar el valor.
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
            key, field = name.rsplit("__", 1)
            answers.setdefault(key, {})[field] = widget.field_value
    document.close()
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", default=str(RESULTS_DIR / "figures" / "golden_review_2P.pdf"))
    parser.add_argument("--peaks", default=str(RESULTS_DIR / "golden" / "match_peaks.csv"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    output = Path(args.out) if args.out else RESULTS_DIR / "golden" / "veredicto_2P.csv"
    answers = read_widgets(args.pdf)

    peaks = pd.read_csv(args.peaks)
    reported = peaks[peaks.reportado].copy()
    reported["key"] = (reported.TIC.astype(int).astype(str) + "_"
                       + reported.sector.astype(int).astype(str))
    reported = reported.set_index("key")

    rows = []
    for key, fields in sorted(answers.items()):
        marked = [choice for choice in CHOICE_TO_VERDICT
                  if is_checked(fields.get(choice))]
        multi = is_checked(fields.get(MULTI))
        if key not in reported.index:
            continue
        source = reported.loc[key]
        rows.append({
            "TIC": int(source.TIC),
            "sector": int(source.sector),
            "class_gold": source.class_gold,
            "period_gold": source.period_gold,
            "per_nuestro": source.per,
            "clase_nuestra": source.clase,
            "log_pLPV": round(float(source.log_pLPV), 2),
            "nivel": source.nivel,
            "marcado": " + ".join(marked + ([MULTI] if multi else [])),
            "multi": multi,
            "veredicto": (CHOICE_TO_VERDICT[marked[0]] if len(marked) == 1
                          else ("" if not marked else "conflicto")),
            "nota": (fields.get("nota") or "").strip().replace("\n", " "),
        })
    per_sector = pd.DataFrame(rows).sort_values(["TIC", "sector"])

    star_rows = []
    for tic, block in per_sector.groupby("TIC"):
        answered = block[block.veredicto != ""]
        counts = Counter(answered.veredicto)
        answered_or_multi = block[(block.veredicto != "") | block.multi]
        star_rows.append({
            "TIC": tic,
            "class_gold": block.class_gold.iloc[0],
            "clase_nuestra": block.clase_nuestra.mode().iloc[0],
            "period_gold": block.period_gold.iloc[0],
            "sectores": len(block),
            "respondidos": len(answered_or_multi),
            "multi": bool(block.multi.any()),
            "multi_fraccion": round(float(block.multi.mean()), 2),
            "veredicto": counts.most_common(1)[0][0] if counts else "",
            "conflicto": len(counts) > 1,
            "detalle": " ".join(f"s{row.sector}:{row.veredicto or '-'}"
                                for row in block.itertuples()),
            "notas": " | ".join(note for note in block.nota if note),
        })
    per_star = pd.DataFrame(star_rows)

    per_sector.to_csv(output, index=False)
    per_star.to_csv(output.with_name(output.stem + "_por_estrella.csv"), index=False)

    answered = int((per_sector.veredicto != "").sum())
    print(f"{answered} de {len(per_sector)} estrella-sector respondidas "
          f"({per_star.respondidos.gt(0).sum()} de {len(per_star)} estrellas)")
    if answered:
        print("\n=== veredicto por estrella ===")
        print(per_star.veredicto.value_counts().to_string())
        print("\n=== cruzado con nuestra clase ===")
        decided = per_star[per_star.veredicto != ""]
        print(pd.crosstab(decided.clase_nuestra, decided.veredicto).to_string())
        print("\n=== cruzado con la clase del paper ===")
        print(pd.crosstab(decided.class_gold, decided.veredicto).to_string())
        print("\n=== multi periodico ===")
        print(pd.crosstab(per_star.class_gold, per_star.multi).to_string())
        conflicting = per_star[per_star.conflicto]
        if len(conflicting):
            print(f"\n{len(conflicting)} estrellas con sectores en conflicto:")
            print(conflicting[["TIC", "class_gold", "detalle"]].to_string(index=False))
    conflicting_row = per_sector[per_sector.veredicto == "conflicto"]
    if len(conflicting_row):
        print(f"\n{len(conflicting_row)} filas con más de un excluyente marcado:")
        print(conflicting_row[["TIC", "sector", "marcado"]].to_string(index=False))
    print(f"\nescrito {output}")


if __name__ == "__main__":
    main()
