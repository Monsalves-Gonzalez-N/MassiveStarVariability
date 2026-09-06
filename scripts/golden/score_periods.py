#!/usr/bin/env python
"""Etapa 2: puntuar el período reportado contra la golden, con tres veredictos.

`match_periods.py` puntúa binario: el pico reportado cae en 1:1 con el período
publicado o no. Eso hace dos supuestos que la muestra no sostiene, y los dos
inflan el error:

1. **Que el período publicado existe.** No siempre. Las Be ya salieron por
   `period_is_group` (su "período" es el centro de un grupo de frecuencias),
   pero incluso dentro de las clases que sí tienen período hay estrella-sector
   donde el valor publicado no tiene potencia en ESTE sector. Contra un número
   que no está en los datos no se puede acertar.

2. **Que cuando reportamos 2P nos equivocamos.** Una curva de doble onda es
   fotométricamente IDÉNTICA plegada en P y en 2P: si los dos semiciclos son
   iguales, los armónicos impares valen cero y no existe medición que separe
   las dos hipótesis. La única evidencia posible a favor de doblar es que los
   semiciclos difieran, o sea potencia en el fundamental de 2P — eclipses de
   distinta profundidad, dos grupos de manchas desiguales. Así que el 2P se
   adjudica con la sonda, no con el paper:

       SNR(fundamental en NUESTRO período) >= 4   -> el doblado es real
       solo el publicado tiene potencia           -> doblamos sin evidencia
       ninguno de los dos                         -> no hay señal que arbitrar

   El caso del medio NO es equivalente a "el período está mal": la frecuencia
   reportada es la misma señal, elegida en el armónico equivocado. Contarlo
   junto con `otro` — donde reportamos una frecuencia que no tiene relación con
   la publicada — mezcla un error de convención con un error de detección.

Los veredictos son entonces:

    acierto        1:1 con el publicado
    doblado_ok     armónico del publicado, y el fundamental nuestro tiene señal
    doblado_sin_ev armónico del publicado, sin potencia propia: misma señal,
                   armónico equivocado
    fallo          otra frecuencia, y el publicado sí está en los datos
    sin_senal      el período publicado no tiene potencia en este sector

El denominador honesto es `acierto + doblado_* + fallo`; `sin_senal` sale de la
métrica por el mismo motivo que SLF y las Be.

    PYTHONPATH=src python scripts/golden/score_periods.py
"""
import argparse

import numpy as np
import pandas as pd

from msv.config import CATALOGS_DIR, RESULTS_DIR

HARMONICS = {"1:1": 1.0, "2P": 2.0, "P/2": 0.5, "3P": 3.0, "P/3": 1.0 / 3.0}
TOLERANCE = 0.05
SNR_MIN = 4.0


def harmonic_tag(ratio):
    for name, target in HARMONICS.items():
        if np.abs(ratio / target - 1.0) < TOLERANCE:
            return name
    return "otro"


def verdict(tag, snr_gold, snr_ours):
    if tag == "1:1":
        return "acierto"
    gold_present = np.isfinite(snr_gold) and snr_gold >= SNR_MIN
    ours_present = np.isfinite(snr_ours) and snr_ours >= SNR_MIN
    if tag == "otro":
        return "fallo" if gold_present else "sin_senal"
    if ours_present:
        return "doblado_ok"
    if gold_present:
        return "doblado_sin_ev"
    return "sin_senal"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets",
                        default=str(RESULTS_DIR / "golden" / "probe_targets.csv"))
    parser.add_argument("--probes",
                        default=str(RESULTS_DIR / "golden" / "probe_all" /
                                    "prewhiten_components.csv"))
    parser.add_argument("--truth", default=str(CATALOGS_DIR / "golden_truth.csv"))
    parser.add_argument("--out",
                        default=str(RESULTS_DIR / "golden" / "score_periods.csv"))
    args = parser.parse_args()

    targets = pd.read_csv(args.targets)
    probes = pd.read_csv(args.probes)
    truth = pd.read_csv(args.truth)

    snr = probes[probes.kind.isin(["GOLD", "OURS"])].pivot_table(
        index=["TIC", "sector"], columns="kind", values="snr")
    amplitude = probes[probes.kind.isin(["GOLD", "OURS"])].pivot_table(
        index=["TIC", "sector"], columns="kind", values="amplitude_ppt")

    scored = targets.merge(
        snr.rename(columns={"GOLD": "snr_gold", "OURS": "snr_ours"}),
        left_on=["TIC", "sector"], right_index=True, how="left")
    scored = scored.merge(
        amplitude.rename(columns={"GOLD": "amp_gold", "OURS": "amp_ours"}),
        left_on=["TIC", "sector"], right_index=True, how="left")

    scored["ratio"] = scored.per_nuestro / scored.period_gold
    scored["tag"] = [harmonic_tag(value) for value in scored.ratio]
    scored["veredicto"] = [verdict(row.tag, row.snr_gold, row.snr_ours)
                           for row in scored.itertuples(index=False)]
    scored.to_csv(args.out, index=False)
    print(f"escrito {args.out}")

    usable = scored[scored.period_usable]
    print(f"\n{len(usable)} estrella-sector con período comparable "
          f"({usable.TIC.nunique()} TIC); "
          f"{len(scored) - len(usable)} excluidas por period_usable")

    print("\n=== veredicto por clase publicada ===")
    table = pd.crosstab(usable.class_gold, usable.veredicto, margins=True)
    order = [name for name in ["acierto", "doblado_ok", "doblado_sin_ev",
                               "fallo", "sin_senal", "All"]
             if name in table.columns]
    print(table[order].to_string())

    decided = usable[usable.veredicto != "sin_senal"]
    correct = decided.veredicto.isin(["acierto", "doblado_ok"]).sum()
    same_signal = decided.veredicto.isin(["acierto", "doblado_ok",
                                          "doblado_sin_ev"]).sum()
    print(f"\nsobre las {len(decided)} donde el período publicado ESTÁ en los "
          f"datos:")
    print(f"  período correcto                 {correct:3d}  "
          f"{correct / len(decided):.1%}")
    print(f"  la señal correcta (algún armónico) {same_signal:3d}  "
          f"{same_signal / len(decided):.1%}")

    print("\n=== el mismo corte sobre las excluidas, para el registro ===")
    for label, block in [("Be (período de grupo)",
                          scored[scored.period_is_group]),
                         ("no usable, otras", scored[~scored.period_usable
                                                     & ~scored.period_is_group])]:
        if len(block):
            counts = block.veredicto.value_counts().to_dict()
            print(f"{label}: n={len(block)}  {counts}")

    print("\n=== potencia del fundamental, publicado vs nuestro ===")
    summary = usable.groupby("class_gold").agg(
        n=("TIC", "size"),
        snr_gold=("snr_gold", "median"),
        snr_ours=("snr_ours", "median"),
        gold_con_senal=("snr_gold", lambda s: (s >= SNR_MIN).mean()),
        nuestro_con_senal=("snr_ours", lambda s: (s >= SNR_MIN).mean()),
    )
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
