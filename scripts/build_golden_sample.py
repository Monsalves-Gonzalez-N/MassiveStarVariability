"""Assemble a golden sample of human-classified TESS OB/B star variability.

Reads the raw published tables stored in catalogs/golden/ (downloaded from
VizieR, from the AAS machine-readable tables and from the arXiv LaTeX source)
and writes catalogs/golden_sample.csv plus the overlap statistics used in
docs/GOLDEN_SAMPLE.md.
"""

import pathlib
import re

import pandas


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN_DIRECTORY = REPOSITORY_ROOT / "catalogs" / "golden"
OUTPUT_CATALOG = REPOSITORY_ROOT / "catalogs" / "golden_sample.csv"

CONTROLLED_VOCABULARY_PRIORITY = [
    "ECL",
    "ELL",
    "BE",
    "ROT",
    "PULS",
    "SLF",
    "OTHER",
    "AMBIGUOUS",
    "NOISY",
]

# Token-level dictionary. The raw labels are split on separators before lookup,
# so that "EB" and "BE" or "SPB" and "SB" never collide by substring matching.
TOKEN_TO_CLASS = {
    "rot": "ROT",
    "rotation": "ROT",
    "rotational": "ROT",
    "acv": "ROT",
    "sxari": "ROT",
    "sxarl": "ROT",
    "msrot": "ROT",
    "spb": "PULS",
    "bcep": "PULS",
    "betacep": "PULS",
    "beta cep": "PULS",
    "maia": "PULS",
    "acyg": "PULS",
    "deltasct": "PULS",
    "dsct": "PULS",
    "puls": "PULS",
    "pul": "PULS",
    "pulsation": "PULS",
    "oscillation": "PULS",
    "igw": "PULS",
    "hybrid": "PULS",
    "sdbhybrid": "PULS",
    "eb": "ECL",
    "ea": "ECL",
    "eclipsingbinary": "ECL",
    "transit": "ECL",
    "transity": "ECL",
    "ell": "ELL",
    "ev": "ELL",
    "ellipsoidal": "ELL",
    "be": "BE",
    "gcas": "BE",
    "oe": "BE",
    "em": "BE",
    "outburst": "BE",
    "slf": "SLF",
    "sdb": "OTHER",
    "hs": "OTHER",
    "flare": "OTHER",
    "flares": "OTHER",
    "binary": "OTHER",
    "bin": "OTHER",
    "sb": "OTHER",
    "sb1": "OTHER",
    "sb2": "OTHER",
    "double": "OTHER",
    "multiple": "OTHER",
    "multi": "OTHER",
    "ep": "OTHER",
    "ambiguous": "AMBIGUOUS",
    "instr": "AMBIGUOUS",
    "instrumental": "AMBIGUOUS",
    "cont": "AMBIGUOUS",
    "contaminated": "AMBIGUOUS",
    "pq": "AMBIGUOUS",
    "noisy": "NOISY",
    "const": "NOISY",
    "constant": "NOISY",
}

TOKEN_SPLIT_PATTERN = re.compile(r"[+/,;()\[\]&]| - ")


def normalise_label(raw_label):
    """Map a published variability label onto the controlled vocabulary.

    Returns (primary_class, sorted_token_list). Question marks are stripped
    before lookup, so an uncertain label keeps its physical class; the
    uncertainty is preserved verbatim in the class_raw column.
    """
    if raw_label is None:
        return "", []
    text = str(raw_label).strip()
    if text in ("", "-", "--", "---", "nan", "..."):
        return "", []
    text = text.replace("{Beta}", "beta").replace("{beta}", "beta")
    text = text.replace("{delta}", "delta").replace("$", "")
    text = text.replace("\\beta", "beta").replace("\\delta", "delta")
    text = re.sub(r"\{|\}", "", text)
    text = re.sub(r"P=[^)]*", "", text)
    text = re.sub(r"nu_?inst[^)]*", "instr", text)

    found_classes = set()
    for piece in TOKEN_SPLIT_PATTERN.split(text):
        token = piece.strip().strip("?:.").lower()
        token = token.replace(" ", "")
        token = re.sub(r"^\d+$", "", token)
        if not token:
            continue
        if token in TOKEN_TO_CLASS:
            found_classes.add(TOKEN_TO_CLASS[token])
            continue
        # A couple of concatenated labels that survive the split.
        for compound, mapped in (
            ("betacep", "PULS"),
            ("deltasct", "PULS"),
            ("mini-outburst", "BE"),
            ("multiflares", "OTHER"),
        ):
            if compound in token:
                found_classes.add(mapped)
    if not found_classes:
        return "OTHER", ["OTHER"]
    # When an author states in words that the light curve is noisy or that the
    # classification is ambiguous, that statement is the verdict and it wins
    # over any physical class also mentioned in the same remark.
    lowered = text.lower()
    if "noisy" in lowered:
        return "NOISY", sorted(found_classes)
    if "ambiguous" in lowered:
        return "AMBIGUOUS", sorted(found_classes)
    for candidate in CONTROLLED_VOCABULARY_PRIORITY:
        if candidate in found_classes:
            primary = candidate
            break
    return primary, sorted(found_classes)


def read_vizier_tsv(path):
    """Parse the TSV that the VizieR asu-tsv endpoint returns."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_lines = []
    column_names = None
    for index, line in enumerate(lines):
        if line.startswith("#") or not line.strip():
            continue
        if set(line.replace("\t", "")) <= set("-"):
            column_names = [name.strip() for name in lines[index - 2].split("\t")]
            data_lines = []
            continue
        if column_names is not None:
            data_lines.append(line.split("\t"))
    frame = pandas.DataFrame(data_lines, columns=column_names)
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def read_machine_readable_table(path, column_slices):
    """Parse an AAS machine-readable table given {name: (start, stop)} in 1-based
    inclusive byte columns, as printed in the MRT header."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    dash_positions = [i for i, line in enumerate(lines) if line.startswith("-----")]
    first_data_line = dash_positions[-1] + 1
    records = []
    for line in lines[first_data_line:]:
        if not line.strip():
            continue
        record = {}
        for name, (start, stop) in column_slices.items():
            record[name] = line[start - 1:stop].strip()
        records.append(record)
    return pandas.DataFrame(records)


def make_rows(tic_values, class_raw_values, reference, method, period_values=None,
              class_is_inferred=False, source_table=""):
    rows = []
    for index, tic in enumerate(tic_values):
        raw = class_raw_values[index]
        primary, tokens = normalise_label(raw)
        period = None
        if period_values is not None:
            period = period_values[index]
        rows.append(
            {
                "TIC": int(tic),
                "class_raw": "" if raw is None else str(raw).strip(),
                "class_paper": primary,
                "class_tokens": "|".join(tokens),
                "reference": reference,
                "method": method,
                "period": period,
                "class_is_inferred": class_is_inferred,
                "source_table": source_table,
            }
        )
    return rows


def to_float(value):
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number


def build_balona_2019():
    frame = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_MNRAS_485_3457_table1.tsv")
    periods = []
    for frequency in frame["nurot"]:
        value = to_float(frequency)
        if value is None or value == 0.0:
            periods.append(None)
        else:
            periods.append(1.0 / value)
    return make_rows(
        tic_values=[int(name) for name in frame["Name"]],
        class_raw_values=list(frame["VarType"]),
        reference="2019MNRAS.485.3457B (Balona+ 2019)",
        method="visual",
        period_values=periods,
        source_table="J/MNRAS/485/3457/table1",
    )


def build_pedersen_2019():
    frame = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_ApJ_872_L9_table1.tsv")
    return make_rows(
        tic_values=[int(tic) for tic in frame["TIC"]],
        class_raw_values=list(frame["VarType"]),
        reference="2019ApJ...872L...9P (Pedersen+ 2019)",
        method="visual+spectroscopy",
        source_table="J/ApJ/872/L9/table1",
    )


def build_burssens_2020():
    table2 = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_A_A_639_A81_table2.tsv")
    tablea1 = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_A_A_639_A81_tablea1.tsv")
    name_to_tic = {}
    name_to_dominant_frequency = {}
    for _, row in tablea1.iterrows():
        name_to_tic[row["Name"]] = row["TIC"]
        name_to_dominant_frequency[row["Name"]] = row["nudom"]

    tic_values = []
    class_raw_values = []
    period_values = []
    missing_tic_names = []
    for _, row in table2.iterrows():
        tic = name_to_tic.get(row["Name"], "")
        if not tic:
            missing_tic_names.append(row["Name"])
            continue
        frequency = to_float(name_to_dominant_frequency.get(row["Name"], ""))
        tic_values.append(int(tic))
        class_raw_values.append(row["VarTypeT"])
        if frequency is None or frequency == 0.0:
            period_values.append(None)
        else:
            period_values.append(1.0 / frequency)
    if missing_tic_names:
        print(f"Burssens 2020: no TIC for {len(missing_tic_names)} names: {missing_tic_names}")
    return make_rows(
        tic_values=tic_values,
        class_raw_values=class_raw_values,
        reference="2020A&A...639A..81B (Burssens+ 2020)",
        method="visual+spectroscopy",
        period_values=period_values,
        source_table="J/A+A/639/A81/table2 x tablea1",
    )


def build_bowman_2020():
    tablea2 = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_A_A_640_A36_tablea2.tsv")
    tic_values = [int(tic) for tic in tablea2["TIC"] if tic]
    class_raw_values = ["SLF"] * len(tic_values)
    return make_rows(
        tic_values=tic_values,
        class_raw_values=class_raw_values,
        reference="2020A&A...640A..36B (Bowman+ 2020)",
        method="visual+spectroscopy (SLF morphology fit)",
        class_is_inferred=True,
        source_table="J/A+A/640/A36/tablea2",
    )


def build_labadie_bartz_2022():
    frame = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_AJ_163_226_table2.tsv")
    tic_values = []
    class_raw_values = []
    period_values = []
    for _, row in frame.iterrows():
        quality = row["q"]
        be_flag = row["Be"]
        signals = row["signals"]
        if quality == "0" or be_flag == "":
            continue
        if be_flag in ("Y", "S"):
            raw = f"Be={be_flag}; signals={signals}"
        else:
            raw = f"Be={be_flag} (rejected/other); signals={signals}"
        frequency = to_float(row["fg1"])
        tic_values.append(int(row["TIC"]))
        class_raw_values.append(raw)
        if frequency is None or frequency == 0.0:
            period_values.append(None)
        else:
            period_values.append(1.0 / frequency)

    rows = []
    for index, tic in enumerate(tic_values):
        raw = class_raw_values[index]
        if raw.startswith("Be=Y") or raw.startswith("Be=S"):
            primary, tokens = "BE", ["BE"]
        else:
            primary, tokens = "OTHER", ["OTHER"]
        rows.append(
            {
                "TIC": tic,
                "class_raw": raw,
                "class_paper": primary,
                "class_tokens": "|".join(tokens),
                "reference": "2022AJ....163..226L (Labadie-Bartz+ 2022)",
                "method": "visual",
                "period": period_values[index],
                "class_is_inferred": False,
                "source_table": "J/AJ/163/226/table2",
            }
        )
    return rows


LATEX_CLEANUP_PATTERN = re.compile(r"\\cite[a-z]*\{[^}]*\}|\\citet\{[^}]*\}|\\[a-zA-Z]+|\{|\}")


def parse_barraza_latex_table(latex_text, table_number, tic_column, class_column,
                              period_column=None):
    """Extract one deluxetable-like block from the arXiv LaTeX source."""
    marker = f"\\tablenum{{{table_number}}}"
    start = latex_text.index(marker)
    body_start = latex_text.index("\\hline", latex_text.index("\\begin{tabular}", start))
    body_end = latex_text.index("\\end{tabular}", body_start)
    body = latex_text[body_start:body_end]

    records = []
    for line in body.split("\\\\"):
        line = line.replace("\\hline", " ").strip()
        if not line or "&" not in line:
            continue
        cells = [cell.strip() for cell in line.split("&")]
        first_cell = LATEX_CLEANUP_PATTERN.sub("", cells[tic_column]).strip()
        if not re.fullmatch(r"\d{4,}", first_cell):
            continue
        raw_class = LATEX_CLEANUP_PATTERN.sub("", cells[class_column]).strip()
        period = None
        if period_column is not None:
            period_text = re.sub(r"\(.*\)", "", cells[period_column]).strip()
            period = to_float(period_text)
        records.append((int(first_cell), raw_class, period))
    return records


def build_barraza_2022():
    latex_text = (GOLDEN_DIRECTORY / "barraza2022_src" / "arxiv.tex").read_text(
        encoding="utf-8", errors="replace"
    )
    rows = []

    rotation = parse_barraza_latex_table(latex_text, 2, tic_column=0, class_column=8,
                                         period_column=2)
    for tic, remark, period in rotation:
        raw = "Rotation" if not remark else f"Rotation ({remark})"
        primary, tokens = normalise_label(raw)
        rows.append(
            {
                "TIC": tic,
                "class_raw": raw,
                "class_paper": "ROT",
                "class_tokens": "|".join(sorted(set(tokens) | {"ROT"})),
                "reference": "2022ApJ...924....2B (Barraza+ 2022)",
                "method": "visual",
                "period": period,
                "class_is_inferred": False,
                "source_table": "arXiv:2202.01022 Table 2",
            }
        )

    noisy = parse_barraza_latex_table(latex_text, 3, tic_column=0, class_column=4)
    for tic, remark, _ in noisy:
        primary, tokens = normalise_label(remark)
        rows.append(
            {
                "TIC": tic,
                "class_raw": remark,
                "class_paper": primary,
                "class_tokens": "|".join(tokens),
                "reference": "2022ApJ...924....2B (Barraza+ 2022)",
                "method": "visual",
                "period": None,
                "class_is_inferred": False,
                "source_table": "arXiv:2202.01022 Table 3",
            }
        )

    ambiguous = parse_barraza_latex_table(latex_text, 4, tic_column=0, class_column=4)
    for tic, remark, _ in ambiguous:
        primary, tokens = normalise_label(remark)
        rows.append(
            {
                "TIC": tic,
                "class_raw": remark,
                "class_paper": primary,
                "class_tokens": "|".join(tokens),
                "reference": "2022ApJ...924....2B (Barraza+ 2022)",
                "method": "visual",
                "period": None,
                "class_is_inferred": False,
                "source_table": "arXiv:2202.01022 Table 4",
            }
        )
    return rows


def build_campelo_2025():
    reference = "2025ApJ...989..177C (Campelo+ 2025)"
    rows = []

    rotation = pandas.read_csv(GOLDEN_DIRECTORY / "campelo2025_table2_rotation_from_article.csv")
    for _, row in rotation.iterrows():
        rows.append(
            {
                "TIC": int(row["TIC"]),
                "class_raw": "Rotation",
                "class_paper": "ROT",
                "class_tokens": "ROT",
                "reference": reference,
                "method": "visual",
                "period": row["P_rot_d"],
                "class_is_inferred": False,
                "source_table": "ApJ 989,177 Table 2 (article HTML)",
            }
        )

    pulsation = pandas.read_csv(GOLDEN_DIRECTORY / "campelo2025_table4_pulsation_from_article.csv")
    for _, row in pulsation.iterrows():
        notes = "" if pandas.isna(row["notes"]) else str(row["notes"])
        raw = "Pulsation" if not notes else f"Pulsation ({notes})"
        first_period = to_float(str(row["P_A_d"]).split("/")[0])
        rows.append(
            {
                "TIC": int(row["TIC"]),
                "class_raw": raw,
                "class_paper": "PULS",
                "class_tokens": "PULS",
                "reference": reference,
                "method": "visual",
                "period": first_period,
                "class_is_inferred": False,
                "source_table": "ApJ 989,177 Table 4 (article HTML)",
            }
        )

    binary = pandas.read_csv(GOLDEN_DIRECTORY / "campelo2025_table5_binary_from_article.csv")
    for _, row in binary.iterrows():
        rows.append(
            {
                "TIC": int(row["TIC"]),
                "class_raw": "Binary (eclipsing/ellipsoidal)",
                "class_paper": "ECL",
                "class_tokens": "ECL",
                "reference": reference,
                "method": "visual",
                "period": row["P_A_d"],
                "class_is_inferred": False,
                "source_table": "ApJ 989,177 Table 5 (article HTML)",
            }
        )

    subdwarf = read_machine_readable_table(
        GOLDEN_DIRECTORY / "campelo2025_table6_mrt.txt",
        {"TIC": (1, 9), "PA": (11, 15), "VarType": (42, 62)},
    )
    for _, row in subdwarf.iterrows():
        raw = "sdB candidate" if not row["VarType"] else f"sdB candidate ({row['VarType']})"
        rows.append(
            {
                "TIC": int(row["TIC"]),
                "class_raw": raw,
                "class_paper": "OTHER",
                "class_tokens": "OTHER",
                "reference": reference,
                "method": "visual",
                "period": to_float(row["PA"]),
                "class_is_inferred": False,
                "source_table": "ApJ 989,177 Table 6 (MRT)",
            }
        )

    noisy = read_machine_readable_table(
        GOLDEN_DIRECTORY / "campelo2025_table3_mrt.txt", {"TIC": (1, 9)}
    )
    for _, row in noisy.iterrows():
        rows.append(
            {
                "TIC": int(row["TIC"]),
                "class_raw": "Noisy",
                "class_paper": "NOISY",
                "class_tokens": "NOISY",
                "reference": reference,
                "method": "visual",
                "period": None,
                "class_is_inferred": False,
                "source_table": "ApJ 989,177 Table 3 (MRT)",
            }
        )

    ambiguous = pandas.read_csv(GOLDEN_DIRECTORY / "campelo2025_table7_ambiguous_from_article.csv")
    for _, row in ambiguous.iterrows():
        rows.append(
            {
                "TIC": int(row["TIC"]),
                "class_raw": "Ambiguous variability",
                "class_paper": "AMBIGUOUS",
                "class_tokens": "AMBIGUOUS",
                "reference": reference,
                "method": "visual",
                "period": None,
                "class_is_inferred": False,
                "source_table": "ApJ 989,177 Table 7 (article HTML)",
            }
        )
    return rows


def build_coordinate_lookup():
    """TIC -> (ra, dec) from the source tables that publish positions.

    Barraza+ 2022 and Campelo+ 2025 publish no coordinates, so their stars keep
    empty positions unless another paper in the sample covers them.
    """
    lookup = {}

    balona = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_MNRAS_485_3457_table1.tsv")
    for _, row in balona.iterrows():
        lookup[int(row["Name"])] = (to_float(row["_RA"]), to_float(row["_DE"]))

    pedersen = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_ApJ_872_L9_table1.tsv")
    for _, row in pedersen.iterrows():
        lookup[int(row["TIC"])] = (to_float(row["_RA"]), to_float(row["_DE"]))

    bowman = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_A_A_640_A36_tablea1.tsv")
    for _, row in bowman.iterrows():
        if row["TIC"]:
            lookup[int(row["TIC"])] = (to_float(row["_RA"]), to_float(row["_DE"]))

    labadie_bartz = read_vizier_tsv(GOLDEN_DIRECTORY / "v_J_AJ_163_226_table2.tsv")
    for _, row in labadie_bartz.iterrows():
        lookup[int(row["TIC"])] = (to_float(row["RAJ2000"]), to_float(row["DEJ2000"]))

    return lookup


def main():
    all_rows = []
    all_rows.extend(build_balona_2019())
    all_rows.extend(build_pedersen_2019())
    all_rows.extend(build_burssens_2020())
    all_rows.extend(build_bowman_2020())
    all_rows.extend(build_labadie_bartz_2022())
    all_rows.extend(build_barraza_2022())
    all_rows.extend(build_campelo_2025())

    golden = pandas.DataFrame(all_rows)
    coordinate_lookup = build_coordinate_lookup()
    golden["ra_deg"] = golden["TIC"].map(
        lambda tic: coordinate_lookup.get(tic, (None, None))[0]
    )
    golden["dec_deg"] = golden["TIC"].map(
        lambda tic: coordinate_lookup.get(tic, (None, None))[1]
    )
    golden = golden.drop_duplicates(subset=["TIC", "reference", "source_table"])
    golden = golden.sort_values(["TIC", "reference"]).reset_index(drop=True)
    golden.to_csv(OUTPUT_CATALOG, index=False)

    print(f"wrote {OUTPUT_CATALOG} with {len(golden)} rows, "
          f"{golden['TIC'].nunique()} unique TIC")
    print(golden.groupby("reference")["TIC"].nunique())
    print(golden["class_paper"].value_counts())


if __name__ == "__main__":
    main()
