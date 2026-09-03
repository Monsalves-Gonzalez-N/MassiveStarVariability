"""Overlap and agreement statistics for catalogs/golden_sample.csv.

Prints the numbers that are quoted in docs/GOLDEN_SAMPLE.md: per-paper counts,
the pairwise overlap matrix, class agreement for the shared stars, and the
overlap with the user's own OB samples.
"""

import itertools
import pathlib

import pandas


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN_CATALOG = REPOSITORY_ROOT / "catalogs" / "golden_sample.csv"
PARENT_SAMPLE = REPOSITORY_ROOT / "catalogs" / "1_MassiveXTessV8.csv"
CLEAN_SAMPLE = REPOSITORY_ROOT / "catalogs" / "3_MassiveXTessV8_LC_DeltaMagnitude_Clean.csv"
REVIEW_SAMPLE = REPOSITORY_ROOT / "results" / "clasificacion_review50.csv"


def collapse_to_one_row_per_star_and_reference(golden):
    """A star can appear in two tables of the same paper. Keep the most
    physically specific label, following the vocabulary priority."""
    priority = {
        "ECL": 0, "ELL": 1, "BE": 2, "ROT": 3, "PULS": 4, "SLF": 5,
        "OTHER": 6, "AMBIGUOUS": 7, "NOISY": 8, "": 9,
    }
    golden = golden.copy()
    golden["class_paper"] = golden["class_paper"].fillna("")
    golden["priority"] = golden["class_paper"].map(lambda value: priority.get(value, 9))
    golden = golden.sort_values("priority")
    return golden.drop_duplicates(subset=["TIC", "reference"], keep="first")


def main():
    golden = pandas.read_csv(GOLDEN_CATALOG)
    collapsed = collapse_to_one_row_per_star_and_reference(golden)

    print("== rows and unique TIC per reference")
    summary = collapsed.groupby("reference").agg(
        n_stars=("TIC", "nunique"),
        n_classified=("class_paper", lambda values: (values != "").sum()),
    )
    print(summary.to_string())
    print(f"\ntotal rows in golden_sample.csv: {len(golden)}")
    print(f"total unique TIC: {golden['TIC'].nunique()}")
    print(f"unique TIC with a non-empty normalised class: "
          f"{collapsed[collapsed['class_paper'] != '']['TIC'].nunique()}")

    references = sorted(collapsed["reference"].unique())
    tic_sets = {
        reference: set(collapsed[collapsed["reference"] == reference]["TIC"])
        for reference in references
    }

    print("\n== pairwise overlap matrix (number of shared TIC)")
    matrix = pandas.DataFrame(index=references, columns=references, dtype=int)
    for first in references:
        for second in references:
            matrix.loc[first, second] = len(tic_sets[first] & tic_sets[second])
    print(matrix.to_string())

    print("\n== class agreement on shared stars (pairwise)")
    for first, second in itertools.combinations(references, 2):
        shared = sorted(tic_sets[first] & tic_sets[second])
        if not shared:
            continue
        first_rows = collapsed[collapsed["reference"] == first].set_index("TIC")
        second_rows = collapsed[collapsed["reference"] == second].set_index("TIC")
        exact = 0
        token_overlap = 0
        comparable = 0
        for tic in shared:
            class_first = first_rows.loc[tic, "class_paper"]
            class_second = second_rows.loc[tic, "class_paper"]
            if class_first == "" or class_second == "":
                continue
            comparable += 1
            if class_first == class_second:
                exact += 1
            tokens_first = set(str(first_rows.loc[tic, "class_tokens"]).split("|"))
            tokens_second = set(str(second_rows.loc[tic, "class_tokens"]).split("|"))
            if tokens_first & tokens_second:
                token_overlap += 1
        print(f"{first}  vs  {second}: shared={len(shared)} comparable={comparable} "
              f"exact={exact} token_overlap={token_overlap}")

    print("\n== stars with >= 2 independent references")
    counts = collapsed[collapsed["class_paper"] != ""].groupby("TIC")["reference"].nunique()
    for threshold in (2, 3, 4):
        print(f"  in >= {threshold} references: {(counts >= threshold).sum()}")

    multi_reference_tics = counts[counts >= 2].index
    fully_concordant = 0
    partially_concordant = 0
    discordant = 0
    for tic in multi_reference_tics:
        rows = collapsed[(collapsed["TIC"] == tic) & (collapsed["class_paper"] != "")]
        distinct_classes = set(rows["class_paper"])
        token_sets = [set(str(value).split("|")) for value in rows["class_tokens"]]
        common_tokens = set.intersection(*token_sets)
        if len(distinct_classes) == 1:
            fully_concordant += 1
        elif common_tokens:
            partially_concordant += 1
        else:
            discordant += 1
    print(f"  fully concordant (identical class_paper): {fully_concordant}")
    print(f"  partially concordant (share >=1 token):   {partially_concordant}")
    print(f"  discordant (no token in common):          {discordant}")

    print("\n== overlap with the user's samples")
    parent = pandas.read_csv(PARENT_SAMPLE, low_memory=False)
    clean = pandas.read_csv(CLEAN_SAMPLE, low_memory=False)
    review = pandas.read_csv(REVIEW_SAMPLE)
    golden_tics = set(golden["TIC"])
    for label, frame in (
        ("1_MassiveXTessV8.csv (parent OB x TIC v8)", parent),
        ("3_MassiveXTessV8_LC_DeltaMagnitude_Clean.csv (clean LC sample)", clean),
        ("results/clasificacion_review50.csv (pipeline review subset)", review),
    ):
        sample_tics = set(frame["TIC"])
        shared = golden_tics & sample_tics
        print(f"  {label}: N={len(sample_tics)}  shared with golden={len(shared)}")
        if len(shared) <= 60:
            print(f"    TIC: {sorted(shared)}")

    print("\n== golden class breakdown restricted to the clean sample")
    clean_tics = set(clean["TIC"])
    inside = collapsed[collapsed["TIC"].isin(clean_tics)]
    print(inside.groupby(["reference", "class_paper"]).size().to_string())


if __name__ == "__main__":
    main()
