# Golden sample: human-classified variability of TESS OB/B stars

Benchmark built from refereed papers that publish a **visual (human) variability
classification** of OB/B stars observed by TESS, keyed by **TIC**. It is meant
to be used as external ground truth against the classification pipeline of this
repository.

- Raw tables as downloaded: `catalogs/golden/`
- Unified catalogue: `catalogs/golden_sample.csv`
- Build script: `scripts/build_golden_sample.py`
- Statistics script: `scripts/analyse_golden_sample.py`

Built on 2026-09-03.

---

## 1. Sources

| # | Reference | Bibcode | N stars in golden sample | Where the table came from | Method |
|---|-----------|---------|--------------------------|---------------------------|--------|
| 1 | Balona et al. 2019, MNRAS 485, 3457 | `2019MNRAS.485.3457B` | 160 | VizieR `J/MNRAS/485/3457/table1` | visual |
| 2 | Pedersen et al. 2019, ApJL 872, L9 | `2019ApJ...872L...9P` | 154 | VizieR `J/ApJ/872/L9/table1` | visual + spectroscopy |
| 3 | Burssens et al. 2020, A&A 639, A81 | `2020A&A...639A..81B` | 98 | VizieR `J/A+A/639/A81` table2 x tablea1 | visual + spectroscopy (IACOB/OWN) |
| 4 | Bowman et al. 2020, A&A 640, A36 | `2020A&A...640A..36B` | 70 | VizieR `J/A+A/640/A36/tablea2` | SLF morphology fit (**class inferred**, see §4) |
| 5 | Labadie-Bartz et al. 2022, AJ 163, 226 | `2022AJ....163..226L` | 479 | VizieR `J/AJ/163/226/table2` | visual |
| 6 | Barraza et al. 2022, ApJ 924, 2 | `2022ApJ...924....2B` | 105 | arXiv:2202.01022 LaTeX source, Tables 2/3/4 | visual (FFT + Lomb-Scargle + wavelet) |
| 7 | Campelo et al. 2025, ApJ 989, 177 | `2025ApJ...989..177C` | 369 | AAS MRT (Tables 1/3/6) + article HTML (Tables 2/4/5/7) | visual (FFT + Lomb-Scargle + wavelet) |

**Author-name correction.** The task listed papers 6 and 7 as "Balona et al.
2022" and "Balona et al. 2025". They are in fact **Barraza et al. 2022** (ApJ
924, 2, DOI 10.3847/1538-4357/ac3335) and **Campelo et al. 2025** (ApJ 989, 177,
DOI 10.3847/1538-4357/adef2f), both from the De Medeiros group. Balona is not an
author on either. This matters for the concordance analysis below: Barraza+2022
is an **independent re-analysis of Balona+2019's own 160-star sample**, which is
exactly the disagreement test that was asked for.

**Pedersen et al. 2019 was added** (not in the original list). It is a refereed
TESS OB variability paper with a visual classification and TIC identifiers, and
it is the third independent classification of the same Balona 160-star sample,
so it strengthens the concordance test at no cost.

**Redundancy check requested for Balona+2019 vs the ApJ papers**: Barraza+2022
re-analyses exactly the same 160 stars (105 of them are individually tabulated;
overlap with Balona+2019 = 105/105). Campelo+2025 is a genuinely new 373-star
sample: it shares only **2 TIC** with Balona+2019.

### What could not be obtained

| Item | Status |
|------|--------|
| Barraza+2022 Table 1 (full 160-star list, MRT) | **Not downloaded.** No MRT is served by IOP for `10.3847/1538-4357/ac3335` and the arXiv LaTeX only prints 5 rows plus an ellipsis. No information lost: that table's star list is identical to Balona+2019 Table 1, which we do have. |
| Barraza+2022 the 56 stars in Tables 1 but not in Tables 2/3/4 | **Not published individually.** Only 104-105 of the 160 stars appear in a per-star table. The remaining ~55 carry no per-star Barraza label and are absent from the golden sample under that reference. |
| Campelo+2025 Tables 2, 4, 5, 7 | **No MRT exists** (only t1, t3 and t6 are served). Transcribed from the article HTML via web fetch. See the validation in §4 — every TIC checks out against the MRT of Table 1, and the six tables are mutually disjoint, so the transcription is verified, but it is not a first-party machine-readable file. |
| Bowman+2020 variability class | **Does not exist in the published tables.** Bowman+2020 publishes only a luminosity-class category (O/B dwarf, giant, ...) and the SLF fit parameters `alpha0, nu_char, gamma`. See §4. |

---

## 2. Controlled vocabulary

`class_raw` keeps the published label verbatim. `class_paper` is a single
normalised label; `class_tokens` is the pipe-separated set of *all* normalised
labels present in a compound published label (e.g. `SLF+SPB?` -> tokens
`PULS|SLF`).

| `class_paper` | Meaning | Published labels mapped onto it |
|---------------|---------|----------------------------------|
| `ROT` | rotational modulation | `rot`, `ROT`, `ACV`, `SXARI`, `MSRot` |
| `PULS` | coherent pulsation | `SPB`, `bCep`/`beta Cep`, `MAIA`, `ACYG`, `delta Sct`, `puls`, `IGW`, `hybrid` |
| `ECL` | eclipsing binary | `EB`, `EA`, `transit` |
| `ELL` | ellipsoidal variable | `ELL`, `EV` |
| `BE` | classical Be / emission-line variability | `Be`, `GCAS`, `Oe`, `Em`, `outburst` |
| `SLF` | stochastic low-frequency variability | `SLF` |
| `AMBIGUOUS` | author states the classification is ambiguous, instrumental or contaminated | `Ambiguous variability`, `instr`, `cont.`, `PQ` |
| `NOISY` | author states the light curve is noisy or constant | `Noisy`, `const` |
| `OTHER` | a real class outside the above | `sdB`, `HS` (hot subdwarf), `flare`, `SB1`/`SB2`, `Multi`, `Double` |
| *(empty)* | the paper publishes no class for that star | Balona+2019 `-` (17 stars) |

Normalisation rules, in order:

1. The raw label is split on `+ / , ; ( ) [ ] &`, so `EB` and `BE`, or `SPB` and
   `SB`, can never collide by substring matching.
2. Trailing `?` is stripped before lookup: an uncertain label keeps its physical
   class. The uncertainty is preserved verbatim in `class_raw`.
3. If the remark literally contains "noisy" or "ambiguous", **that is the
   verdict** and it overrides any physical class mentioned alongside it (this is
   what makes Barraza's "Ambiguous variability/Rotation" come out as
   `AMBIGUOUS`, not `ROT`).
4. Otherwise the primary class is the first match in the priority order
   `ECL > ELL > BE > ROT > PULS > SLF > OTHER > AMBIGUOUS > NOISY`.

Priority order 4 is a convention, not a physical statement. **For any serious
comparison use `class_tokens`, not `class_paper` alone** - e.g. Burssens's
`SLF+rot?` becomes `class_paper = ROT` but `class_tokens = ROT|SLF`.

---

## 3. Contents of `catalogs/golden_sample.csv`

Columns: `TIC, class_raw, class_paper, class_tokens, reference, method, period,
class_is_inferred, source_table, ra_deg, dec_deg`.

- **1436 rows**, one per (TIC, reference, source table).
- **1081 unique TIC.**
- 631 rows carry a `period` (days). Provenance of `period` by paper:
  Balona+2019 `1/nu_rot`; Burssens+2020 `1/nu_dom` (dominant frequency, *not*
  necessarily the classification period); Labadie-Bartz+2022 `1/f_g1`;
  Barraza+2022 and Campelo+2025 the published `P_rot`/`P_A`. Empty elsewhere.
- 695 unique TIC carry published coordinates (Barraza+2022 and Campelo+2025
  publish none; their stars only get coordinates if another paper covers them).

### Class distribution per paper

| Reference | *(empty)* | AMBIGUOUS | BE | ECL | ELL | NOISY | OTHER | PULS | ROT | SLF | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Balona+2019 | 17 | 0 | 9 | 16 | 4 | 0 | 1 | 56 | 57 | 0 | 160 |
| Pedersen+2019 | 0 | 3 | 24 | 9 | 5 | 9 | 1 | 20 | 57 | 26 | 154 |
| Burssens+2020 | 0 | 7 | 0 | 9 | 0 | 0 | 0 | 41 | 17 | 24 | 98 |
| Bowman+2020 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 70 | 70 |
| Labadie-Bartz+2022 | 0 | 0 | 430 | 0 | 0 | 0 | 49 | 0 | 0 | 0 | 479 |
| Barraza+2022 | 0 | 23 | 0 | 0 | 0 | 29 | 14 | 11 | 28 | 0 | 105 |
| Campelo+2025 | 0 | 33 | 0 | 7 | 0 | 215 | 48 | 36 | 30 | 0 | 369 |

---

## 4. Caveats that must be read before using this file

1. **Bowman+2020 classes are inferred, not published.** The paper's tables carry
   no variability label. Every one of its 70 stars is entered as `SLF` with
   `class_is_inferred = True`, on the grounds that the paper's premise is that
   these stars' low-frequency variability is fitted with the SLF morphology
   (`alpha0, nu_char, gamma` in `tablea2`). **Filter on
   `class_is_inferred == False` if you want only labels an author actually
   wrote.** All other 1366 rows have `class_is_inferred = False`.

2. **Campelo+2025 Tables 2, 4, 5 and 7 were transcribed from the article HTML**,
   because IOP serves no MRT for them. They were validated as follows and the
   check passed cleanly: all 30 + 36 + 7 + 33 TICs appear in the Table 1 MRT of
   373 stars; the four transcribed tables are mutually disjoint and disjoint
   from the two MRT tables (Table 3 noisy = 216, Table 6 sdB = 48); the six
   tables together account for 370 of the 373 stars with exactly one duplicate
   (TIC 455755305, listed both as an sdB candidate and as noisy). Still, these
   are transcriptions, not first-party files - treat them as slightly lower
   confidence than the VizieR/MRT rows.

3. **The Campelo+2025 Table 3 MRT has a wrong `Title:` line.** The file
   `campelo2025_table3_mrt.txt` says "ambiguous variability" in its header but
   its columns (TIC, CROWD, RUWE) and its 216 rows match the article's Table 3,
   "TESS B-type Stars with Noisy Behavior". It is treated here as the noisy
   table. The article's ambiguous table is Table 7 (33 stars, disjoint from
   these 216), which is what makes the identification unambiguous.

4. **Labadie-Bartz+2022 does not publish a variability class in the usual
   sense.** It publishes a Be-status flag (`Be` = Y/S/U/N) and a set of
   morphological signal letters (`signals` = S/I/L/H/V/F/G). Here `Be` in
   `{Y, S}` -> `BE` (430 stars) and `Be` in `{U, N}` -> `OTHER` (49 stars). The
   64 stars with quality flag `q = 0` (not observed, or unusable data) are
   dropped. The full raw string is kept in `class_raw`. This paper therefore
   contributes a Be/not-Be label, not a ROT/PULS/ECL label, and it dominates the
   `BE` bin of the whole golden sample.

5. **Balona+2019 leaves 17 of its 160 stars unclassified** (`VarType = -`).
   Those rows are kept with an empty `class_paper` rather than being guessed.

6. **Nothing was extrapolated across TIC versions.** All identifiers are the TIC
   numbers as published by each paper.

---

## 5. Overlap between papers

Number of shared TIC (diagonal = sample size):

| | Pedersen+2019 | Balona+2019 | Burssens+2020 | Bowman+2020 | Labadie-Bartz+2022 | Barraza+2022 | Campelo+2025 |
|---|---|---|---|---|---|---|---|
| **Pedersen+2019** | **154** | 152 | 1 | 1 | 11 | 100 | 2 |
| **Balona+2019** | 152 | **160** | 1 | 1 | 11 | 105 | 2 |
| **Burssens+2020** | 1 | 1 | **98** | 70 | 6 | 1 | 7 |
| **Bowman+2020** | 1 | 1 | 70 | **70** | 4 | 1 | 5 |
| **Labadie-Bartz+2022** | 11 | 11 | 6 | 4 | **479** | 9 | 1 |
| **Barraza+2022** | 100 | 105 | 1 | 1 | 9 | **105** | **2** |
| **Campelo+2025** | 2 | 2 | 7 | 5 | 1 | 2 | **369** |

The structure is two tight clusters plus one large near-disjoint catalogue:

- **Cluster A (the Balona 160):** Balona+2019, Pedersen+2019, Barraza+2022 -
  same stars, three independent classifications.
- **Cluster B (the IACOB/OWN OB stars):** Burssens+2020 and Bowman+2020 -
  Bowman's 70 stars are a subset of Burssens's 98.
- **Labadie-Bartz+2022 (479 Be stars) and Campelo+2025 (369 B stars)** are
  essentially disjoint from everything else and from each other.

### Stars with more than one reference

| | count |
|---|---|
| in >= 2 references | 226 |
| in >= 3 references | 98 |
| in >= 4 references | 12 |

---

## 6. Do independent authors agree?

This is the load-bearing result. "exact" = identical `class_paper`;
"token overlap" = the two compound labels share at least one normalised token
(the permissive criterion).

| Pair | shared | comparable | exact | token overlap |
|---|---|---|---|---|
| Pedersen+2019 vs Balona+2019 | 152 | 138 | 66 (48%) | 94 (68%) |
| Pedersen+2019 vs Barraza+2022 | 100 | 100 | 35 (35%) | 65 (65%) |
| Balona+2019 vs Barraza+2022 | 105 | 89 | 21 (24%) | 43 (48%) |
| Burssens+2020 vs Bowman+2020 | 70 | 70 | 20 (29%) | 51 (73%) |
| Pedersen+2019 vs Labadie-Bartz+2022 | 11 | 11 | 9 | 9 |
| Balona+2019 vs Labadie-Bartz+2022 | 11 | 10 | 5 | 6 |
| Burssens+2020 vs Campelo+2025 | 7 | 7 | 3 | 3 |
| Bowman+2020 vs Campelo+2025 | 5 | 5 | 0 | 0 |
| Burssens+2020 vs Labadie-Bartz+2022 | 6 | 6 | 0 | 0 |
| Bowman+2020 vs Labadie-Bartz+2022 | 4 | 4 | 0 | 0 |
| Labadie-Bartz+2022 vs Barraza+2022 | 9 | 9 | 0 | 0 |
| all remaining pairs | 1-2 | 1-2 | 0-2 | 0-2 |

Aggregated over the 226 stars that have >= 2 references:

| | count | fraction |
|---|---|---|
| fully concordant (identical `class_paper` across all references) | 61 | 27% |
| partially concordant (share >= 1 token, but not identical) | 51 | 23% |
| **discordant (no normalised token in common)** | **114** | **50%** |

### The rotation controversy, quantified

Balona+2019 (rows) versus Barraza+2022 (columns), on their 105 shared stars:

| Balona+2019 \ Barraza+2022 | AMBIGUOUS | NOISY | OTHER | PULS | ROT |
|---|---|---|---|---|---|
| *(no class)* | 2 | 12 | 2 | 0 | 0 |
| BE | 3 | 0 | 2 | 1 | 0 |
| ECL | 3 | 1 | 1 | 0 | 2 |
| ELL | 1 | 0 | 1 | 1 | 1 |
| OTHER | 0 | 1 | 0 | 0 | 0 |
| PULS | 7 | 2 | 3 | 0 | 4 |
| **ROT** | **7** | **13** | **5** | **9** | **21** |

Of the 55 stars Balona+2019 calls rotational variables, Barraza+2022's
independent re-analysis of the *same* light curves keeps only **21 (38%)** as
rotational; 13 become noisy, 9 become pulsators, 7 become ambiguous and 5 fall
into other classes. This is the quantitative form of the literature dispute over
Balona's rotational interpretation, and it is the single most important reason
not to treat any Balona-only `ROT` label as ground truth.

Balona+2019 (rows) versus Pedersen+2019 (columns) on their 152 shared stars is
milder but still weak: of Balona's 56 shared `ROT` stars, Pedersen keeps 36
(64%) as `rot`, calls 8 `Be`, 6 pulsators, 3 noisy/constant and 3 ambiguous.

Burssens+2020 versus Bowman+2020 (70 shared) looks like strong disagreement on
the exact criterion (20/70) but is really a granularity effect: Burssens's
labels are compounds like `SLF+SPB?`, so the token-overlap rate is 51/70 (73%)
and the residual disagreement is Burssens calling 33 of them primarily
pulsators. Note that this pair is *not* fully independent - Bowman's SLF label
is our inference (§4.1) and the two papers share authors and data.

### Practical recommendation for a high-confidence subset

The most defensible benchmark subset is:

- `class_is_inferred == False`, **and**
- `class_paper` not in `{AMBIGUOUS, NOISY}` and not empty, **and**
- at least two independent references agreeing on `class_paper` -> the **61
  fully concordant stars**, or the 112 stars that agree at token level.

Using single-reference labels (855 of the 1081 stars) is defensible for
Labadie-Bartz's Be flags and for Burssens's spectroscopically supported labels,
and clearly risky for Balona-only `ROT`.

---

## 7. Overlap with the user's samples

The candidate list of ~3000 OB stars with TIC is
`catalogs/3_MassiveXTessV8_LC_DeltaMagnitude_Clean.csv` (3578 unique TIC).
`catalogs/cds_A155/` (Monsalves+2025, A&A 704 A155) was inspected and **carries
no TIC column at all** - `candms.dat` and `g12par.dat` are keyed by Gaia DR3
source id - so it cannot be matched to the golden sample without a new
cross-match, and was not used.

| User sample | N | Shared with golden sample |
|---|---|---|
| `catalogs/1_MassiveXTessV8.csv` (parent, OB x TIC v8) | 13482 | **134** |
| `catalogs/3_MassiveXTessV8_LC_DeltaMagnitude_Clean.csv` (clean LC sample) | 3578 | **65** |
| `results/clasificacion_review50.csv` (pipeline review subset) | 44 | **1** |

The one review-subset star is **TIC 179639066**, classified `SLF` by
Pedersen+2019 and `ACYG` (-> `PULS`) by Balona+2019 - itself a discordant case.

**The low overlap is real, not an identifier artefact.** Cross-matching the 695
golden stars that have published coordinates against the parent sample within
3 arcsec returns 106 matches, and in all 106 the TIC identifiers agree; there
are zero positional matches with disagreeing TIC. So nothing is being lost to
TIC-version drift or bad identifiers - the samples genuinely barely intersect.
The reason is selection: the golden-sample papers target bright, mostly southern
B stars with 2-minute cadence, while `1_MassiveXTessV8` is a Gaia-selected G<12
massive-candidate list.

### The 65 golden stars inside the clean ~3578 sample

Of these 65: 46 have one reference, 18 have two, 1 has three.

```
10536200, 25883580, 29122883, 29207816, 30268695, 30653985, 31181554, 33287617,
35311214, 48217508, 53327951, 95513457, 129361218, 168627067, 179639066,
209829040, 215511795, 234009943, 234752466, 234834992, 234840662, 234947719,
253177102, 257065501, 270934336, 280835427, 285187855, 285916305, 292362364,
307100246, 309702035, 313955481, 314868712, 315036866, 315830698, 319936861,
325237845, 325273334, 329059929, 330281456, 337364158, 338640317, 338738989,
339565205, 339567904, 339570292, 340878553, 343376491, 381747495, 385626507,
389921913, 404967301, 405577964, 427395058, 447970564, 449459169, 455463415,
455809360, 465012898, 466353921, 466527712, 466715331, 466880435, 467065657,
467409163
```

Class breakdown of those 65 (one row per star per paper):

| Reference | classes |
|---|---|
| Pedersen+2019 | SLF 6 |
| Balona+2019 | PULS 6 |
| Burssens+2020 | PULS 6, SLF 4, ROT 3, AMBIGUOUS 3 |
| Bowman+2020 | SLF 13 (inferred) |
| Labadie-Bartz+2022 | BE 38, OTHER 3 |
| Campelo+2025 | AMBIGUOUS 2, ROT 1 |

**Bottom line for benchmarking:** the usable intersection is small. 65 stars
overlap the clean sample, and of those, 38 are Be flags from Labadie-Bartz and
13 are inferred SLF labels from Bowman. Only about a dozen carry a
spectroscopically supported, author-written ROT/PULS/ECL label inside the user's
own sample. If a larger benchmark is needed, the practical route is to *extend
the user's sample downwards to the golden-sample stars* (the 1081 golden TIC are
almost all bright B stars with TESS 2-min data) rather than to hunt for more
overlap.

---

## 8. Files in `catalogs/golden/`

| File | Content |
|---|---|
| `v_J_MNRAS_485_3457_table1.tsv` | Balona+2019 Table 1, 160 B stars |
| `v_J_ApJ_872_L9_table1.tsv` | Pedersen+2019 Table 1, 154 OB stars |
| `v_J_A_A_639_A81_table2.tsv` | Burssens+2020 Table 2, classes, no TIC |
| `v_J_A_A_639_A81_tablea1.tsv` | Burssens+2020 Table A1, supplies the TIC |
| `v_J_A_A_640_A36_tablea1.tsv` | Bowman+2020 Table A1, 70 OB stars + TIC |
| `v_J_A_A_640_A36_tablea2.tsv` | Bowman+2020 Table A2, SLF fit parameters |
| `v_J_AJ_163_226_table2.tsv` | Labadie-Bartz+2022 Table 2, 543 rows |
| `barraza2022_arxiv_source.tar.gz`, `barraza2022_src/arxiv.tex` | Barraza+2022 arXiv LaTeX source with Tables 2, 3, 4 |
| `campelo2025_table1_mrt.txt` | Campelo+2025 373-star sample (AAS MRT) |
| `campelo2025_table3_mrt.txt` | Campelo+2025 noisy stars, 216 (AAS MRT; header `Title:` line is wrong, see §4.3) |
| `campelo2025_table6_mrt.txt` | Campelo+2025 sdB candidates, 48 (AAS MRT) |
| `campelo2025_table2_rotation_from_article.csv` | Campelo+2025 Table 2, 30 rotators (transcribed) |
| `campelo2025_table4_pulsation_from_article.csv` | Campelo+2025 Table 4, 36 pulsators (transcribed) |
| `campelo2025_table5_binary_from_article.csv` | Campelo+2025 Table 5, 7 binaries (transcribed) |
| `campelo2025_table7_ambiguous_from_article.csv` | Campelo+2025 Table 7, 33 ambiguous (transcribed) |
| `ReadMe_*.txt` | The CDS ReadMe of each VizieR catalogue, with the byte-by-byte descriptions and the class-code notes |

## 9. Reproducing

```
cd /Users/bhianca/MassiveStarVariability
PYTHONPATH=src conda run --no-capture-output -n CNN_TESS python scripts/build_golden_sample.py
PYTHONPATH=src conda run --no-capture-output -n CNN_TESS python scripts/analyse_golden_sample.py
```
