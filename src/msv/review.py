"""Revisión visual estrella por estrella del benchmark VSX.

Los `Type`/`Period` de VSX son heterogéneos (survey-dependientes, a veces
heredados de literatura vieja): no son ground truth ciego. Este módulo arma
una tabla de revisión con contexto (período VSX, período detectado, clase
BRF), la muestra en un widget para marcarla a mano y guarda el veredicto en
CSV para pegarlo de vuelta a la tabla de información.

Flujo:

    from msv import review
    tab  = review.build_review_table(xm, preds)          # 567 estrellas
    lcs  = review.load_lc_cache()                        # curvas cacheadas
    rev  = review.VisualReviewer(tab, lcs); rev.show()   # marcar a mano
    xm2  = review.merge_review(xm)                       # columnas vis_*

El CSV (`catalogs/vsx_visual_review.csv`) se escribe en cada marca y se
relee al reconstruir la tabla: la revisión se reanuda donde quedó.
"""
import datetime as _dt
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cleaning import clean_lightcurve
from .config import CATALOGS_DIR, LC_PARQUET_MASSIVE, PERIODIC, RESULTS_DIR

# --- VSX -> clases del BRF --------------------------------------------------
# VSX usa tipos compuestos (`DSCT|GDOR|SXPHE`, `EA/DM`, `DCEP:`): se toma el
# token principal, sin flag `:` ni sufijos entre paréntesis. Los tipos de
# masivas sin equivalente (BCEP, ACYG, SPB, BE, GCAS...) quedan en None.
TYPE2CLASS_VSX = {
    "DCEP": "CEP", "DCEPS": "CEP", "CEP": "CEP", "CWA": "CEP", "CWB": "CEP",
    "CW": "CEP", "RVA": "CEP",
    "DSCT": "DST", "HADS": "DST", "SXPHE": "DST",
    "RRAB": "RR", "RRC": "RR", "RRD": "RR", "RR": "RR",
    "M": "M",
    "EA": "E", "EB": "E", "EW": "E", "E": "E", "EP": "E",
    "ESD": "E", "ED": "E", "EC": "E",   # subtipos eclipsantes GCVS-style
    "ELL": "ELL",
    "SR": "LPV", "SRA": "LPV", "SRB": "LPV", "SRC": "LPV", "SRD": "LPV",
    "SRS": "LPV", "L": "LPV", "LB": "LPV", "LC": "LPV",
}

# Umbral de detectabilidad en un sector TESS: P < 13 d -> label=1.
LABEL_PER_MAX = 13.0


def vsx_main_token(t):
    """Token principal de un `Type` de VSX, en mayúsculas y sin flags."""
    if not isinstance(t, str) or not t.strip():
        return None
    tok = re.split(r"[|+/]", t.strip())[0]
    return re.sub(r"\(.*\)$", "", tok).rstrip(":").upper()


def load_vsx_xmatch(path=None, per_max=LABEL_PER_MAX):
    """Cruce masivas×VSX con `per_vsx`, `label`, `vsx_token`, `expected_class`."""
    path = path or CATALOGS_DIR / "1_MassiveXVSX.csv"
    xm = pd.read_csv(path)
    xm["per_vsx"] = xm["Period"]
    xm["label"] = np.where(xm["per_vsx"] < per_max, 1,
                           np.where(xm["per_vsx"] >= per_max, 0, np.nan))
    xm["vsx_token"] = xm["Type"].map(vsx_main_token)
    xm["expected_class"] = xm["vsx_token"].map(TYPE2CLASS_VSX)
    return xm

# --- Vocabulario del veredicto ---------------------------------------------
# Período: ¿el `per_vsx` pliega bien la curva TESS?
VIS_VERDICTS = ("ok", "maybe", "bad", "sin_lc")
# Morfología: qué se ve en la curva plegada, independiente de lo que diga VSX.
VIS_SHAPES = ("EB", "sinusoidal", "pulsante", "multi", "irregular", "ruido", "?")
# Traducción morfología -> clases del BRF, para cruzar con `expected_class`.
SHAPE2CLASS = {"EB": {"E"}, "sinusoidal": {"ELL", "E"},
               "pulsante": {"CEP", "DST", "RR", "M"}, "multi": {"CEP", "DST"},
               "irregular": {"LPV", "Rndm"}, "ruido": {"Rndm"}}

VIS_COLS = ["vis_verdict", "vis_shape", "vis_notes", "vis_date"]
REVIEW_CSV = CATALOGS_DIR / "vsx_visual_review.csv"
LC_CACHE = RESULTS_DIR / "lc_vsx_review.parquet"

HARMONICS = (1.0, 2.0, 0.5, 3.0, 1 / 3)


# --- Tabla de revisión ------------------------------------------------------
def _per_detected(preds, tol=0.05):
    """Por TIC: período representativo detectado y si alguno matchea per_vsx.

    Se elige el peak con match de período (mayor `brf_prob`); si ninguno
    matchea, el peak periódico más probable; si tampoco hay, el de mayor
    prominencia. Es el período que se grafica al lado del de VSX.
    """
    df = preds.copy()
    if "period_match" not in df.columns:
        ratio = (df["per"].to_numpy()[:, None]
                 / (df["per_vsx"].to_numpy()[:, None] * np.array(HARMONICS)[None, :]))
        df["period_match"] = np.abs(ratio - 1).min(axis=1) <= tol
    df["is_periodic"] = df["brf_class"].isin(PERIODIC)
    # prioridad: match+periódico > match > periódico > resto
    rank = (df["period_match"].astype(int) * 2 + df["is_periodic"].astype(int))
    df = df.assign(_rank=rank).sort_values(
        ["_rank", "brf_prob", "prominence"], ascending=False)
    best = df.groupby("TIC").first()
    return pd.DataFrame({
        "per_det": best["per"],
        "det_source": best["source"],
        "det_sector": best["sector"],
        "brf_class": best["brf_class"],
        "brf_prob": best["brf_prob"],
        "instability": best["instability"],
        "period_match": df.groupby("TIC")["period_match"].any(),
        "n_peaks": df.groupby("TIC").size(),
    })


def build_review_table(xm, preds=None, review_csv=REVIEW_CSV, tol=0.05, save=True):
    """Tabla de estrellas a revisar (label 0/1) con contexto y veredictos.

    `xm` es el cruce masivas×VSX con `per_vsx`, `label`, `vsx_token` y
    `expected_class`; `preds` el CSV de predicciones por peak (ya mergeado
    con `per_vsx`). Si `review_csv` existe, sus columnas `vis_*` se
    reinyectan — nunca se pisan veredictos ya puestos.
    """
    cols = [c for c in ["TIC", "Type", "per_vsx", "label", "vsx_token",
                        "expected_class"] if c in xm.columns]
    tab = xm.loc[xm["label"].notna(), cols].copy()
    tab["label"] = tab["label"].astype(int)

    if preds is not None:
        tab = tab.merge(_per_detected(preds, tol=tol), on="TIC", how="left")
    for c in ("per_det", "brf_class", "period_match"):
        tab[c] = tab.get(c, np.nan)

    # Grupos de prioridad: primero lo que puede romper las conclusiones.
    if preds is None:
        tab["grupo"] = np.where(tab["label"] == 0, "FP_label0", "acuerdo")
    else:
        mism = (tab["expected_class"].notna() & tab["brf_class"].notna()
                & (tab["brf_class"] != tab["expected_class"]))
        tab["grupo"] = np.select(
            [tab["label"] == 0,
             (tab["label"] == 1) & (tab["period_match"] != True),  # noqa: E712
             (tab["label"] == 1) & mism],
            ["FP_label0", "sin_recovery", "clase_discrepante"],
            default="acuerdo")

    for c in VIS_COLS:
        tab[c] = ""
    if review_csv is not None and review_csv.exists():
        old = pd.read_csv(review_csv, usecols=lambda c: c in ["TIC"] + VIS_COLS)
        tab = tab.drop(columns=VIS_COLS).merge(old, on="TIC", how="left")
        for c in VIS_COLS:
            tab[c] = tab[c].fillna("").astype(str)

    order = ["FP_label0", "sin_recovery", "clase_discrepante", "acuerdo"]
    tab["grupo"] = pd.Categorical(tab["grupo"], order, ordered=True)
    tab = tab.sort_values(["grupo", "TIC"]).reset_index(drop=True)
    if save and review_csv is not None:
        save_review(tab, review_csv)
    return tab


def save_review(tab, review_csv=REVIEW_CSV):
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(review_csv, index=False)
    return review_csv


def review_summary(tab_or_csv=REVIEW_CSV):
    """Conteo de veredictos por grupo de prioridad."""
    tab = (pd.read_csv(tab_or_csv) if not isinstance(tab_or_csv, pd.DataFrame)
           else tab_or_csv)
    v = tab["vis_verdict"].fillna("").replace("", "sin_revisar")
    return pd.crosstab(tab["grupo"], v, margins=True)


def merge_review(df, review_csv=REVIEW_CSV, how="left"):
    """Pega las columnas `vis_*` (+ `grupo`) a cualquier tabla con `TIC`."""
    rev = pd.read_csv(review_csv)
    keep = ["TIC", "grupo"] + [c for c in VIS_COLS if c in rev.columns]
    out = df.merge(rev[keep], on="TIC", how=how)
    out["vis_verdict"] = out["vis_verdict"].fillna("").replace("", "sin_revisar")
    for c in ("vis_shape", "vis_notes"):
        if c in out.columns:
            out[c] = out[c].fillna("")
    return out


def shape_matches_class(shape, expected_class):
    """¿La morfología vista a ojo es compatible con `expected_class` de VSX?"""
    if not isinstance(shape, str) or shape in ("", "?") or pd.isna(expected_class):
        return np.nan
    return expected_class in SHAPE2CLASS.get(shape, set())


# --- Curvas de luz ----------------------------------------------------------
def build_lc_cache(tics, lc_parquet=LC_PARQUET_MASSIVE, out=LC_CACHE,
                   clean=True, chunk=100):
    """Cachea las curvas de `tics` (limpias y normalizadas) en un parquet.

    Evita filtrar el parquet de 73M filas una vez por estrella durante la
    revisión. El flujo se normaliza por (TIC, sector) a mediana 1 para poder
    superponer sectores en la curva plegada.
    """
    tics = [int(t) for t in pd.unique(pd.Series(tics))]
    parts = []
    for i in range(0, len(tics), chunk):
        blk = tics[i:i + chunk]
        df = pd.read_parquet(lc_parquet, columns=["TIC", "sector", "Time", "flux"],
                             filters=[("TIC", "in", blk)])
        if df.empty:
            continue
        if clean:
            rows = []
            for (tic, sec), g in df.groupby(["TIC", "sector"], sort=False):
                t, f = clean_lightcurve(g["Time"].to_numpy(), g["flux"].to_numpy())
                if len(t) == 0:
                    continue
                rows.append(pd.DataFrame({"TIC": tic, "sector": sec,
                                          "Time": t, "flux": f}))
            df = pd.concat(rows, ignore_index=True) if rows else df.iloc[:0]
        med = df.groupby(["TIC", "sector"])["flux"].transform("median")
        df["flux"] = df["flux"] / med
        parts.append(df)
        print(f"  cache {min(i + chunk, len(tics))}/{len(tics)} TICs", end="\r")
    cache = pd.concat(parts, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(out, index=False)
    print(f"\n{out}: {len(cache)} puntos, {cache['TIC'].nunique()} TICs")
    return out


def load_lc_cache(path=LC_CACHE):
    """dict TIC -> DataFrame(Time, flux, sector) desde el parquet cacheado."""
    df = pd.read_parquet(path)
    return {int(t): g.sort_values("Time").reset_index(drop=True)
            for t, g in df.groupby("TIC")}


# --- Revisor interactivo ----------------------------------------------------
def plot_star(row, lc, axes=None, phase_bins=40, sector=None):
    """Serie temporal + fold a per_vsx + fold al período detectado.

    `sector` restringe el plegado a un sector: con sectores separados por
    cientos de días, un error minúsculo de período basta para descoherentar
    la fase entre ellos y ensuciar la curva plegada.
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(16, 3.6))
    tic = int(row["TIC"])
    if lc is None or lc.empty:
        for ax in axes:
            ax.axis("off")
        axes[0].set_title(f"TIC {tic}: sin curva de luz", fontsize=10)
        return axes

    t = lc["Time"].to_numpy()
    f = lc["flux"].to_numpy()
    sec = lc["sector"].to_numpy()
    secs = np.unique(sec)

    # un color fijo por sector, el mismo en la serie y en las plegadas
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {s: cyc[k % len(cyc)] for k, s in enumerate(secs)}

    ax = axes[0]
    for s in secs:
        q = sec == s
        ax.plot(t[q], f[q], ".", ms=2, alpha=0.6, color=colors[s], label=f"s{int(s)}")
    ax.legend(fontsize=6, markerscale=3, ncol=2, loc="best")
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("flux / mediana")
    ax.set_title(f"TIC {tic} | VSX: {row.get('Type', '?')} | {len(secs)} sectores",
                 fontsize=10)
    ax.grid(alpha=0.3)

    # los folds usan solo el sector pedido (los sectores lejanos descoherentan)
    keep = np.ones(len(t), bool) if sector is None else (sec == int(sector))
    tf, ff, secf = t[keep], f[keep], sec[keep]
    if sector is not None:
        axes[0].axvspan(tf.min(), tf.max(), color="tab:orange", alpha=0.12, zorder=0)

    for ax, per, tag in ((axes[1], row.get("per_vsx"), "P_VSX"),
                         (axes[2], row.get("per_det"), "P_detectado")):
        if per is None or not np.isfinite(per) or per <= 0 or len(tf) == 0:
            ax.axis("off")
            ax.set_title(f"{tag}: no disponible", fontsize=10)
            continue
        ph = np.mod(tf, per) / per
        for s in np.unique(secf):
            q = secf == s
            ax.plot(np.r_[ph[q], ph[q] + 1], np.r_[ff[q], ff[q]], ".", ms=2,
                    alpha=0.5, color=colors[s])
        # mediana en bins de fase: la señal coherente salta a la vista
        b = np.linspace(0, 1, phase_bins + 1)
        idx = np.clip(np.digitize(ph, b) - 1, 0, phase_bins - 1)
        med = pd.Series(ff).groupby(idx).median()
        xc = 0.5 * (b[:-1] + b[1:])
        ax.plot(np.r_[xc[med.index], xc[med.index] + 1], np.r_[med, med], "-",
                color="k", lw=1.6, zorder=5)
        extra = (f" [{row.get('det_source', '')} {row.get('brf_class', '')}]"
                 if tag == "P_detectado" else "")
        sec_lab = "todos los sectores" if sector is None else f"sector {int(sector)}"
        ax.set_xlabel("fase (0->2)")
        ax.set_title(f"{tag} = {per:.5f} d{extra}\n{sec_lab}", fontsize=10)
        ax.grid(alpha=0.3)
    return axes


class VisualReviewer:
    """Widget de revisión: dibuja una estrella y guarda el veredicto al vuelo.

    Botones de período (OK/Dudoso/Mal/sin LC) marcan y **avanzan solo**; la
    morfología y las notas se aplican a la estrella en pantalla. Cada cambio
    se escribe en `review_csv`, así que cerrar el notebook no pierde nada.
    """

    def __init__(self, tab, lcs, review_csv=REVIEW_CSV, grupos=None,
                 solo_pendientes=False, figsize=(16, 3.6)):
        import ipywidgets as widgets  # import perezoso: el módulo sirve sin él

        self.tab = tab
        self.lcs = lcs
        self.csv = review_csv
        self.figsize = figsize

        mask = pd.Series(True, index=tab.index)
        if grupos is not None:
            mask &= tab["grupo"].isin(np.atleast_1d(grupos))
        if solo_pendientes:
            mask &= tab["vis_verdict"].fillna("").eq("")
        self.idx = list(tab.index[mask])
        if not self.idx:
            raise ValueError("No quedan estrellas que cumplan el filtro")
        self.i = 0

        self.out = widgets.Output()
        self.info = widgets.HTML()
        b = lambda d, st, ic="": widgets.Button(description=d, button_style=st,  # noqa: E731
                                                icon=ic, layout=widgets.Layout(width="110px"))
        self.b_ok, self.b_maybe = b("OK", "success", "check"), b("Dudoso", "warning")
        self.b_bad, self.b_nolc = b("Mal", "danger", "times"), b("sin LC", "")
        self.b_prev, self.b_next = b("< Prev", "info"), b("Next >", "info")
        self.b_pend = widgets.Button(description="siguiente sin revisar",
                                     layout=widgets.Layout(width="180px"))
        self.shape = widgets.ToggleButtons(options=("",) + VIS_SHAPES,
                                           description="forma:", value="")
        self.notes = widgets.Text(description="notas:", placeholder="opcional",
                                  continuous_update=False,
                                  layout=widgets.Layout(width="60%"))
        self.jump = widgets.IntText(value=1, layout=widgets.Layout(width="80px"))
        self.sector = widgets.Dropdown(description="fold:", options=[("todos", None)],
                                       layout=widgets.Layout(width="220px"))
        self._quiet = False   # evita reentrar en _plot al reponer las opciones

        self.b_ok.on_click(lambda _: self._mark("ok"))
        self.b_maybe.on_click(lambda _: self._mark("maybe"))
        self.b_bad.on_click(lambda _: self._mark("bad"))
        self.b_nolc.on_click(lambda _: self._mark("sin_lc"))
        self.b_prev.on_click(lambda _: self._go(-1))
        self.b_next.on_click(lambda _: self._go(+1))
        self.b_pend.on_click(lambda _: self._goto_pending())
        self.shape.observe(self._on_shape, names="value")
        self.notes.observe(self._on_notes, names="value")   # dispara al dar Enter
        self.jump.observe(self._on_jump, names="value")
        self.sector.observe(self._on_sector, names="value")

        self.box = widgets.VBox([
            self.info,
            widgets.HBox([self.b_prev, self.b_ok, self.b_maybe, self.b_bad,
                          self.b_nolc, self.b_next, self.b_pend,
                          widgets.Label("ir a #"), self.jump]),
            widgets.HBox([self.shape, self.sector]), self.notes, self.out,
        ])

    # -- estado --------------------------------------------------------------
    @property
    def row(self):
        return self.tab.loc[self.idx[self.i]]

    def _set(self, col, val, redraw=True):
        self.tab.loc[self.idx[self.i], col] = val
        self.tab.loc[self.idx[self.i], "vis_date"] = _dt.date.today().isoformat()
        save_review(self.tab, self.csv)
        if redraw:
            self._draw()

    def _mark(self, verdict):
        self.tab.loc[self.idx[self.i], "vis_verdict"] = verdict
        if self.notes.value:
            self.tab.loc[self.idx[self.i], "vis_notes"] = self.notes.value
        self.tab.loc[self.idx[self.i], "vis_date"] = _dt.date.today().isoformat()
        save_review(self.tab, self.csv)
        if self.i < len(self.idx) - 1:
            self.i += 1
        self._draw()

    def _on_shape(self, change):
        # sin redibujar: cambiar la forma no cambia los paneles
        if change["new"] != str(self.row.get("vis_shape") or ""):
            self._set("vis_shape", change["new"], redraw=False)
            self._refresh_info()

    def _on_notes(self, change):
        if change["new"] != str(self.row.get("vis_notes") or ""):
            self._set("vis_notes", change["new"], redraw=False)

    def _go(self, d):
        self.i = int(np.clip(self.i + d, 0, len(self.idx) - 1))
        self._draw()

    def _on_jump(self, change):
        n = int(np.clip(change["new"], 1, len(self.idx))) - 1
        if n != self.i:
            self.i = n
            self._draw()

    def _goto_pending(self):
        pend = [k for k in range(len(self.idx))
                if not str(self.tab.loc[self.idx[k], "vis_verdict"] or "").strip()]
        nxt = next((k for k in pend if k > self.i), pend[0] if pend else None)
        if nxt is None:
            with self.out:
                print("No quedan estrellas sin revisar en este filtro.")
            return
        self.i = nxt
        self._draw()

    # -- dibujo --------------------------------------------------------------
    def _refresh_info(self):
        r = self.row
        done = int((self.tab.loc[self.idx, "vis_verdict"].fillna("") != "").sum())
        self.info.value = (
            f"<b>#{self.i + 1}/{len(self.idx)}</b> &nbsp; TIC {int(r['TIC'])} "
            f"&nbsp;|&nbsp; grupo <b>{r['grupo']}</b> &nbsp;|&nbsp; label {r['label']} "
            f"&nbsp;|&nbsp; VSX <b>{r.get('Type', '?')}</b> "
            f"(esperado: {r.get('expected_class') or '—'}) "
            f"&nbsp;|&nbsp; BRF <b>{r.get('brf_class') or '—'}</b> "
            f"&nbsp;|&nbsp; veredicto: <b>{r['vis_verdict'] or '—'}</b> / "
            f"forma: <b>{r['vis_shape'] or '—'}</b> "
            f"&nbsp;|&nbsp; revisadas {done}/{len(self.idx)}")

    def _on_sector(self, change):
        if not self._quiet:
            self._plot()

    def _sector_options(self, lc):
        """('todos', None) + un ítem por sector; default = el más poblado.

        Con un solo sector no hay nada que elegir; con varios conviene
        plegar de a uno, porque la fase entre sectores lejanos se pierde si
        el período está apenas corrido.
        """
        if lc is None or lc.empty:
            return [("todos", None)], None
        n = lc.groupby("sector").size().sort_index()
        opts = [("todos", None)] + [(f"s{int(s)} ({v} pts)", int(s))
                                    for s, v in n.items()]
        default = None if len(n) == 1 else int(n.idxmax())
        return opts, default

    def _plot(self):
        r = self.row
        with self.out:
            self.out.clear_output(wait=True)
            _, axes = plt.subplots(1, 3, figsize=self.figsize)
            plot_star(r, self.lcs.get(int(r["TIC"])), axes=axes,
                      sector=self.sector.value)
            plt.tight_layout()
            plt.show()

    def _draw(self):
        r = self.row
        self._refresh_info()
        self._quiet = True
        self.sector.options, self.sector.value = self._sector_options(
            self.lcs.get(int(r["TIC"])))
        self._quiet = False
        self._plot()
        self.shape.unobserve(self._on_shape, names="value")
        self.shape.value = str(r.get("vis_shape") or "")
        self.shape.observe(self._on_shape, names="value")
        self.notes.unobserve(self._on_notes, names="value")
        self.notes.value = str(r.get("vis_notes") or "")
        self.notes.observe(self._on_notes, names="value")
        self.jump.value = self.i + 1

    def show(self):
        from IPython.display import display
        display(self.box)
        self._draw()
        return self
