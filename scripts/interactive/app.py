"""Banco interactivo: nuestros periodogramas, clase por pico, whitening a mano.

El lazo que implementa: correr LS y ACF como los corre `run_peaks.py`,
seleccionar picos con `select_peaks`, colapsar el peine con
`candidate_periods`, clasificar CADA candidato con la CNN, y blanquear de la
curva los que salen `Rndm` o `LPV` — o los que el ojo diga — para volver a
correr el periodograma sobre lo que queda.

El motivo es que la CNN fue entrenada con estrellas cuya curva ES una sola
variación: en una multiperiódica el fold en cualquier período arrastra las
otras como dispersión, y la red lee esa dispersión como LPV. Sacar lo que ya
se juzgó incoherente deja ver lo que había debajo.

Corre en el env que tiene msv; la CNN vive en otro proceso (`cnn_client.py`):

    MSV_WEIGHTS=$HOME/ViT_VariableStars/pretrained/keras_checkpoints \
    PYTHONPATH=src /opt/anaconda3/envs/CNN_TESS/bin/python -m streamlit run \
    scripts/interactive/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from cnn_client import PredictServer  # noqa: E402
from msv import cleaning, config, features, io, peaks, periodograms, prewhiten  # noqa: E402

GOLDEN_DIR = Path.home() / "Dropbox/MassiveStarVariability/raw/golden"
VERDICTS = REPO_ROOT / "results/golden/veredicto_whitening.csv"
DEFAULT_TIC = 337886863
# Published periods, drawn on the periodogram so a candidate can be compared
# against what the papers report without leaving the app.
REFERENCE_PERIODS = {
    337886863: [("Campelo P", 2.754, "#d62728")],
    464295672: [("Burssens P_dom", 1.721170, "#d62728")],
}
# The classes this loop treats as "no coherent variation here". They are the
# default whitening selection, not a rule: the checkbox is the point.
INCOHERENT = ("Rndm", "LPV")
# Adopted floor: one TESS sector is ~26.5 d, so a 10 d period is 2.6 cycles and
# the CNN never saw a fold that short in training (OGLE baselines are years).
MIN_CYCLES = 4.0
HARMONIC_MAX_ORDER = 8
CLASS_COLORS = {"ELL": "#d62728", "E": "#ff7f0e", "LPV": "#7f7f7f",
                "Rndm": "#bbbbbb", "CEP": "#2ca02c", "DST": "#9467bd",
                "RR": "#8c564b", "M": "#17becf"}
KIND_SYMBOLS = {"fundamental": "star", "subarmonico": "triangle-down",
                "aislado": "circle"}


@st.cache_data(show_spinner=False)
def load_sectors(tic):
    paths = sorted(GOLDEN_DIR.glob(f"*{int(tic):016d}-s*_lc.fits"))
    return {io.parse_fits_name(path)[1]: str(path) for path in paths}


@st.cache_data(show_spinner="Leyendo y limpiando la curva...")
def load_light_curve(paths):
    """Curva concatenada y limpia, en las unidades de flujo originales.

    Cada sector se lleva a su media 1.0 antes de concatenar: el nivel de flujo
    de TESS cambia entre sectores con la apertura y el crowding, y unir cuentas
    crudas mete un escalon que el periodograma lee como una variacion larga.
    """
    times, fluxes, errors = [], [], []
    for path in paths:
        frame = io.read_lc_fits(path)
        time, flux, error = cleaning.clean_lightcurve(
            frame["Time"].values, frame["flux"].values, frame["flux_err"].values)
        mean_flux = float(np.mean(flux))
        times.append(time)
        fluxes.append(flux / mean_flux)
        errors.append(error / mean_flux)
    time = np.concatenate(times)
    order = np.argsort(time)
    return (time[order], np.concatenate(fluxes)[order],
            np.concatenate(errors)[order])


@st.cache_data(show_spinner="Periodogramas LS + ACF...")
def run_periodograms(time, flux, error, sources, top_n, min_peak_sep_days):
    """Exactamente el camino de `run_peaks.py`: periodograma, picos, candidatos."""
    grids, all_peaks = {}, []
    if "LS" in sources:
        grids["LS"] = periodograms.ls_periodogram(time, flux, error)
    if "ACF" in sources:
        grids["ACF"] = periodograms.acf_periodogram(time, flux, error)
    for source, grid in grids.items():
        found = peaks.select_peaks(grid, source, top_n=top_n,
                                   min_peak_sep_days=min_peak_sep_days)
        if not found.empty:
            all_peaks.append(candidates_of(found).assign(source=source))
    # `DataFrame.attrs` does not survive Streamlit's cache round-trip, so the
    # scalars the plot needs are pulled out here, while the frames are fresh.
    levels = {source: grid.attrs.get("fap_level")
              for source, grid in grids.items()}
    if not all_peaks:
        return grids, levels, pd.DataFrame(columns=["per", "period", "kind",
                                                    "source", "power"])
    return grids, levels, pd.concat(all_peaks, ignore_index=True)


def candidates_of(found):
    """`candidate_periods` sobre los picos de una fuente."""
    return peaks.candidate_periods(found.reset_index(drop=True))


@st.cache_resource(show_spinner="Cargando la CNN (una sola vez)...")
def predict_server(checkpoint):
    return PredictServer(checkpoint)


def classify_candidates(checkpoint, time, flux, candidates):
    """Un phase-fold 32x32 por candidato -> clase y probabilidades."""
    if candidates.empty:
        return candidates
    images = np.stack([features.phase_fold_hist2d_log(time, flux, period)
                       for period in candidates["period"]])
    probabilities = predict_server(checkpoint).predict(images)
    table = candidates.copy()
    for position, name in enumerate(config.CLASS_NAMES):
        table[f"p_{name}"] = probabilities[:, position]
    table["clase"] = [config.CLASS_NAMES[index]
                      for index in probabilities.argmax(axis=1)]
    table["p_clase"] = probabilities.max(axis=1)
    table["log_pLPV"] = -np.log10(
        np.clip(probabilities[:, config.CLASS_NAMES.index("LPV")], 1e-300, None))
    # Nobody in the pipeline asks whether the frequency has any power; the
    # class is read off a fold that exists whether or not there is a signal.
    # An empty component list makes `probe` a bare fit at that frequency.
    table["snr"] = [prewhiten.probe(time, flux, [], 1.0 / period)["snr"]
                    for period in candidates["period"]]
    table["ciclos"] = (time.max() - time.min()) / candidates["period"]
    table["parte_de"] = harmonic_owner(table)
    return table


def harmonic_owner(table):
    """Para cada candidato, el período coherente del que es armónico.

    La potencia de una eclipsante se concentra en los armónicos, no en el
    orbital, asi que sus P/2, P/3, P/4 aparecen como candidatos propios y la
    red, que ve cada fold por separado, los llama LPV o Rndm. Blanquear uno de
    esos no saca una variacion: le arranca un pedazo al eclipse, y lo que queda
    es una onda con dos maximos por ciclo — o sea una ELL. Esta columna es lo
    que impide que el lazo se coma su propia senal.
    """
    coherent = table[~table["clase"].isin(INCOHERENT)]
    owners = []
    for _, row in table.iterrows():
        owner = ""
        for _, other in coherent.iterrows():
            if other["period"] == row["period"]:
                continue
            # max_order 8, no el 4 por defecto de msv: el peine de una
            # eclipsante llega hondo, y con 4 un P/6 no se reconoce como
            # armonico y el lazo se lo come.
            if prewhiten.commensurate(1.0 / row["period"], 1.0 / other["period"],
                                      max_order=HARMONIC_MAX_ORDER):
                owner = f"{other['period']:.4f} ({other['clase']})"
                break
        owners.append(owner)
    return owners


def whiten(time, flux, periods, n_harmonics):
    """Resta el modelo de fundamental + armonicos en cada periodo dado.

    El ajuste es lineal, asi que corre en las unidades de flujo que reciba y
    devuelve la curva en esas mismas — la que despues vuelve al periodograma.
    """
    residual = np.asarray(flux, float).copy()
    for period in periods:
        model, _ = prewhiten.fit_harmonic_model(
            time, residual - np.mean(residual), 1.0 / period, n_harmonics)
        residual = residual - model
    return residual


def periodogram_figure(grids, levels, candidates, selected_period,
                       reference_periods, whitened_periods):
    figure = go.Figure()
    colors = {"LS": "#333333", "ACF": "#8c8c8c"}
    for source, grid in grids.items():
        figure.add_trace(go.Scatter(
            x=grid["per"], y=grid["power"], mode="lines", name=source,
            line=dict(color=colors[source], width=1),
            hovertemplate="P=%{x:.5f} d<br>power=%{y:.4f}<extra>" + source + "</extra>"))
        if source == "LS" and "window" in grid:
            figure.add_trace(go.Scatter(
                x=grid["per"], y=grid["window"], mode="lines", name="ventana LS",
                line=dict(color="#c49a6c", width=1, dash="dot")))
            level = levels.get("LS")
            if level is not None:
                figure.add_hline(y=float(level), line_width=1, line_dash="dash",
                                 line_color="#c49a6c",
                                 annotation_text="FAP LS", annotation_font_size=10)
        if source == "ACF" and "fap" in grid:
            figure.add_trace(go.Scatter(
                x=grid["per"], y=grid["fap"], mode="lines", name="FAP ACF",
                line=dict(color="#8c8c8c", width=1, dash="dash")))

    for clase, group in candidates.groupby("clase") if not candidates.empty else []:
        figure.add_trace(go.Scatter(
            x=group["period"], y=group["power"], mode="markers",
            name=f"{clase} ({len(group)})",
            marker=dict(
                size=14, color=CLASS_COLORS.get(clase, "#cccccc"),
                symbol=[KIND_SYMBOLS.get(kind, "circle") for kind in group["kind"]],
                line=dict(width=1.5, color="black")),
            customdata=group[["period", "kind", "source", "p_clase", "log_pLPV"]],
            hovertemplate=("P=%{customdata[0]:.5f} d<br>%{customdata[1]} / "
                           "%{customdata[2]}<br>" + clase +
                           " p=%{customdata[3]:.3f}<br>"
                           "-log10 p_LPV=%{customdata[4]:.2f}<extra></extra>")))

    marks = [(name, period, color, 2)
             for name, period, color in reference_periods]
    marks += [("blanqueado", period, "#1f77b4", 1)
              for period in whitened_periods]
    if selected_period:
        marks.append(("sel", selected_period, "#2ca02c", 2))
    for position, (name, period, color, width) in enumerate(marks):
        figure.add_vline(x=period, line_width=width, line_dash="dash",
                         line_color=color, annotation_text=name,
                         annotation_font_size=10, annotation_font_color=color,
                         annotation_yshift=-14 * (position % 3))

    figure.update_layout(
        height=430, margin=dict(l=55, r=20, t=50, b=45),
        xaxis_title="periodo (d)", yaxis_title="power",
        xaxis_type="log", clickmode="event", dragmode="zoom",
        hovermode="closest", legend=dict(orientation="h", y=1.14))
    return figure


def light_curve_figure(time, original, current):
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=time, y=original, mode="markers", name="original",
        marker=dict(size=2, color="#cccccc")))
    figure.add_trace(go.Scatter(
        x=time, y=current, mode="markers", name="blanqueada",
        marker=dict(size=2, color="black")))
    figure.update_layout(height=230, margin=dict(l=55, r=20, t=30, b=40),
                         xaxis_title="tiempo (d)", yaxis_title="flujo relativo",
                         legend=dict(orientation="h", y=1.2))
    return figure


def fold_figure(time, flux, period, title):
    phase = np.mod(time, period) / period
    figure = go.Figure(go.Scatter(
        x=np.concatenate([phase, phase + 1.0]),
        y=np.concatenate([flux, flux]), mode="markers",
        marker=dict(size=2.5, color="black", opacity=0.45), showlegend=False))
    figure.update_layout(height=250, margin=dict(l=55, r=10, t=34, b=36),
                         title=title, xaxis_title="fase",
                         yaxis_title="flujo relativo")
    return figure


def image_figure(image, title):
    figure = go.Figure(go.Heatmap(z=image, colorscale="Greys", reversescale=True,
                                  showscale=False))
    figure.update_layout(height=230, margin=dict(l=10, r=10, t=34, b=10),
                         title=title, xaxis=dict(visible=False),
                         yaxis=dict(visible=False, scaleanchor="x"))
    return figure


def probability_figure(row, title):
    values = [row[f"p_{name}"] for name in config.CLASS_NAMES]
    order = np.argsort(values)[::-1]
    names = [config.CLASS_NAMES[position] for position in order]
    figure = go.Figure(go.Bar(
        x=[values[position] for position in order], y=names, orientation="h",
        marker_color=[CLASS_COLORS.get(name, "#cccccc") for name in names]))
    figure.update_layout(height=230, margin=dict(l=50, r=10, t=34, b=30),
                         title=title, xaxis=dict(range=[0, 1], title="p"),
                         yaxis=dict(autorange="reversed"))
    return figure


def main():
    st.set_page_config(page_title="Whitening bench", layout="wide")
    sidebar = st.sidebar

    tic = int(sidebar.number_input("TIC", value=DEFAULT_TIC, step=1, format="%d"))
    sectors = load_sectors(tic)
    if not sectors:
        st.error(f"Sin FITS para TIC {tic} en {GOLDEN_DIR}")
        return
    chosen = sidebar.multiselect("Sectores", sorted(sectors),
                                 default=sorted(sectors)[:1])
    if not chosen:
        st.info("Elegi al menos un sector.")
        return
    sources = sidebar.multiselect("Periodograma", ["LS", "ACF"],
                                  default=["LS", "ACF"])
    top_n = sidebar.slider("Picos por fuente", 1, 20, 8)
    min_peak_sep_days = sidebar.number_input("Separacion minima (d)", value=0.5,
                                             step=0.1, format="%.2f")
    n_harmonics = sidebar.slider("Armonicos al blanquear", 1, 6,
                                 prewhiten.DEFAULT_HARMONICS)
    checkpoint = sidebar.text_input(
        "Checkpoint", str(config.WEIGHTS_DIR / "Number_ELL" / "cp.ckpt"))

    key = (tic, tuple(chosen))
    if st.session_state.get("key") != key:
        st.session_state.key = key
        st.session_state.whitened = []

    time, flux, error = load_light_curve(tuple(sectors[s] for s in chosen))
    current = whiten(time, flux, st.session_state.whitened, n_harmonics)
    grids, levels, candidates = run_periodograms(
        time, current, error, tuple(sources), top_n, min_peak_sep_days)
    candidates = classify_candidates(checkpoint, time, current, candidates)
    if not candidates.empty:
        candidates["ya_blanqueado"] = [
            any(abs(period / done - 1.0) < 0.01
                for done in st.session_state.whitened)
            for period in candidates["period"]]

    baseline = time.max() - time.min()
    sidebar.markdown(
        f"**{len(time)} puntos**, baseline {baseline:.2f} d  \n"
        f"resolucion 1/T = {1.0 / baseline:.4f} d-1  \n"
        f"{len(candidates)} candidatos  \n"
        f"**{len(st.session_state.whitened)} periodos blanqueados**")

    label = "+".join(f"s{sector}" for sector in chosen)
    st.markdown(f"### TIC {tic} — {label}"
                + (f" — iteracion {len(st.session_state.whitened)}"
                   if st.session_state.whitened else ""))

    if candidates.empty:
        st.warning("Ningun pico sobrevive la seleccion sobre esta curva.")
        st.plotly_chart(light_curve_figure(time, flux, current),
                        use_container_width=True, key="lightcurve")
        return

    if "period" not in st.session_state or st.session_state.period not in set(
            candidates["period"]):
        st.session_state.period = float(candidates["period"].iloc[0])

    event = st.plotly_chart(
        periodogram_figure(grids, levels, candidates, st.session_state.period,
                           REFERENCE_PERIODS.get(tic, []),
                           st.session_state.whitened),
        use_container_width=True, on_select="rerun", key="periodogram",
        selection_mode=("points",))
    points = event.get("selection", {}).get("points", []) if event else []
    if points:
        st.session_state.period = float(points[0]["x"])

    st.plotly_chart(light_curve_figure(time, flux, current),
                    use_container_width=True, key="lightcurve")

    st.markdown("#### Candidatos — marcá los que querés blanquear")
    st.caption(
        f"`ciclos` = baseline/P; abajo de {MIN_CYCLES} no hay período que "
        "acertar. `snr` es la amplitud del fundamental sobre el ruido local: "
        "abajo de 4 (Breger) la frecuencia no se distingue del ruido. "
        "Ninguna de las dos filtra sola — están para decidir cuándo parar. "
        "`parte_de` marca los armónicos de una variación ya clasificada como "
        "coherente: esos NO vienen pre-marcados, porque blanquearlos le "
        "arranca un pedazo a esa variación.")
    table = candidates[["period", "kind", "source", "power", "prominence",
                        "ciclos", "snr", "clase", "p_clase", "log_pLPV",
                        "parte_de", "ya_blanqueado"]].copy()
    table.insert(0, "blanquear", table["clase"].isin(INCOHERENT)
                 & table["parte_de"].eq("") & ~table["ya_blanqueado"])
    edited = st.data_editor(
        table, hide_index=True, use_container_width=True,
        disabled=[column for column in table.columns if column != "blanquear"],
        key=f"editor_{len(st.session_state.whitened)}")

    actions = st.columns([2, 2, 1, 3])
    to_whiten = edited.loc[edited["blanquear"], "period"].tolist()
    if actions[0].button(f"Blanquear {len(to_whiten)} y recalcular",
                         disabled=not to_whiten):
        st.session_state.whitened = st.session_state.whitened + to_whiten
        st.rerun()
    if actions[1].button("Deshacer ultimo", disabled=not st.session_state.whitened):
        st.session_state.whitened = st.session_state.whitened[:-1]
        st.rerun()
    if actions[2].button("Reset", disabled=not st.session_state.whitened):
        st.session_state.whitened = []
        st.rerun()
    if st.session_state.whitened:
        actions[3].caption("blanqueados: " + ", ".join(
            f"{period:.4f} d" for period in st.session_state.whitened))

    period = float(st.session_state.period)
    row = candidates.iloc[
        int(np.argmin(np.abs(candidates["period"] - period)))]
    st.markdown(f"#### Pico seleccionado: P = {row['period']:.5f} d "
                f"({row['kind']} / {row['source']}) — **{row['clase']}** "
                f"p={row['p_clase']:.3f}, -log10 p_LPV={row['log_pLPV']:.2f}")

    columns = st.columns([2, 1, 1])
    with columns[0]:
        st.plotly_chart(fold_figure(time, current, row["period"], "phase fold"),
                        use_container_width=True, key="fold")
    with columns[1]:
        st.plotly_chart(
            image_figure(features.phase_fold_hist2d_log(time, current,
                                                        row["period"]),
                         "entrada CNN 32x32"),
            use_container_width=True, key="image")
    with columns[2]:
        st.plotly_chart(probability_figure(row, row["clase"]),
                        use_container_width=True, key="probability")

    note = st.text_input("Nota", key="note")
    if st.button("Guardar veredicto de esta iteracion"):
        save_verdict(tic, label, st.session_state.whitened, candidates, note)
        st.success(f"Guardado en {VERDICTS}")


def save_verdict(tic, label, whitened, candidates, note):
    rows = candidates.assign(
        TIC=tic, sectors=label, iteration=len(whitened),
        whitened="|".join(f"{period:.6f}" for period in whitened), note=note)
    VERDICTS.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(VERDICTS, mode="a", header=not VERDICTS.exists(), index=False)


if __name__ == "__main__":
    main()
