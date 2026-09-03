#!/usr/bin/env python
"""Interactive 3D map of the massive-star sample as a standalone HTML.

    python scripts/plot_3d_gaia_html.py
    python scripts/plot_3d_gaia_html.py --exaggeration 15 --output reports/massive_3d.html

Galactocentric cartesian, Galactic centre at the origin. Two populations, two
colour encodings:
  parent massive sample (label <= 12) -> discrete colour per spectral subtype
  BRF candidates                      -> continuous red-to-blue ramp on B2-cut-prob

`label` is the Skiff spectral type code, so the massive training class of the
paper is label <= 12 (B2 or earlier): the parent sample only spans O, B0, B1
and B2, one colour each.

The sample is a magnitude-limited (G < 12) set of young disc stars: |Z| has a
median of 67 pc against a ~10 kpc span in X and Y. At true aspect ratio that
pancake degenerates into a plane, so the Z axis is stretched by a factor the
reader can change from the buttons — the data is untouched, only the aspect
ratio of the scene. A reference grid (solar circle, constant-R rings, lines of
sight, Reid+2019 spiral arms) gives the eye something to anchor on, since
otherwise the points float without any frame.

Stars with plx/e_plx < 5 are drawn small and faint: their 1/plx distance is
unreliable and they are what produces the radial fingers pointing away from
the Sun.
"""
import argparse
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.coordinates import Galactocentric

REPO_ROOT = Path(__file__).resolve().parents[1]

SUN_RADIUS = Galactocentric().galcen_distance.to_value(u.kpc)
SUN_HEIGHT = Galactocentric().z_sun.to_value(u.kpc)
SUN_POSITION = (-SUN_RADIUS, 0.0, SUN_HEIGHT)

# Skiff spectral types are coded as 10 * class_index + subtype (O = 0-9,
# B = 10-19), so the massive cut label <= 12 spans O, B0, B1 and B2 only.
SPECTRAL_GROUPS = ["O", "B0", "B1", "B2"]

# An ordered teal-to-green family, one step per subtype: the parent sample is a
# spectral sequence, so the colours are sequential rather than arbitrary. The
# whole family sits away from the red-orange-gold-blue of the candidate ramp,
# so no parent star can be mistaken for a probability.
SPECTRAL_COLORS = {
    "O": (0, 66, 72),
    "B0": (10, 116, 110),
    "B1": (26, 158, 133),
    "B2": (74, 190, 145),
}

# Amber to navy, no red at all: the low end of the ramp is the paper's 0.6
# selection threshold, not a failure, and a red there reads as one. Warm to
# cool still orders the scale without alarming.
CANDIDATE_COLORSCALE = [
    [0.00, "#e8a33c"],
    [0.25, "#c9a95f"],
    [0.50, "#8ba3a8"],
    [0.75, "#5a8fc9"],
    [1.00, "#10265c"],
]

# The BRF selection threshold of the paper: no candidate exists below it, so
# anchoring the ramp at 0 would collapse the whole sample into the blue end.
CANDIDATE_PROBABILITY_MIN = 0.6

SCENE_WIDTH = 0.76            # fraction of the figure left to the 3D scene
RELIABLE_PARALLAX = 5.0        # plx / e_plx below which 1/plx is not trustworthy
GRID_COLOR = "rgba(130,130,130,0.35)"
ARM_COLOR = "rgba(45,45,45,0.95)"
SOLAR_CIRCLE_COLOR = "rgba(196,138,0,0.9)"

# Reid et al. (2019), ApJ 885, 131, Table 2. Log-periodic spirals with a kink:
#   ln(R / R_kink) = -(beta - beta_kink) * tan(psi)
# with psi = psi_low below the kink and psi_high above it, and beta the
# galactocentric azimuth measured from the Sun's direction towards rotation.
SPIRAL_ARMS = [
    # name,        beta_min, beta_max, R_kink, beta_kink, psi_low, psi_high
    ("3-kpc",         15,  18,  3.52, 15,  -4.2,  -4.2),
    ("Norma",          5,  54,  4.46, 18,  -1.0,  19.5),
    ("Scutum-Centaurus", 0, 104, 4.91, 23,  14.1,  12.1),
    ("Sagittarius-Carina", 2, 97, 6.04, 24,  17.1,   1.0),
    ("Local",         -8,  34,  8.26,  9,  11.4,  11.4),
    ("Perseus",      -21,  88,  8.87, 40,  10.3,   8.7),
    ("Outer",         -6,  56, 12.24, 18,   3.0,   9.4),
]

EXAGGERATION_STEPS = [1, 5, 15]


def spectral_group(type_code):
    if pd.isna(type_code):
        return None
    code = int(round(float(type_code)))
    if 0 <= code <= 9:
        return "O"
    if 10 <= code <= 12:
        return f"B{code - 10}"
    return None


def arm_track(beta_min, beta_max, radius_kink, beta_kink, psi_low, psi_high):
    beta = np.radians(np.linspace(beta_min, beta_max, 200))
    pitch = np.where(beta < np.radians(beta_kink),
                     np.radians(psi_low), np.radians(psi_high))
    radius = radius_kink * np.exp(-(beta - np.radians(beta_kink)) * np.tan(pitch))
    # beta = 0 points from the Galactic centre towards the Sun (X < 0), and
    # grows towards Galactic rotation (+Y at the solar circle).
    return -radius * np.cos(beta), radius * np.sin(beta)


def hover_text(frame):
    # Column access by name, not itertuples: "BP-RP" is not a valid identifier
    # and itertuples would rename it to a positional _N.
    return [
        f"Gaia DR3 {source_id}<br>"
        f"d = {distance:.2f} kpc   (plx/e_plx = {reliability:.1f})<br>"
        f"R = {radius:.2f} kpc   Z = {height * 1000:.0f} pc<br>"
        f"l = {longitude:.2f}°   b = {latitude:.2f}°<br>"
        f"G = {magnitude:.2f}   BP-RP = {colour:.2f}"
        for (source_id, distance, reliability, radius, height,
             longitude, latitude, magnitude, colour) in zip(
            frame["GaiaDR3"], frame["distance_kpc"], frame["RPlx"],
            frame["R_galactocentric_kpc"], frame["Z_kpc"],
            frame["l"], frame["b"], frame["Gmag"], frame["BP-RP"])
    ]


def add_reference_grid(figure, max_radius):
    angle = np.radians(np.linspace(0, 360, 361))

    ring_x, ring_y = [], []
    for radius in np.arange(2, max_radius + 2, 2):
        ring_x.extend(list(radius * np.cos(angle)) + [None])
        ring_y.extend(list(radius * np.sin(angle)) + [None])
    figure.add_trace(go.Scatter3d(
        x=ring_x, y=ring_y, z=[0] * len(ring_x),
        mode="lines", name="R rings (2 kpc)",
        legendgroup="reference", legendgrouptitle_text="Reference",
        line=dict(color=GRID_COLOR, width=1),
        hoverinfo="skip",
    ))

    figure.add_trace(go.Scatter3d(
        x=SUN_RADIUS * np.cos(angle), y=SUN_RADIUS * np.sin(angle),
        z=np.zeros_like(angle),
        mode="lines", name=f"Solar circle ({SUN_RADIUS:.2f} kpc)",
        legendgroup="reference",
        line=dict(color=SOLAR_CIRCLE_COLOR, width=3),
        hoverinfo="skip",
    ))

    sight_x, sight_y = [], []
    for longitude in np.radians(np.arange(0, 360, 30)):
        sight_x.extend([SUN_POSITION[0],
                        SUN_POSITION[0] + max_radius * np.cos(longitude), None])
        sight_y.extend([SUN_POSITION[1],
                        SUN_POSITION[1] + max_radius * np.sin(longitude), None])
    figure.add_trace(go.Scatter3d(
        x=sight_x, y=sight_y, z=[0] * len(sight_x),
        mode="lines", name="Lines of sight (l, 30°)",
        legendgroup="reference",
        line=dict(color=GRID_COLOR, width=1, dash="dot"),
        hoverinfo="skip", visible="legendonly",
    ))

    arm_x, arm_y, arm_name = [], [], []
    for name, *parameters in SPIRAL_ARMS:
        x, y = arm_track(*parameters)
        arm_x.extend(list(x) + [None])
        arm_y.extend(list(y) + [None])
        arm_name.extend([name] * len(x) + [""])
    figure.add_trace(go.Scatter3d(
        x=arm_x, y=arm_y, z=[0] * len(arm_x),
        mode="lines", name="Spiral arms (Reid+2019)",
        legendgroup="reference",
        line=dict(color=ARM_COLOR, width=6),
        text=arm_name, hovertemplate="%{text}<extra></extra>",
    ))


def add_populations(figure, catalog, point_size, probability_min):
    reliable = catalog["RPlx"] >= RELIABLE_PARALLAX
    labelled = catalog.loc[catalog["is_labelled"]]
    candidates = catalog.loc[catalog["is_candidate"]]

    for spectral in SPECTRAL_GROUPS:
        subset = labelled.loc[labelled["spectral_group"] == spectral]
        if len(subset) == 0:
            continue
        red, green, blue = SPECTRAL_COLORS[spectral]
        trustworthy = subset["RPlx"] >= RELIABLE_PARALLAX
        # marker.opacity is scalar-only in Scatter3d, so the per-point fading
        # of the unreliable distances has to travel inside the rgba colours.
        colors = [f"rgba({red},{green},{blue},{0.95 if ok else 0.2})"
                  for ok in trustworthy]
        sizes = np.where(trustworthy, point_size * 1.5, point_size * 0.9)
        figure.add_trace(go.Scatter3d(
            x=subset["X_kpc"], y=subset["Y_kpc"], z=subset["Z_kpc"],
            mode="markers",
            name=f"{spectral} ({trustworthy.sum()})",
            legendgroup="parent",
            legendgrouptitle_text="Parent massive sample (label ≤ B2)",
            marker=dict(size=sizes, color=colors, line=dict(width=0)),
            text=hover_text(subset),
            hovertemplate="%{text}<extra>" + spectral + "</extra>",
        ))

    unclassified = labelled.loc[labelled["spectral_group"].isna()]
    if len(unclassified) > 0:
        figure.add_trace(go.Scatter3d(
            x=unclassified["X_kpc"], y=unclassified["Y_kpc"],
            z=unclassified["Z_kpc"],
            mode="markers", name=f"no type ({len(unclassified)})",
            legendgroup="parent",
            marker=dict(size=point_size, color="rgba(154,160,166,0.45)",
                        line=dict(width=0)),
            text=hover_text(unclassified),
            hovertemplate="%{text}<extra>no type</extra>",
            visible="legendonly",
        ))

    # A colorscale forbids per-point alpha, so the two reliability regimes are
    # two traces sharing one scale instead of one trace with rgba colours.
    for trustworthy, opacity, size_scale, suffix in [
            (True, 0.8, 1.0, ""), (False, 0.15, 0.6, ", plx/e_plx < 5")]:
        subset = candidates.loc[
            (candidates["RPlx"] >= RELIABLE_PARALLAX) == trustworthy]
        if len(subset) == 0:
            continue
        figure.add_trace(go.Scatter3d(
            x=subset["X_kpc"], y=subset["Y_kpc"], z=subset["Z_kpc"],
            mode="markers",
            name=f"candidates ({len(subset)}){suffix}",
            legendgroup="candidates",
            legendgrouptitle_text="BRF candidates",
            marker=dict(
                size=point_size * size_scale,
                color=subset["B2-cut-prob"],
                colorscale=CANDIDATE_COLORSCALE,
                cmin=probability_min, cmax=1.0,
                opacity=opacity,
                line=dict(width=0),
                showscale=trustworthy,
                # Horizontal and inside the bottom margin: a vertical bar next
                # to the scene gets clipped by the iframe on a narrow page.
                colorbar=dict(
                    title=dict(text="BRF probability (B2 cut)", side="top"),
                    orientation="h", x=SCENE_WIDTH, xanchor="right",
                    y=1.015, yanchor="bottom",
                    len=0.34, thickness=11, outlinewidth=0,
                    tickvals=np.round(np.linspace(probability_min, 1.0, 5), 2)),
            ),
            text=hover_text(subset),
            customdata=subset["B2-cut-prob"],
            hovertemplate="%{text}<br>B2-cut prob = %{customdata:.3f}<extra></extra>",
        ))

    figure.add_trace(go.Scatter3d(
        x=[SUN_POSITION[0]], y=[SUN_POSITION[1]], z=[SUN_POSITION[2]],
        mode="markers", name="Sun", legendgroup="reference",
        marker=dict(size=point_size + 5, color="#ffd400", symbol="diamond",
                    line=dict(width=1, color="#333")),
        hovertemplate=f"Sun<br>R = {SUN_RADIUS:.3f} kpc<extra></extra>",
    ))
    figure.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        name="Galactic centre", legendgroup="reference",
        marker=dict(size=point_size + 5, color="#000000", symbol="square",
                    line=dict(width=1)),
        hovertemplate="Galactic centre<extra></extra>",
    ))


def build_figure(catalog, exaggeration, point_size, view_radius,
                 height_percentile, probability_min):
    figure = go.Figure()
    add_reference_grid(figure, view_radius)
    add_populations(figure, catalog, point_size, probability_min)

    # The Z box hugs the disc instead of the outliers: at p99.5 the half-range
    # is 3 kpc, so a 15x stretch turns the scene into a tower with the same
    # unreadable sheet at its waist.
    height_limit = np.nanpercentile(np.abs(catalog["Z_kpc"]), height_percentile)
    scene_ranges = dict(x=[-view_radius, view_radius],
                        y=[-view_radius, view_radius],
                        z=[-height_limit, height_limit])

    def aspect(factor):
        return dict(x=1.0, y=1.0,
                    z=factor * height_limit / view_radius)

    axis = dict(backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(128,128,128,0.2)",
                zerolinecolor="rgba(128,128,128,0.45)", showspikes=False)

    figure.update_layout(
        template="plotly_white",
        scene=dict(
            # Short axis titles: plotly draws them along the axis, and the
            # long parenthetical version ran straight through the disc.
            xaxis=dict(title="X [kpc]", range=scene_ranges["x"], **axis),
            yaxis=dict(title="Y [kpc]", range=scene_ranges["y"], **axis),
            zaxis=dict(title="Z [kpc]", range=scene_ranges["z"], **axis),
            domain=dict(x=[0.0, SCENE_WIDTH], y=[0.0, 1.0]),
            aspectmode="manual",
            aspectratio=aspect(exaggeration),
            # Nearly top-down: at Z x1 the disc is a sheet, so the face-on map
            # is the only informative angle; the small tilt keeps the Z stretch
            # perceptible when the reader switches it.
            camera=dict(eye=dict(x=0.0, y=-0.45, z=1.7),
                        up=dict(x=0, y=1, z=0)),
        ),
        legend=dict(itemsizing="constant", groupclick="toggleitem",
                    x=SCENE_WIDTH + 0.015, xanchor="left",
                    y=0.89, yanchor="top",
                    bgcolor="rgba(255,255,255,0)"),
        margin=dict(l=10, r=10, t=70, b=20),
        autosize=True,
        updatemenus=[
            dict(
                type="buttons", direction="right",
                x=0, y=1.015, xanchor="left", yanchor="bottom",
                showactive=True,
                active=EXAGGERATION_STEPS.index(exaggeration),
                buttons=[
                    dict(label=f"Z ×{factor}", method="relayout",
                         args=[{"scene.aspectratio": aspect(factor)}])
                    for factor in EXAGGERATION_STEPS
                ],
            ),
        ],
    )
    return figure


# Injected into the standalone page: a fullscreen toggle that also works when
# the page is embedded in an iframe (the host iframe needs `allowfullscreen`),
# plus the resize calls plotly needs whenever the viewport changes.
PAGE_SCRIPT = """
(function () {
    var graph = document.getElementById('{plot_id}');

    var style = document.createElement('style');
    style.textContent = 'html, body { margin: 0; height: 100%; }'
        + '.fullscreen-toggle { position: fixed; bottom: 12px; right: 12px;'
        + ' z-index: 10; padding: 5px 12px; font: 13px/1.4 system-ui, sans-serif;'
        + ' color: #2a3f5f; background: #fff; border: 1px solid #c8d4e3;'
        + ' border-radius: 4px; cursor: pointer; }'
        + '.fullscreen-toggle:hover { background: #f2f6fa; }';
    document.head.appendChild(style);

    var button = document.createElement('button');
    button.className = 'fullscreen-toggle';
    button.textContent = 'Fullscreen';
    document.body.appendChild(button);

    button.addEventListener('click', function () {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen();
        }
    });

    function refresh() {
        button.textContent = document.fullscreenElement ? 'Exit fullscreen'
                                                        : 'Fullscreen';
        Plotly.Plots.resize(graph);
    }
    document.addEventListener('fullscreenchange', refresh);
    window.addEventListener('resize', function () { Plotly.Plots.resize(graph); });
})();
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path,
                        default=REPO_ROOT / "catalogs" / "0_MassiveG12_3D.csv")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "reports" / "massive_stars_3d.html")
    parser.add_argument("--exaggeration", type=int, default=1,
                        choices=EXAGGERATION_STEPS,
                        help="Initial Z stretch; the reader can change it.")
    parser.add_argument("--view-radius", type=float, default=16.0,
                        help="Half-size of the initial X/Y box [kpc]. Every "
                             "star stays in the file; 'full extent' unzooms.")
    parser.add_argument("--height-percentile", type=float, default=95.0,
                        help="Percentile of |Z| setting the initial Z box.")
    parser.add_argument("--probability-min", type=float,
                        default=CANDIDATE_PROBABILITY_MIN,
                        help="Low end of the candidate colour ramp.")
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--plotly-js", default="cdn", choices=["cdn", "inline"],
                        help="'cdn' keeps the page under 1 MB.")
    arguments = parser.parse_args()

    catalog = pd.read_csv(arguments.catalog, low_memory=False)
    catalog = catalog.loc[catalog["X_kpc"].notna()]
    catalog["spectral_group"] = catalog["sk-type-mean"].map(spectral_group)
    # Plotly serialises float64 in full precision; 0.1 pc is far finer than the
    # ~100 pc distance uncertainty and roughly halves the page weight.
    catalog[["X_kpc", "Y_kpc", "Z_kpc"]] = catalog[
        ["X_kpc", "Y_kpc", "Z_kpc"]].round(4)

    figure = build_figure(catalog, arguments.exaggeration,
                          arguments.point_size, arguments.view_radius,
                          arguments.height_percentile,
                          arguments.probability_min)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        arguments.output,
        include_plotlyjs=True if arguments.plotly_js == "inline" else "cdn",
        full_html=True,
        default_height="100%", default_width="100%",
        config=dict(responsive=True),
        post_script=PAGE_SCRIPT)
    size_mb = arguments.output.stat().st_size / 1e6
    print(f"Written {arguments.output} ({size_mb:.1f} MB, "
          f"{len(catalog)} stars, plotly.js {arguments.plotly_js})")


if __name__ == "__main__":
    main()
