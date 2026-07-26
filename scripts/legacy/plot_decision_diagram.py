"""
Decision flowchart for the unified LS+ACF classification pipeline.
Uses Graphviz for proper flowchart rendering.
Generates: diagrama_clasificacion.pdf / .png
"""

import os
import graphviz

dot = graphviz.Digraph("pipeline", format="png")
dot.attr(rankdir="TB", fontname="Helvetica", bgcolor="white",
         nodesep="0.4", ranksep="0.55", margin="0.3")
dot.attr("node", fontname="Helvetica", fontsize="11", style="filled",
         shape="box", penwidth="1.5")
dot.attr("edge", fontname="Helvetica", fontsize="9", color="#555555",
         arrowsize="0.8")

# ── Colors ───────────────────────────────────────────────────────────
C_INPUT  = "#D1C4E9"  # purple — input
C_METHOD = "#BBDEFB"  # blue — methods
C_MODEL  = "#F3E5F5"  # light purple — model
C_STEP   = "#E8F0FE"  # light blue — process steps
C_DECIDE = "#FFF3E0"  # orange — decisions
C_RESULT = "#C8E6C9"  # green — final results
C_FILTER = "#FFCDD2"  # red — filtered out
C_WARN   = "#FFF8E1"  # yellow — intermediate
C_FINAL  = "#1B5E20"  # dark green border

# ── Input ────────────────────────────────────────────────────────────
dot.node("input",
         "3557 OB stars  ×  71 TESS sectors\n7455 unique TIC–sector pairs",
         fillcolor=C_INPUT, shape="box", style="filled,rounded",
         fontsize="13", penwidth="2", color="#5E35B1")

# ── Methods ──────────────────────────────────────────────────────────
dot.node("ls",  "Lomb-Scargle\n168582 peaks",
         fillcolor=C_METHOD, color="#1565C0", style="filled,rounded")
dot.node("acf", "Autocorrelation (ACF)\n303417 peaks",
         fillcolor=C_METHOD, color="#1565C0", style="filled,rounded")

dot.edge("input", "ls")
dot.edge("input", "acf")

# ── CNN + RF ─────────────────────────────────────────────────────────
dot.node("cnnrf",
         "CNN + Random Forest\n471999 peaks classified in total",
         fillcolor=C_MODEL, color="#7B1FA2", style="filled,rounded",
         fontsize="12")

dot.edge("ls",  "cnnrf")
dot.edge("acf", "cnnrf")

# ── Combine ──────────────────────────────────────────────────────────
dot.node("combine",
         "Combine LS + ACF peaks per TIC–sector\n7455 groups  (~63 peaks each)",
         fillcolor=C_STEP, color="#4285F4", style="filled,rounded")

dot.edge("cnnrf", "combine")

# ── Step 1a: No peak above FAP ───────────────────────────────────────
dot.node("d_fap",
         "No peak above FAP\n(all sectors)?",
         fillcolor=C_DECIDE, shape="diamond", color="#E65100",
         fontsize="10", height="0.8", width="2.8")

dot.edge("combine", "d_fap")

dot.node("non_periodic",
         "Non-periodic — 146 TICs\n(No_periodo: 8  |  Rndm: 138)",
         fillcolor=C_FILTER, color="#C62828", style="filled,rounded",
         fontsize="10")
dot.edge("d_fap", "non_periodic", label="  Yes  ", fontcolor="#C62828", color="#C62828")

# ── Step 1b: All high-prob CNN+RF = Rndm ─────────────────────────────
dot.node("d_rndm",
         "All CNN+RF classifications\n(prob > threshold) = Rndm?",
         fillcolor=C_DECIDE, shape="diamond", color="#E65100",
         fontsize="10", height="0.8", width="3.2")

dot.edge("d_fap", "d_rndm", label="  No  ")
dot.edge("d_rndm", "non_periodic", label="  Yes  ", fontcolor="#C62828", color="#C62828")

# ── Step 1c: All high-prob CNN+RF = Rndm or LPV ──────────────────────
dot.node("d_lpv",
         "All CNN+RF classifications\n(prob > threshold) = Rndm or LPV?",
         fillcolor=C_DECIDE, shape="diamond", color="#E65100",
         fontsize="10", height="0.8", width="3.2")

dot.edge("d_rndm", "d_lpv", label="  No  ")

dot.node("irregular",
         "Irregular — 23 TICs",
         fillcolor=C_FILTER, color="#C62828", style="filled,rounded",
         fontsize="10")
dot.edge("d_lpv", "irregular", label="  Yes  ", fontcolor="#C62828", color="#C62828")

# ── Step 1d: Periodic candidates ─────────────────────────────────────
dot.node("periodic_cand",
         "Periodic candidates",
         fillcolor=C_STEP, color="#4285F4", style="filled,rounded",
         fontsize="10")

dot.edge("d_lpv", "periodic_cand", label="  No  ")

# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — Best classification per sector
# ═══════════════════════════════════════════════════════════════════════

# ── Step 2a: Best per periodogram ────────────────────────────────────
dot.node("group_pg",
         "Select highest CNN+RF prob\nfor ACF and for LS",
         fillcolor=C_STEP, color="#4285F4", style="filled,rounded",
         fontsize="10")

dot.edge("periodic_cand", "group_pg")

# caveat: Tie 1 (to the right)
dot.node("tie1",
         "Tie 1  equal prob within periodogram:\nselect lower power peak",
         fillcolor=C_WARN, color="#F9A825", style="filled,rounded",
         fontsize="9")
dot.edge("group_pg", "tie1",
         style="dashed", color="#F9A825", fontcolor="#F9A825",
         label="  tie within periodogram  ")

# ── Step 2b: Select highest classification (ACF vs LS) ───────────────
dot.node("best_class",
         "Select highest classification\n(ACF vs LS)",
         fillcolor=C_STEP, color="#4285F4", style="filled,rounded",
         fontsize="10")

dot.edge("group_pg", "best_class")

# caveat: Tie 2 (to the right)
dot.node("tie2",
         "Tie 2  equal prob ACF vs LS:\nkeep both as probable period",
         fillcolor=C_WARN, color="#F9A825", style="filled,rounded",
         fontsize="9")
dot.edge("best_class", "tie2",
         style="dashed", color="#F9A825", fontcolor="#F9A825",
         xlabel="  equal prob  ")

# caveat: Unconstrained (to the right)
dot.node("unconstrained_s2",
         "Unconstrained sector",
         fillcolor=C_WARN, color="#F9A825", style="filled,rounded",
         fontsize="9")
dot.edge("best_class", "unconstrained_s2",
         style="dashed", color="#F9A825", fontcolor="#F9A825",
         label="  prob < 0.8  ")

# ── Step 2 output ─────────────────────────────────────────────────────
dot.node("best_class_sector",
         "Best classification per sector",
         fillcolor=C_RESULT, color="#2E7D32", style="filled,rounded",
         fontsize="10", penwidth="2")

dot.edge("best_class", "best_class_sector")

# side-effect nodes also feed into per-sector catalog
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("group_pg")
    s.node("tie1")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("best_class")
    s.node("tie2")
    s.node("unconstrained_s2")

# ── Per-sector catalog ───────────────────────────────────────────────
dot.node("per_sector",
         "Per-sector catalog\n11658 entries — 3557 TICs",
         fillcolor="#E0F2F1", color="#00695C", style="filled,rounded",
         shape="box", fontsize="10")

dot.edge("best_class_sector", "per_sector")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Best classification per TIC
# ═══════════════════════════════════════════════════════════════════════

# ── Step 3a: Mode of sector votes ────────────────────────────────────
dot.node("mode_tic",
         "Mode of sector votes per TIC\n(Unconstrained sectors: 0 votes)",
         fillcolor=C_STEP, color="#4285F4", style="filled,rounded",
         fontsize="10")

dot.edge("per_sector", "mode_tic")

# caveat: Mixed (to the right)
dot.node("mixed_tic",
         "Mixed\n(tie in mode — for review)",
         fillcolor=C_FILTER, color="#C62828", style="filled,rounded",
         fontsize="9")
dot.edge("mode_tic", "mixed_tic",
         style="dashed", color="#C62828", fontcolor="#C62828",
         xlabel="  tie in mode  ")

# caveat: Unconstrained (to the right)
dot.node("unconstrained_tic",
         "Unconstrained\n(no votes for TIC)",
         fillcolor=C_WARN, color="#F9A825", style="filled,rounded",
         fontsize="9")
dot.edge("mode_tic", "unconstrained_tic",
         style="dashed", color="#F9A825", fontcolor="#F9A825",
         xlabel="  all sectors\nUnconstrained  ")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("mode_tic")
    s.node("mixed_tic")
    s.node("unconstrained_tic")

# ── FINAL CATALOG ────────────────────────────────────────────────────
dot.node("final",
         '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3">'
         '<TR><TD COLSPAN="2"><B><FONT POINT-SIZE="14">Final Catalog — 3557 TICs</FONT></B></TD></TR>'
         '<TR><TD>─────────────────────────</TD></TR>'
         '<TR><TD ALIGN="LEFT"><FONT COLOR="#1565C0">◆</FONT> ELL:</TD>'
         '    <TD ALIGN="RIGHT"><B>1849</B></TD></TR>'
         '<TR><TD ALIGN="LEFT"><FONT COLOR="#7B1FA2">◆</FONT> Pulsating:</TD>'
         '    <TD ALIGN="RIGHT"><B>581</B></TD></TR>'
         '<TR><TD ALIGN="LEFT"><FONT COLOR="#F9A825">◆</FONT> Unconstrained:</TD>'
         '    <TD ALIGN="RIGHT"><B>477</B></TD></TR>'
         '<TR><TD ALIGN="LEFT"><FONT COLOR="#E65100">◆</FONT> Eclipsing (E):</TD>'
         '    <TD ALIGN="RIGHT"><B>281</B></TD></TR>'
         '<TR><TD ALIGN="LEFT"><FONT COLOR="#9E9E9E">◆</FONT> Mixed:</TD>'
         '    <TD ALIGN="RIGHT"><B>200</B></TD></TR>'
         '<TR><TD ALIGN="LEFT"><FONT COLOR="#616161">◆</FONT> Non-periodic:</TD>'
         '    <TD ALIGN="RIGHT"><B>146</B></TD></TR>'
         '<TR><TD ALIGN="LEFT"><FONT COLOR="#2E7D32">◆</FONT> Irregular:</TD>'
         '    <TD ALIGN="RIGHT"><B>23</B></TD></TR>'
         '</TABLE>>',
         fillcolor=C_RESULT, color=C_FINAL, style="filled,rounded",
         shape="box", penwidth="2.5", fontsize="11")

dot.edge("mode_tic", "final", penwidth="2", color=C_FINAL, fontcolor=C_FINAL)

# ── Step labels on left ──────────────────────────────────────────────
dot.node("s1", "Step 1\nNon-periodic /\nIrregular filter",
         fillcolor="#FFCDD2", color="#B71C1C", fontsize="9",
         fontcolor="#B71C1C", shape="note", style="filled",
         width="1.2", height="0.6")
dot.node("s2", "Step 2\nBest\nclassification\nper sector",
         fillcolor="#FFCDD2", color="#B71C1C", fontsize="9",
         fontcolor="#B71C1C", shape="note", style="filled",
         width="1.2", height="0.6")
dot.node("s3", "Step 3\nBest\nclassification\nper TIC",
         fillcolor="#FFCDD2", color="#B71C1C", fontsize="9",
         fontcolor="#B71C1C", shape="note", style="filled",
         width="1.2", height="0.6")

# Invisible edges to position step labels
dot.edge("s1", "d_fap",    style="invis")
dot.edge("s2", "group_pg", style="invis")
dot.edge("s3", "mode_tic", style="invis")

# Align step labels with first node of each step
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("s1")
    s.node("d_fap")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("s2")
    s.node("group_pg")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("s3")
    s.node("mode_tic")

# Align side outputs (Step 1)
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("d_fap")
    s.node("non_periodic")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("d_lpv")
    s.node("irregular")

# ── Render ───────────────────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
output_base = os.path.join(FIGURES_DIR, "diagrama_clasificacion")
dot.render(output_base, format="png", cleanup=True)
dot.render(output_base, format="pdf", cleanup=True)
print("Saved: diagrama_clasificacion.png / .pdf")
