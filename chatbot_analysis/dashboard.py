"""
dashboard.py
============
Executive dashboard for the Airport Multimodal Chatbot analytics.

Design language: "Departure-Board Analytics"
  * Deep navy canvas (evokes airport FIDS displays)
  * Amber + cyan accents (classic split-flap board palette)
  * Monospaced numerics for KPI values
  * Custom-drawn cards with subtle borders (no default matplotlib chrome)
  * Ring gauges for accuracy / CSAT (instead of bars)
  * Sparkline row for hourly session pulse
  * Explicit narrative captions on each panel

Fulfils the Part-3 dashboard requirement using only matplotlib so it runs
inside the project's conda env and in Colab with zero extra dependencies.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge, Circle, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "outputs" / "logs"
AN   = ROOT / "outputs" / "analytics"
OUT  = ROOT / "outputs" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
turns    = pd.read_csv(LOGS / "conversation_logs.csv")
sessions = pd.read_csv(LOGS / "sessions.csv")
summary  = json.loads((AN / "summary.json").read_text())
turns["timestamp"] = pd.to_datetime(turns["timestamp"])

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG          = "#0b1220"     # deep navy canvas
CARD        = "#111a2e"     # panel bg
CARD_EDGE   = "#1f2a44"     # subtle border
INK         = "#e6edf6"     # primary text
INK_DIM     = "#8b96b0"     # secondary text
INK_MUTE    = "#4b5573"     # tertiary / axis
AMBER       = "#f5a524"     # amber (FIDS)
CYAN        = "#22d3ee"     # cyan accent
GREEN       = "#34d399"
RED         = "#f87171"
VIOLET      = "#a78bfa"
GRID        = "#1a2540"

# Try to grab a monospaced font already present
MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": BG,
    "font.family": SANS,
    "text.color": INK,
    "axes.edgecolor": CARD_EDGE,
    "axes.labelcolor": INK_DIM,
    "axes.titlecolor": INK,
    "xtick.color": INK_MUTE,
    "ytick.color": INK_MUTE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": False,
})

# ---------------------------------------------------------------------------
# Helper drawing primitives
# ---------------------------------------------------------------------------
def panel(ax, title=None, caption=None):
    """Convert an axes into a titled card with a subtle border."""
    ax.set_facecolor(CARD)
    ax.set_xticks([]); ax.set_yticks([])
    # rounded border via a background patch drawn behind
    for spine in ax.spines.values(): spine.set_visible(False)
    # draw a light rounded border using a FancyBboxPatch on the axes background
    bbox = FancyBboxPatch((0, 0), 1, 1,
                          transform=ax.transAxes,
                          boxstyle="round,pad=0.005,rounding_size=0.018",
                          linewidth=1.0, edgecolor=CARD_EDGE,
                          facecolor=CARD, zorder=0, clip_on=False)
    ax.add_patch(bbox)
    if title:
        ax.text(0.03, 0.955, title.upper(), transform=ax.transAxes,
                fontsize=9.5, color=AMBER, fontweight="bold",
)
    if caption:
        # caption inside the panel, just above the bottom edge
        ax.text(0.03, 0.04, caption, transform=ax.transAxes,
                fontsize=8, color=INK_DIM, style="italic",
                va="bottom")


def kpi_tile(ax, label, value, delta=None, delta_good=True, accent=CYAN):
    ax.set_facecolor(CARD); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.add_patch(FancyBboxPatch((0,0),1,1, transform=ax.transAxes,
                boxstyle="round,pad=0.005,rounding_size=0.03",
                linewidth=1.0, edgecolor=CARD_EDGE, facecolor=CARD, zorder=0))
    # accent bar (left)
    ax.add_patch(Rectangle((0.02, 0.15), 0.015, 0.7, transform=ax.transAxes,
                           color=accent, zorder=1, clip_on=False))
    ax.text(0.08, 0.78, label.upper(), transform=ax.transAxes,
            fontsize=9, color=INK_DIM, fontweight="bold")
    ax.text(0.08, 0.36, value, transform=ax.transAxes,
            fontsize=30, color=INK, fontweight="bold", fontfamily=MONO)
    if delta is not None:
        col = GREEN if delta_good else RED
        arrow = "▲" if delta_good else "▼"
        ax.text(0.08, 0.16, f"{arrow} {delta}", transform=ax.transAxes,
                fontsize=9, color=col, fontweight="bold")


def ring_gauge(ax, value_pct, label, sub, color=CYAN):
    """A donut-style KPI gauge that reads value_pct (0-100)."""
    panel(ax)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.55, 1.55); ax.set_aspect("equal")
    # title at top (inside the panel)
    ax.text(0, 1.42, label.upper(), ha="center", va="center",
            fontsize=9, color=AMBER, fontweight="bold")
    # background ring
    ax.add_patch(Wedge((0,0.05), 0.85, 0, 360, width=0.16,
                       facecolor=GRID, edgecolor="none"))
    # value ring
    end_angle = 90 - (value_pct/100)*360
    ax.add_patch(Wedge((0,0.05), 0.85, end_angle, 90, width=0.16,
                       facecolor=color, edgecolor="none"))
    # centre value
    ax.text(0, 0.05, f"{value_pct:.1f}%", ha="center", va="center",
            fontsize=19, color=INK, fontweight="bold", fontfamily=MONO)
    # sub label BELOW the ring, well clear
    ax.text(0, -1.15, sub, ha="center", va="center",
            fontsize=8, color=INK_DIM)


def style_axes(ax):
    """Give a normal chart axes the dashboard look."""
    ax.set_facecolor(CARD)
    ax.tick_params(colors=INK_MUTE, labelsize=8)
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(CARD_EDGE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)


# ---------------------------------------------------------------------------
# Figure layout
#   6 rows x 6 cols grid, canvas 18x13"
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor(BG)

gs = GridSpec(6, 6, figure=fig,
              hspace=0.75, wspace=0.35,
              left=0.035, right=0.985, top=0.93, bottom=0.045)

# --- Header band ------------------------------------------------------------
eda = summary["eda"]
fig.text(0.035, 0.965, "AIRPORT MULTIMODAL CHATBOT",
         fontsize=11, color=AMBER, fontweight="bold")
fig.text(0.035, 0.945, "Analytics & Optimisation Dashboard",
         fontsize=22, color=INK, fontweight="bold")
fig.text(0.035, 0.926,
         "Chatbot Analytics & Optimization  •  MSc Artificial Intelligence  •  BSBI 2026",
         fontsize=10, color=INK_DIM)

# right-side "flight info" style stamp
right = 0.965
fig.text(right, 0.965, "DATA WINDOW", fontsize=8, color=INK_DIM, ha="right",
         fontweight="bold")
window = f"{turns['timestamp'].min():%d %b} → {turns['timestamp'].max():%d %b %Y}"
fig.text(right, 0.945, window, fontsize=14, color=CYAN, ha="right",
         fontfamily=MONO, fontweight="bold")
fig.text(right, 0.928,
         f"{eda['n_users']} users  •  {eda['n_sessions']} sessions  •  {eda['n_turns']} turns",
         fontsize=9, color=INK_DIM, ha="right")

# ---------------------------------------------------------------------------
# ROW 1 (rows 0): 4 KPI tiles
# ---------------------------------------------------------------------------
ax_k1 = fig.add_subplot(gs[0, 0:1]); kpi_tile(ax_k1, "Sessions",
    f"{eda['n_sessions']}", delta=f"{eda['n_turns']} turns", delta_good=True, accent=CYAN)
ax_k2 = fig.add_subplot(gs[0, 1:2]); kpi_tile(ax_k2, "Intent accuracy",
    f"{summary['intent_metrics']['accuracy']*100:.1f}%",
    delta=f"weighted F1 {summary['intent_metrics']['weighted_f1']:.2f}",
    delta_good=True, accent=GREEN)
ax_k3 = fig.add_subplot(gs[0, 2:3]); kpi_tile(ax_k3, "Fallback rate",
    f"{eda['fallback_rate_%']}%",
    delta=f"{summary['fallback_report']['combined_nlp_flags']} turns flagged",
    delta_good=False, accent=RED)
ax_k4 = fig.add_subplot(gs[0, 3:4]); kpi_tile(ax_k4, "Avg CSAT",
    f"{eda['csat_mean']}",
    delta=f"completion {eda['session_completion_rate_%']}%",
    delta_good=True, accent=AMBER)
ax_k5 = fig.add_subplot(gs[0, 4:5]); kpi_tile(ax_k5, "Median latency",
    f"{eda['response_time_ms_median']} ms",
    delta=f"p95 {eda['response_time_ms_p95']} ms",
    delta_good=True, accent=VIOLET)
ax_k6 = fig.add_subplot(gs[0, 5:6]); kpi_tile(ax_k6, "Frustration",
    f"{summary['frustration_session_share_%']}%",
    delta="of sessions", delta_good=False, accent=RED)

# ---------------------------------------------------------------------------
# ROW 2 (rows 1): three ring gauges + hourly pulse sparkline
# ---------------------------------------------------------------------------
ax_g1 = fig.add_subplot(gs[1, 0:1])
ring_gauge(ax_g1, summary['intent_metrics']['accuracy']*100,
           "NLU accuracy", "on classifiable turns", CYAN)

ax_g2 = fig.add_subplot(gs[1, 1:2])
ring_gauge(ax_g2, eda['session_completion_rate_%'],
           "Session completion", "task success proxy", GREEN)

ax_g3 = fig.add_subplot(gs[1, 2:3])
ring_gauge(ax_g3, (eda['csat_mean']/5)*100,
           "CSAT score", "1–5 scale, normalised", AMBER)

# Hourly session pulse (sparkline)
ax_pulse = fig.add_subplot(gs[1, 3:6])
panel(ax_pulse, "TRAFFIC PULSE  ·  sessions per hour of day",
      "sessions aggregated across the 2-week data window")
hours = turns.groupby(turns["timestamp"].dt.hour)["session_id"].nunique()
hours = hours.reindex(range(24), fill_value=0)
xs = np.arange(24)
ax_pulse.fill_between(xs, hours.values, color=CYAN, alpha=0.18)
ax_pulse.plot(xs, hours.values, color=CYAN, linewidth=2.2)
peak = int(hours.idxmax())
ax_pulse.scatter([peak],[hours.iloc[peak]],
                 s=60, color=AMBER, zorder=5,
                 edgecolor=BG, linewidth=1.5)
# put peak annotation to the LEFT of the point if peak is on the right side
if peak > 16:
    ax_pulse.annotate(f"peak {peak:02d}:00 · {int(hours.iloc[peak])} sessions",
                      xy=(peak, hours.iloc[peak]),
                      xytext=(peak-6, hours.iloc[peak]-3),
                      color=AMBER, fontsize=8.5, fontweight="bold",
                      ha="left")
else:
    ax_pulse.annotate(f"peak {peak:02d}:00 · {int(hours.iloc[peak])} sessions",
                      xy=(peak, hours.iloc[peak]),
                      xytext=(peak+1.2, hours.iloc[peak]-3),
                      color=AMBER, fontsize=8.5, fontweight="bold")
ax_pulse.set_xticks([0,4,8,12,16,20,23])
ax_pulse.set_xticklabels(["00","04","08","12","16","20","23"], fontsize=8)
ax_pulse.set_yticks([]); ax_pulse.tick_params(colors=INK_MUTE)
ax_pulse.set_xlim(-0.4, 23.4)
ax_pulse.set_ylim(0, hours.max()*1.25)
for s in ("top","right","left"): ax_pulse.spines[s].set_visible(False)
ax_pulse.spines["bottom"].set_color(CARD_EDGE)

# ---------------------------------------------------------------------------
# ROW 3 (rows 2): funnel (left 3 cols) + top intents (right 3 cols)
# ---------------------------------------------------------------------------
ax_fun = fig.add_subplot(gs[2, 0:3])
panel(ax_fun, "SESSION FUNNEL  ·  where users drop off",
      "each bar = # sessions reaching that stage")
ax_fun.set_facecolor(CARD)
ax_fun.set_xlim(0, 1); ax_fun.set_ylim(0, 1)
funnel = summary["funnel"]
stages = list(funnel.keys())
values = [funnel[s] for s in stages]
labels_clean = [s.split(". ",1)[1] for s in stages]
colors_f = [CYAN, "#38bdf8", VIOLET, GREEN]
# vertical band 0.15 .. 0.87 for bars
y_top, y_bot = 0.82, 0.20
n = len(stages)
for i,(v,lab,col) in enumerate(zip(values, labels_clean, colors_f)):
    y = y_top - (y_top - y_bot) * (i / (n-1))
    w_frac = 0.55 * v / max(values)
    ax_fun.add_patch(FancyBboxPatch(
        (0.28, y-0.05), w_frac, 0.10,
        boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=0, facecolor=col, alpha=0.85,
        transform=ax_fun.transAxes))
    pct = v/eda['n_sessions']*100
    ax_fun.text(0.28 + w_frac + 0.01, y,
                f"{v}   ({pct:.0f}%)",
                va="center", fontsize=10, color=INK, fontweight="bold",
                transform=ax_fun.transAxes)
    ax_fun.text(0.26, y, lab, va="center", ha="right",
                fontsize=10, color=INK_DIM,
                transform=ax_fun.transAxes)
ax_fun.set_xticks([]); ax_fun.set_yticks([])
for s in ax_fun.spines.values(): s.set_visible(False)

# Top intents — headroom for title, floor for caption
ax_int = fig.add_subplot(gs[2, 3:6])
panel(ax_int, "TOP 10 REQUESTED INTENTS  ·  volume",
      "ranked by turn count in the 2-week window")
ax_int.set_facecolor(CARD)
top10 = turns["true_intent"].value_counts().head(10)
y2 = np.arange(len(top10))[::-1]
for i,(name,val) in enumerate(zip(top10.index, top10.values)):
    # remap y so it fits between ~0.12 (caption) and ~0.9 (title)
    y_scaled = 0.15 + (y2[i] / (len(top10)-1)) * 0.72
    ax_int.add_patch(FancyBboxPatch((0.20, y_scaled-0.025), val/top10.max()*0.72, 0.05,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=0, facecolor=CYAN, alpha=0.85,
        transform=ax_int.transAxes))
    ax_int.text(0.20 + val/top10.max()*0.72 + 0.01, y_scaled, f"{val}",
                va="center", ha="left",
                color=INK, fontsize=9, fontweight="bold", fontfamily=MONO,
                transform=ax_int.transAxes)
    ax_int.text(0.18, y_scaled, name.replace("_"," "),
                va="center", ha="right", fontsize=9, color=INK_DIM,
                transform=ax_int.transAxes)
ax_int.set_xticks([]); ax_int.set_yticks([])
for s in ax_int.spines.values(): s.set_visible(False)

# ---------------------------------------------------------------------------
# ROW 4 (rows 3): fallback by persona | CSAT bars | latency CDF
# ---------------------------------------------------------------------------
ax_a = fig.add_subplot(gs[3, 0:2])
panel(ax_a, "FALLBACK RATE BY PERSONA", "higher = model struggling with this segment")
ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
per = pd.read_csv(AN / "tables" / "segmentation_by_persona.csv", index_col=0)
data = per["fallback_rate"].mul(100).sort_values(ascending=False)
y_top, y_bot = 0.80, 0.22
n = len(data)
for i,(name,val) in enumerate(zip(data.index, data.values)):
    y = y_top - (y_top - y_bot) * (i / (n-1))
    col = RED if val >= 8 else AMBER if val >= 6 else GREEN
    w_frac = 0.55 * val / max(data.values)
    ax_a.add_patch(FancyBboxPatch((0.25, y-0.045), w_frac, 0.09,
        boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=0, facecolor=col, alpha=0.85,
        transform=ax_a.transAxes))
    ax_a.text(0.25 + w_frac + 0.01, y, f"{val:.1f}%", va="center",
              color=INK, fontsize=9, fontweight="bold", fontfamily=MONO,
              transform=ax_a.transAxes)
    ax_a.text(0.23, y, name, va="center", ha="right",
              fontsize=9.5, color=INK_DIM,
              transform=ax_a.transAxes)
ax_a.set_xticks([]); ax_a.set_yticks([])
for s in ax_a.spines.values(): s.set_visible(False)

ax_b = fig.add_subplot(gs[3, 2:4])
panel(ax_b, "CSAT DISTRIBUTION  ·  per turn", "1 = worst, 5 = best")
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
counts = [(turns["csat"]==i).sum() for i in range(1,6)]
csat_cols = [RED, "#fb923c", AMBER, "#84cc16", GREEN]
max_c = max(counts)
n = 5
x_left, x_right = 0.10, 0.95
bar_w = (x_right - x_left) / n * 0.62
y_bar_bot, y_bar_top = 0.20, 0.80
for i,(c,val,col) in enumerate(zip(range(1,6), counts, csat_cols)):
    cx = x_left + (x_right - x_left) * (i + 0.5) / n
    h = (y_bar_top - y_bar_bot) * val / max_c
    ax_b.add_patch(FancyBboxPatch((cx - bar_w/2, y_bar_bot), bar_w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=0, facecolor=col, alpha=0.85,
        transform=ax_b.transAxes))
    ax_b.text(cx, y_bar_bot + h + 0.03, str(val), ha="center",
              color=INK, fontsize=10, fontweight="bold", fontfamily=MONO,
              transform=ax_b.transAxes)
    ax_b.text(cx, y_bar_bot - 0.04, str(c), ha="center", va="top",
              color=INK_DIM, fontsize=9, transform=ax_b.transAxes)
ax_b.set_xticks([]); ax_b.set_yticks([])
for s in ax_b.spines.values(): s.set_visible(False)

ax_c = fig.add_subplot(gs[3, 4:6])
panel(ax_c, "RESPONSE-TIME CDF", "share of turns responded within t (ms)")
# Use an inset so panel title / caption stay clear
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
inner = inset_axes(ax_c, width="88%", height="64%", loc="center",
                   bbox_to_anchor=(0.02, -0.02, 1, 1),
                   bbox_transform=ax_c.transAxes, borderpad=0)
rt = np.sort(turns["response_time_ms"].values)
cdf = np.arange(1, len(rt)+1)/len(rt)*100
inner.fill_between(rt, cdf, color=CYAN, alpha=0.15)
inner.plot(rt, cdf, color=CYAN, linewidth=2.2)
for pct, col in [(50, INK_DIM), (95, RED)]:
    v = int(np.percentile(turns["response_time_ms"], pct))
    # find y value on the curve
    y_at = float(np.interp(v, rt, cdf))
    inner.axvline(v, color=col, linestyle="--", linewidth=1, alpha=0.8)
    inner.scatter([v],[y_at], s=25, color=col, zorder=5,
                  edgecolor=BG, linewidth=1)
    inner.text(v+80, y_at-6 if pct==50 else y_at-6,
               f"p{pct} = {v} ms", color=col, fontsize=8,
               fontweight="bold")
inner.set_xlabel("response time (ms)", color=INK_DIM, fontsize=8.5)
inner.set_ylabel("% of turns", color=INK_DIM, fontsize=8.5)
inner.tick_params(colors=INK_MUTE, labelsize=7.5)
inner.set_facecolor(CARD)
inner.set_ylim(0, 103)
inner.spines["bottom"].set_color(CARD_EDGE)
inner.spines["left"].set_color(CARD_EDGE)
for s in ("top","right"): inner.spines[s].set_visible(False)
inner.grid(color=GRID, linewidth=0.7, alpha=0.6)

# ---------------------------------------------------------------------------
# ROW 5 (rows 4): confusion top-5 offenders | co-occurrence rules table |
#                 LTV bars
# ---------------------------------------------------------------------------
# Top intents driving fallback
ax_x = fig.add_subplot(gs[4, 0:2])
panel(ax_x, "TOP 8 INTENTS DRIVING FALLBACK",
      "% of that intent's turns that fell back")
ax_x.set_xlim(0, 1); ax_x.set_ylim(0, 1)
fb_by = pd.read_csv(AN / "tables" / "fallback_by_intent.csv", index_col=0).head(8)
y_top, y_bot = 0.80, 0.18
n = len(fb_by)
max_v = fb_by["fallback_rate_%"].max()
for i,(name,val) in enumerate(zip(fb_by.index, fb_by["fallback_rate_%"].values)):
    y = y_top - (y_top - y_bot) * (i / (n-1))
    col = RED if val >= 40 else AMBER
    w_frac = 0.50 * val / max_v
    ax_x.add_patch(FancyBboxPatch((0.30, y-0.028), w_frac, 0.056,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=0, facecolor=col, alpha=0.85,
        transform=ax_x.transAxes))
    ax_x.text(0.30 + w_frac + 0.01, y, f"{val:.1f}%",
              va="center", color=INK, fontsize=9,
              fontweight="bold", fontfamily=MONO,
              transform=ax_x.transAxes)
    ax_x.text(0.28, y, name.replace("_"," "),
              va="center", ha="right", fontsize=9, color=INK_DIM,
              transform=ax_x.transAxes)
ax_x.set_xticks([]); ax_x.set_yticks([])
for s in ax_x.spines.values(): s.set_visible(False)

# Co-occurrence rules table
ax_r = fig.add_subplot(gs[4, 2:4])
panel(ax_r, "TOP CO-OCCURRENCE RULES",
      "min support 2% · sorted by lift · airport passenger journeys")
rules = pd.read_csv(AN / "tables" / "cooccurrence_rules.csv").head(7)
# Table drawn manually for full control
ax_r.text(0.03, 0.86, "IF USER ASKS", fontsize=8.5, color=AMBER,
          transform=ax_r.transAxes, fontweight="bold")
ax_r.text(0.38, 0.86, "THEN ALSO",   fontsize=8.5, color=AMBER,
          transform=ax_r.transAxes, fontweight="bold")
ax_r.text(0.68, 0.86, "SUP", fontsize=8.5, color=AMBER,
          transform=ax_r.transAxes, fontweight="bold")
ax_r.text(0.78, 0.86, "CONF", fontsize=8.5, color=AMBER,
          transform=ax_r.transAxes, fontweight="bold")
ax_r.text(0.90, 0.86, "LIFT", fontsize=8.5, color=AMBER,
          transform=ax_r.transAxes, fontweight="bold")
row_y = 0.78
for _, row in rules.iterrows():
    ax_r.text(0.03, row_y, row["antecedent"].replace("_"," "), fontsize=9,
              color=INK, transform=ax_r.transAxes)
    ax_r.text(0.36, row_y, "→", fontsize=10, color=INK_DIM,
              transform=ax_r.transAxes)
    ax_r.text(0.38, row_y, row["consequent"].replace("_"," "), fontsize=9,
              color=INK, transform=ax_r.transAxes)
    ax_r.text(0.68, row_y, f"{row['support']:.2f}", fontsize=9,
              color=INK_DIM, transform=ax_r.transAxes, fontfamily=MONO)
    ax_r.text(0.78, row_y, f"{row['confidence']:.2f}", fontsize=9,
              color=INK_DIM, transform=ax_r.transAxes, fontfamily=MONO)
    lift = row["lift"]
    lc = GREEN if lift >= 2.5 else CYAN if lift >= 1.8 else INK_DIM
    ax_r.text(0.90, row_y, f"{lift:.2f}×", fontsize=9,
              color=lc, transform=ax_r.transAxes,
              fontweight="bold", fontfamily=MONO)
    row_y -= 0.10

# LTV bars grouped — use an inset so we keep the panel chrome
ax_l = fig.add_subplot(gs[4, 4:6])
panel(ax_l, "LTV DRIVERS BY PERSONA",
      "repeat sessions × CSAT × completion → LTV score")
from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _ins
l_inner = _ins(ax_l, width="92%", height="62%", loc="center",
               bbox_to_anchor=(0.02, -0.02, 1, 1),
               bbox_transform=ax_l.transAxes, borderpad=0)
ltv = pd.read_csv(AN / "tables" / "ltv_by_persona.csv", index_col=0)
xp = np.arange(len(ltv))
w = 0.20
l_inner.bar(xp - 1.5*w, ltv["avg_sessions"], w, color=CYAN,   label="sessions/user")
l_inner.bar(xp - 0.5*w, ltv["avg_csat"],     w, color=AMBER,  label="CSAT")
l_inner.bar(xp + 0.5*w, ltv["completion"]/ltv["users"], w, color=GREEN,  label="completions/user")
l_inner.bar(xp + 1.5*w, ltv["avg_ltv"]/2,   w, color=VIOLET, label="LTV ÷2")
l_inner.set_xticks(xp)
l_inner.set_xticklabels(ltv.index, color=INK_DIM, fontsize=8.5)
l_inner.tick_params(axis="y", colors=INK_MUTE, labelsize=7.5)
l_inner.set_facecolor(CARD)
l_inner.spines["bottom"].set_color(CARD_EDGE)
l_inner.spines["left"].set_color(CARD_EDGE)
for s in ("top","right"): l_inner.spines[s].set_visible(False)
l_inner.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
l_inner.legend(loc="upper right", ncol=4, frameon=False,
               fontsize=7.5, labelcolor=INK_DIM,
               bbox_to_anchor=(1.0, 1.18))

# ---------------------------------------------------------------------------
# ROW 6 (rows 5): frustration example strip
# ---------------------------------------------------------------------------
ax_f = fig.add_subplot(gs[5, 0:6])
panel(ax_f, "FRUSTRATION SCENARIO  ·  extracted from a worst-CSAT session",
      "NLP frustration detector: repetition markers + negative sentiment keywords")
frust = pd.read_csv(AN / "tables" / "frustration_examples.csv")
# pick the session with the MOST turns (richer narrative) among the worst
sid_counts = frust.groupby("session_id").size().sort_values(ascending=False)
first_sid = sid_counts.index[0]
scene = frust[frust["session_id"] == first_sid].head(4)
# if scene is very short, fall back to any 4 rows from the whole log for that user
if len(scene) < 3:
    all_turns = pd.read_csv(LOGS / "conversation_logs.csv")
    scene = all_turns[all_turns["session_id"] == first_sid].head(4)
step_x_positions = np.linspace(0.02, 0.98, len(scene)+1)
for i, (_, r) in enumerate(scene.iterrows()):
    x0 = step_x_positions[i]
    x1 = step_x_positions[i+1] - 0.01
    w_ = x1 - x0
    # step card
    ax_f.add_patch(FancyBboxPatch((x0, 0.18), w_, 0.68,
        transform=ax_f.transAxes,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        linewidth=1.0, edgecolor=CARD_EDGE,
        facecolor="#0d1730", zorder=2, clip_on=False))
    # step number badge
    ax_f.add_patch(Circle((x0+0.024, 0.78), 0.028, transform=ax_f.transAxes,
                          color=AMBER, zorder=3, clip_on=False))
    ax_f.text(x0+0.024, 0.78, f"{i+1}", ha="center", va="center",
              transform=ax_f.transAxes, fontsize=9, fontweight="bold", color=BG)
    ax_f.text(x0+0.06, 0.79, f"turn {r['turn_index']}",
              transform=ax_f.transAxes, fontsize=8.5,
              color=INK_DIM, fontweight="bold")
    utt = r["utterance"]
    utt = (utt[:110] + "…") if len(utt) > 110 else utt
    ax_f.text(x0+0.014, 0.62, f'"{utt}"',
              transform=ax_f.transAxes, fontsize=9, color=INK,
              wrap=True, style="italic")
    fb = "FALLBACK" if r["is_fallback"] else f"→ {r['predicted_intent']}"
    col = RED if r["is_fallback"] or r["predicted_intent"]!=r["true_intent"] else GREEN
    ax_f.text(x0+0.014, 0.36, f"true: {r['true_intent']}",
              transform=ax_f.transAxes, fontsize=8, color=INK_DIM)
    ax_f.text(x0+0.014, 0.28, f"pred: {fb}",
              transform=ax_f.transAxes, fontsize=8, color=col, fontweight="bold")
    ax_f.text(x0+0.014, 0.20, f"CSAT {r['csat']}/5",
              transform=ax_f.transAxes, fontsize=8, color=AMBER, fontweight="bold")
    # arrow between cards
    if i < len(scene)-1:
        ax_f.annotate("", xy=(step_x_positions[i+1]-0.005, 0.5),
                      xytext=(x1+0.001, 0.5),
                      xycoords=ax_f.transAxes,
                      arrowprops=dict(arrowstyle="->", color=INK_DIM, lw=1.2))
ax_f.set_xticks([]); ax_f.set_yticks([])
ax_f.set_xlim(0,1); ax_f.set_ylim(0,1)
for s in ax_f.spines.values(): s.set_visible(False)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
fig.text(0.035, 0.014,
         "Source: simulated conversation logs generated from real KB + intent dataset  •  "
         f"n = {eda['n_turns']} turns  •  {eda['n_sessions']} sessions  •  {eda['n_users']} users",
         fontsize=8.5, color=INK_MUTE)
fig.text(0.965, 0.014,
         "Airport Multimodal Chatbot — Analytics & Optimisation, MSc AI 2026",
         fontsize=8.5, color=INK_MUTE, ha="right")

out_path = OUT / "dashboard.png"
fig.savefig(out_path, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Dashboard saved: {out_path}")
