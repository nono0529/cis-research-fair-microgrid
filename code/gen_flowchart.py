#!/usr/bin/env python3
"""Compact 4-step methodology flowchart — 2-line desc, no overflow."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY   = "#081B79"
WHITE  = "#FFFFFF"
DGRAY  = "#2D2D2D"
BLUE   = "#1565C0"
TEAL   = "#007B7F"
BG_BOX = "#FFFFFF"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 250})

fig, ax = plt.subplots(figsize=(8.5, 6.2))
ax.set_xlim(0, 8.5)
ax.set_ylim(0, 6.2)
ax.set_aspect("equal")
ax.axis("off")

BOX_W = 7.6
BOX_H = 1.10
GAP = 0.16
X0 = 0.45
CX = X0 + BOX_W / 2
Y0 = 5.80  # top of step 1

def draw_step(bot, num, title, line1, line2, color):
    x = X0
    ax.add_patch(FancyBboxPatch((x, bot), BOX_W, BOX_H, boxstyle="round,pad=0.06",
                   facecolor=BG_BOX, edgecolor=color, linewidth=1.8, zorder=3))
    # Left bar
    ax.add_patch(FancyBboxPatch((x+0.04, bot+0.06), 0.08, BOX_H-0.12,
                   boxstyle="round,pad=0.03", facecolor=color, edgecolor="none", zorder=4))
    # Number
    ax.add_patch(plt.Circle((x+0.48, bot+BOX_H/2), 0.22, facecolor=color, edgecolor="none", zorder=5))
    ax.text(x+0.48, bot+BOX_H/2, str(num), ha="center", va="center",
            fontsize=8, fontweight="bold", color=WHITE, zorder=6)
    # Title
    ax.text(x+0.90, bot+BOX_H-0.16, title, ha="left", va="top",
            fontsize=9, fontweight="bold", color=color, zorder=5)
    # Two description lines
    ax.text(x+0.90, bot+0.48, line1, ha="left", va="center", fontsize=7.2, color=DGRAY, zorder=5)
    ax.text(x+0.90, bot+0.20, line2, ha="left", va="center", fontsize=7.2, color=DGRAY, zorder=5)

def arrow(yt, yb):
    ax.annotate("", xy=(CX, yb+0.01), xytext=(CX, yt-0.01),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.0), zorder=2)

# ── 4 steps, each 2 lines, ~65 chars max per line ──
steps = [
    (1, "Data Acquisition", [
        "NASA POWER API → hourly GHI & ambient temperature · full year (8760 h) · free, no key",
        "Coordinates: Gansu 35.0°N 104.0°E  |  Cameroon 5.0°N 12.0°E"],
     NAVY),
    (2, "PV + Battery System Modeling", [
        "PV:  P = P_rated × (GHI/1000) × [1 − α_T·(T_cell − 25°C)] × η_derate   |   η = 20% · derating = 0.85",
        "Battery:  SOC tracking with hourly energy balance · η_rt = 92% · DoD_max = 80% · SOC_min = 20%"],
     BLUE),
    (3, "Rural Load Profile Simulation", [
        "200 households × 3.0 kWh/day · peak 19:00–20:00 (1.6× mean) · seasonal variation ±15%",
        "Synthetic profile validated against published rural electricity survey data (Zhou et al., 2018)"],
     BLUE),
    (4, "Multi-Objective PSO Optimization", [
        "Decision variables: PV [20–300 kWp]  &  Battery [50–800 kWh]   |   Constraint: LPSP ≤ 5%",
        "pyswarm solver · 30 particles · 50 iterations · 5 weight combinations → Pareto front tracing"],
     TEAL),
]

for i, (num, title, lines, color) in enumerate(steps):
    top = Y0 - i * (BOX_H + GAP)
    bot = top - BOX_H
    draw_step(bot, num, title, lines[0], lines[1], color)
    if i < len(steps) - 1:
        arrow(bot, bot - GAP)

# ── Title ──────────────────────────────────────
ax.text(X0, 6.10, "METHODOLOGY — Data-to-Optimization Pipeline", ha="left", va="center",
        fontsize=11, fontweight="bold", color=NAVY)

# ── Footer note ────────────────────────────────
ax.text(X0, Y0 - 4*(BOX_H+GAP) - BOX_H - 0.20,
        "100% open-source stack: Python · numpy · pyswarm · matplotlib   |   All data from free public sources (NASA POWER, IRENA, Lazard)",
        ha="left", va="top", fontsize=6.5, color="#999999", style="italic")

out = r"C:\Users\25982\Desktop\CIS RESEARCH FAIR\code\figures\fig_methodology_pipeline.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG_BOX, edgecolor="none", pad_inches=0.10)
plt.close(fig)
print(f"Saved: {out}")
