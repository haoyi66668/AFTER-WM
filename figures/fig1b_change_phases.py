import argparse

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from figures.style import NAMES, plt, results, save

ap = argparse.ArgumentParser(description="Figure 1b.")
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/figures/fig1b_change_phases.pdf")
A = ap.parse_args()
fig = plt.figure(figsize=(2.9, 1.47))

PH = results(A.results, "phases")["summary"]
order = ["JSHC", "Panda", "UR5e", "Hopper"]
keys = ["jshc", "panda", "ur5e", "hopper"]
cols = ["1–4", "5–19", "20–59", "≥60", "none\nbefore $t$"]
vals = np.zeros((4, 5)); names = [[""] * 5 for _ in range(4)]; seeds = np.zeros((4, 5), int)
allneg = PH["afterwm_lower"] == PH["cells"]
for i, ds in enumerate(keys):
    for p, c in enumerate(PH["fig1b"][ds]):
        vals[i, p] = c["min_reduction_pct"]; names[i][p] = NAMES[c["closest"]]; seeds[i, p] = c["seeds_afterwm_lower"]
bx = fig.add_axes([0.150, 0.235, 0.705, 0.60])
cmap = LinearSegmentedColormap.from_list("b", ["#eef3fb", "#2f6db5"])
im = bx.imshow(vals, cmap=cmap, vmin=0, vmax=40, aspect="auto")
for i in range(4):
    for p in range(5):
        c = "white" if vals[i, p] > 26 else "#1a1a1a"
        bx.text(p, i - 0.18, "%.1f" % vals[i, p], ha="center", va="center", fontsize=6.5, color=c, fontweight="bold")
        bx.text(p, i + 0.26, names[i][p], ha="center", va="center", fontsize=5.8, color=c)
bx.set_xticks(range(5)); bx.set_xticklabels(cols, fontsize=5.9)
bx.set_yticks(range(4)); bx.set_yticklabels(order, fontsize=6.3)
bx.set_xlabel("steps since the latest change, at the prediction start", fontsize=6.0, labelpad=1.0)
bx.tick_params(length=0, pad=1.5)
for sp in bx.spines.values(): sp.set_visible(False)
cax = fig.add_axes([0.875, 0.235, 0.022, 0.60])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6.0, length=1.2, pad=1); cb.outline.set_linewidth(0.4)
fig.text(0.02, 0.985, "(b) % lower error than every compared model, all phases", fontsize=6.9, ha="left", va="top")
save(fig, A.out, {"b_min_reduction_pct": vals, "b_closest_model": names, "b_seeds_AFTERWM_lower": seeds,
                  "b_all_point_estimates_negative": bool(allneg)})
