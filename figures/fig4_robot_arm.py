import argparse
import json
import os

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from figures.crop_frames import frames
from figures.style import plt, save

ap = argparse.ArgumentParser(description="Figure 4.")
ap.add_argument("--outputs", default="outputs")
ap.add_argument("--out", default="results/figures/fig4_robot_arm.pdf")
A = ap.parse_args()
J = {}
for ds in ("panda", "ur5e"):
    with open(os.path.join(A.outputs, "examples", f"{ds}.json")) as f:
        J[ds] = json.load(f)
F = frames(A.outputs)
KS = [5, 10, 20, 40]
W, H = 6.875, 1.27
fig = plt.figure(figsize=(W, H))
blocks = [("panda", "(a) Panda: AFTER-WM vs. MOLe", ["AFTER-WM", "MOLe"], 0.012), ("ur5e", "(b) UR5e: AFTER-WM vs. ReBAL", ["AFTER-WM", "ReBAL"], 0.508)]
fw, gap = 0.0805, 0.0035
plot = {}
for ds, title, rows, x0 in blocks:
    a0 = F[(ds, "afterwm", 5)]; asp = a0.shape[0] / a0.shape[1]
    fh = fw * W * asp / H
    lx = x0 + 0.018
    top = 0.82
    fig.text(x0, 0.995, title, fontsize=6.8, ha="left", va="top")
    for j, k in enumerate(KS):
        fig.text(lx + j * (fw + gap) + fw / 2, 0.905, "step %d" % k, fontsize=6.3, ha="center", va="top")
    vmax = J[ds]["vmax_cm"]
    for i, (row, lab) in enumerate(zip(["afterwm", "compared"], rows)):
        y0 = top - (i + 1) * fh - i * 0.075
        fig.text(x0 + 0.006, y0 + fh / 2, lab, fontsize=6.0, rotation=90, ha="center", va="center",
                 fontweight="bold" if i == 0 else "normal")
        for j, k in enumerate(KS):
            ax = fig.add_axes([lx + j * (fw + gap), y0, fw, fh])
            ax.imshow(F[(ds, row, k)], interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.4); sp.set_color("#8a8a8a")
            label = J[ds]["frames"][f"{row}|k{k}"]["label"]
            plot[f"{ds}|{row}|k{k}"] = {"label": label, "frame_sha_input": int(F[(ds, row, k)].astype(np.int64).sum())}
            fig.text(lx + j * (fw + gap) + fw / 2, y0 - 0.012, label,
                     fontsize=6.0, ha="center", va="top", fontweight="bold" if k == 40 else "normal")
    cax = fig.add_axes([lx + 4 * (fw + gap) + 0.002, top - 2 * fh - 0.075, 0.0075, 2 * fh + 0.075])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="viridis"), cax=cax)
    ticks = J[ds]["vmin_cm"] + np.array([0, vmax / 3, 2 * vmax / 3, vmax]).round(0) if ds == "ur5e" else [0, 5, 10, 15, 20]
    cb.set_ticks(ticks)
    plot[f"{ds}|colorbar"] = {"vmax": vmax, "ticks": list(map(float, ticks))}
    cb.ax.tick_params(labelsize=5.4, length=1.2, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("link deviation (cm)", fontsize=5.6, labelpad=1.5)
save(fig, A.out, plot)
