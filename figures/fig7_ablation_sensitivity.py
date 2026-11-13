import argparse

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from figures.style import plt, results, save

ap = argparse.ArgumentParser(description="Figure 7.")
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/figures/fig7_ablation_sensitivity.pdf")
ARGS = ap.parse_args()
AB = results(ARGS.results, "table1")["ablations"]["cells"]
SE = results(ARGS.results, "sensitivity")["cells"]
DSS = ["jshc", "escphc", "panda", "ur5e", "hopper"]
rows_a = [("afterwm_fast_only", "w/o slow correction"), ("afterwm_slow_only", "w/o fast correction"),
          ("afterwm_no_error_input", "w/o error inputs"), ("afterwm_gamma1", r"$\gamma$ fixed to 1")]
A = np.array([[float("%.1f" % AB[ds][v]["pct"]) for ds in DSS] for v, _ in rows_a])
SET = ["$C$=32", "default", "$C$=128", r"$\gamma$=0.05", r"$\gamma$=0.5", r"$\gamma$=1"]
SKEY = ["context_32", "default", "context_128", "gamma_0.05", "gamma_0.5", "gamma_1"]
MOD = [("rebal", "ReBAL"), ("mts3", "MTS3"), ("mlp", "MLP"), ("tdmpc2", "TD-MPC2"), ("tdmpc2_h40", "TD-MPC2-40")]
FULL = {("mts3", "context_32"), ("mts3", "context_128")}


def reduction(s, m):
    v = SE[s][m]["pct"]
    return -(v if (m, s) in FULL else float("%.1f" % v))


Bm = np.array([[reduction(s, m) for s in SKEY] for m, _ in MOD])
nondef = np.array([[reduction(s, m) for s in SKEY if s != "default"] for m, _ in MOD])

W, H = 6.875, 1.12
fig = plt.figure(figsize=(W, H))
reds = LinearSegmentedColormap.from_list("r", ["#fff3ec", "#f4a582", "#d6604d", "#b2182b"])
blues = plt.get_cmap("Blues")
ax = fig.add_axes([1.20 / W, (0.10 + 0.62 * 5 / 4 - 0.62) / H, 2.15 / W, 0.62 / H])
ax.imshow(A, cmap=reds, vmin=0, vmax=60, aspect="auto")
for i in range(4):
    for j in range(5):
        ax.text(j, i, "+%.1f" % A[i, j], ha="center", va="center", fontsize=6.4, color="white" if A[i, j] > 36 else "black")
ax.set_xticks(range(5)); ax.set_xticklabels(["JSHC", "ESCP-HC", "Panda", "UR5e", "Hopper"], fontsize=6.3)
ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.5)
ax.set_yticks(range(4)); ax.set_yticklabels([r for _, r in rows_a], fontsize=6.3)
for sp in ax.spines.values(): sp.set_visible(False)
fig.text(0.04 / W, 0.98, "(a) Ablations (retrained): error increase over AFTER-WM (%)", fontsize=6.9, va="top")
bx = fig.add_axes([4.32 / W, 0.10 / H, 2.50 / W, 0.62 / H * 5 / 4])
Bshow = np.where(np.isfinite(Bm), Bm, 0)
bx.imshow(Bshow, cmap=blues, vmin=0, vmax=45, aspect="auto")
for i in range(5):
    for j in range(6):
        v = Bm[i, j]
        bx.text(j, i, "--" if not np.isfinite(v) else "%.1f" % v, ha="center", va="center", fontsize=6.4,
                color="white" if np.isfinite(v) and v > 27 else "black")
bx.set_xticks(range(6)); bx.set_xticklabels(SET, fontsize=6.3)
bx.xaxis.tick_top(); bx.tick_params(length=0, pad=1.5)
bx.set_yticks(range(5)); bx.set_yticklabels([n for _, n in MOD], fontsize=6.3)
for sp in bx.spines.values(): sp.set_visible(False)
fig.text(3.58 / W, 0.98, "(b) Sensitivity on ESCP-HC: AFTER-WM's error reduction (%)", fontsize=6.9, va="top")
save(fig, ARGS.out, {"ablation": A, "sens_reduction": Bm, "settings": SET, "models": [n for _, n in MOD],
                     "min_nondefault": float(np.nanmin(nondef)), "max_nondefault": float(np.nanmax(nondef))})
