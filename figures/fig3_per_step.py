import argparse

import numpy as np
from matplotlib.lines import Line2D

from figures.style import AFTERWM_BLUE, plt, results, save

ap = argparse.ArgumentParser(description="Figure 3.")
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/figures/fig3_per_step.pdf")
A_ = ap.parse_args()
R = results(A_.results, "per_step")
DS = ["jshc", "escphc", "panda", "ur5e", "hopper"]
DN = {"jshc": "JSHC", "escphc": "ESCP-HC", "panda": "Panda", "ur5e": "UR5e", "hopper": "Hopper"}
COMP = ["rssm", "tdmpc2", "tdmpc2_h40", "mlp", "grbal", "rebal", "mole", "mbcd", "gpmm", "mts3"]
NAME = {"rssm": "RSSM", "tdmpc2": "TD-MPC2", "tdmpc2_h40": "TD-MPC2-40", "mts3": "MTS3", "grbal": "GrBAL",
        "rebal": "ReBAL", "mole": "MOLe", "mbcd": "MBCD", "gpmm": "GPMM", "mlp": "MLP", "afterwm": "AFTER-WM"}
COL = {"rssm": "#7f7f7f", "tdmpc2": "#e6a100", "tdmpc2_h40": "#b07b00", "mts3": "#2ca02c", "grbal": "#9467bd",
       "rebal": "#e377c2", "mole": "#d62728", "mbcd": "#8c564b", "gpmm": "#17becf", "mlp": "#1f3b73"}
DASH = {"rssm": (0, (8, 2)), "tdmpc2": (0, (1, 1.3)), "tdmpc2_h40": (0, (5, 1.6, 1, 1.6)), "mts3": (0, (2.4, 1.4)),
        "grbal": (0, (5, 1.4, 1, 1.4, 1, 1.4)), "rebal": "-", "mole": (0, (10, 2, 3, 2)), "mbcd": (0, (2.6, 1.2, 2.6, 4)),
        "gpmm": (0, (1, 1.3, 1, 1.3, 1, 4.5)), "mlp": (0, (4.5, 3.5))}
h = np.arange(1, 41)

W, H = 6.875, 1.60
fig = plt.figure(figsize=(W, H))
L, Rm, B, T = 0.40, 0.05, 0.56, 0.17
gap, gap6 = 0.25, 0.33
pw = (W - L - Rm - 4 * gap - gap6) / 6.0
ph = H - B - T
plot = {}
for i, ds in enumerate(DS):
    ax = fig.add_axes([(L + i * (pw + gap)) / W, B / H, pw / W, ph / H])
    M = R["datasets"][ds]["curves"]
    for m in COMP:
        y = np.array(M[m]["mean"], float)
        ax.plot(h, y, color=COL[m], lw=0.8, ls=DASH[m])
        plot[f"{ds}/{m}"] = y
    y = np.array(M["afterwm"]["mean"], float)
    ax.plot(h, y, color=AFTERWM_BLUE, lw=1.7, zorder=5)
    plot[f"{ds}/afterwm"] = y
    ax.set_yscale("log")
    lo, hi = R["datasets"][ds]["ylim"]
    plot[f"{ds}/ylim"] = [lo * 0.92, min(hi * 1.15, 10.0)]
    ax.set_ylim(lo * 0.92, min(hi * 1.15, 10.0))
    ax.set_xlim(1, 40); ax.set_xticks([1, 10, 20, 30, 40])
    ax.tick_params(axis="both", labelsize=5.8, pad=1.2)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(alpha=0.22, lw=0.35, which="major")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.set_title(DN[ds], fontsize=6.8, pad=2)
    if i == 0: ax.set_ylabel("nMSE (log scale)", fontsize=6.3, labelpad=1)
    for m, k in R["datasets"][ds]["clipped"].items():
        ax.text(0.04, 0.97, f"{NAME[m]} above\nthe axis from\nstep {k}", transform=ax.transAxes, fontsize=5.6, va="top", ha="left", color="0.25")
        plot[f"{ds}/note"] = [m, k]
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    TICKS = {"jshc": [0.05, 0.1, 0.2, 0.5, 1, 2], "escphc": [0.1, 0.2, 0.5, 1, 2], "panda": [0.03, 0.1, 0.3, 1, 3],
             "ur5e": [0.03, 0.1, 0.3, 1, 3, 10], "hopper": [0.03, 0.1, 0.3, 1, 3]}
    yl = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([t for t in TICKS[ds] if yl[0] <= t <= yl[1]]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: ("%g" % v)))
fig.text((L + 2.5 * pw + 2 * gap) / W, 0.345 / H, "prediction step $h$", ha="center", fontsize=6.6)
handles = [Line2D([0], [0], color=AFTERWM_BLUE, lw=1.7, label=NAME["afterwm"])] + \
          [Line2D([0], [0], color=COL[m], lw=0.8, ls=DASH[m], label=NAME[m]) for m in COMP]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=((L + 2.5 * pw + 2 * gap) / W, -0.025), ncol=6,
           fontsize=5.9, frameon=False, handlelength=2.2, columnspacing=1.0, labelspacing=0.25)

x0 = L + 5 * pw + 4 * gap + gap6
ax = fig.add_axes([x0 / W, B / H, pw / W, ph / H])
RC = {"jshc": "#111111", "escphc": "#6a4bc4", "panda": "#d62728", "ur5e": "#2e7bd6", "hopper": "#9c6b30"}
for ds in DS:
    b = R["datasets"][ds]["strongest"]
    v = R["datasets"][ds]["comparisons"][b]; p = np.array(v["pct"]); lo = np.array(v["pct_ci"][0]); hi = np.array(v["pct_ci"][1])
    ax.fill_between(h, lo, hi, color=RC[ds], alpha=0.13, lw=0)
    ax.plot(h, p, color=RC[ds], lw=0.85, label=DN[ds])
    plot[f"{ds}/strongest"] = b
    plot[f"{ds}/rel"] = [p, lo, hi]
ax.axhline(0, color="0.3", lw=0.5)
ax.set_xlim(1, 40); ax.set_xticks([1, 10, 20, 30, 40]); ax.set_ylim(-62, 12); ax.set_yticks([-60, -40, -20, 0])
ax.tick_params(axis="both", labelsize=5.8, pad=1.2)
ax.grid(alpha=0.22, lw=0.35)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.set_title("AFTER-WM vs. strongest (%)", fontsize=6.8, pad=2, loc="right")
fig.text((x0 + pw / 2) / W, 0.345 / H, "prediction step $h$", ha="center", fontsize=6.6)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=((x0 + pw / 2) / W, -0.025), ncol=2,
           fontsize=5.6, frameon=False, handlelength=1.0, columnspacing=0.5, labelspacing=0.2, handletextpad=0.3)
save(fig, A_.out, plot)
