import argparse

import numpy as np

from figures.style import plt, results, save

ap = argparse.ArgumentParser(description="Figure 6.")
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="results/figures/fig6_three_designs.pdf")
A = ap.parse_args()
M = results(A.results, "mechanism")
PS = results(A.results, "per_step")
DS = ["jshc", "escphc", "panda", "ur5e", "hopper"]
DN = ["JSHC", "ESCP-HC", "Panda", "UR5e", "Hopper"]
ERR, FAST, SLOW, ROUTE = "#c62828", "#e8743b", "#2e9e5b", "#d4609a"
W, H = 6.875, 1.43
fig = plt.figure(figsize=(W, H))
B0, PH = 0.25 / H, 0.80 / H
TT = (0.25 + 0.80 + 0.04) / H
GT = (0.25 + 0.80 + 0.18) / H
summary = {}

def ax_at(x0, w):
    return fig.add_axes([x0 / W, B0, w / W, PH])

def errbars(axis, x, v, lo, hi, horizontal=False):
    v, lo, hi = np.array(v), np.array(lo), np.array(hi)
    if horizontal:
        axis.errorbar(v, x, xerr=[v - lo, hi - v], fmt="none", ecolor="0.25", elinewidth=0.5, capsize=0.8, capthick=0.5)
    else:
        axis.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="0.25", elinewidth=0.5, capsize=0.8, capthick=0.5)

ax = ax_at(0.31, 1.52)
items = [("no_correction", "no corrections", "0.15"), ("zero_errors", "zeroed", "0.55"),
         ("delay_4", "delayed 4 steps", "#f08a7e"), ("other_sequence", "other sequence's", ERR)]
w = 0.19
for j, (key, lab, col) in enumerate(items):
    r = [M["interventions"]["cells"][ds][key] for ds in DS]
    v = [x["pct"] for x in r]; lo = [x["pct_ci95"][0] for x in r]; hi = [x["pct_ci95"][1] for x in r]
    xx = np.arange(5) + (j - 1.5) * w
    ax.bar(xx, v, w, color=col, label=lab, lw=0)
    errbars(ax, xx, v, lo, hi)
    summary[key] = {"value": v, "lo": lo, "hi": hi}
ax.axhline(0, color="0.3", lw=0.5)
ax.set_xticks(range(5)); ax.set_xticklabels([n.replace("ESCP-HC", "ESCP-\nHC") for n in DN], fontsize=5.8, linespacing=0.95)
ax.set_ylabel("error increase (%)", fontsize=6.3, labelpad=1)
ax.set_ylim(-5, 170); ax.set_yticks([0, 50, 100, 150])
ax.legend(loc="upper left", fontsize=6.2, frameon=False, ncol=1, handlelength=1.0, borderaxespad=0.2, labelspacing=0.18)
ax.spines[["top", "right"]].set_visible(False)
fig.text((0.31 + 1.52 / 2) / W, TT, "(a) Role of errors altered", fontsize=6.8, ha="center", va="bottom")

bx = ax_at(2.22, 1.44)
bars = [("no_fast", "step1", "no fast, step 1", FAST, 1.0), ("no_fast", "step40", "no fast, step 40", FAST, 0.35),
        ("no_slow", "step1", "no slow, step 1", SLOW, 0.35), ("no_slow", "step40", "no slow, step 40", SLOW, 1.0)]
w = 0.19
for j, (rk, kk, lab, col, al) in enumerate(bars):
    v = [M["removal"]["cells"][ds][f"{rk}/{kk}"][0] for ds in DS]
    lo = [M["removal"]["cells"][ds][f"{rk}/{kk}"][1] for ds in DS]; hi = [M["removal"]["cells"][ds][f"{rk}/{kk}"][2] for ds in DS]
    xx = np.arange(5) + (j - 1.5) * w
    bx.bar(xx, v, w, color=col, alpha=al, label=lab, lw=0)
    errbars(bx, xx, v, lo, hi)
    summary[rk + "_" + kk] = {"value": v, "lo": lo, "hi": hi}
bx.axhline(0, color="0.3", lw=0.5)
bx.set_xticks(range(5)); bx.set_xticklabels([n.replace("ESCP-HC", "ESCP-\nHC") for n in DN], fontsize=5.8, linespacing=0.95)
bx.set_ylim(-5, 112); bx.set_yticks([0, 20, 40, 60, 80])
bx.set_ylabel("error increase (%)", fontsize=6.3, labelpad=1)
bx.legend(loc="upper left", fontsize=5.9, frameon=False, ncol=2, handlelength=0.8, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.15, columnspacing=0.5)
bx.spines[["top", "right"]].set_visible(False)
fig.text((2.22 + 1.44 / 2) / W, TT, "(b) One correction removed", fontsize=6.8, ha="center", va="bottom")

cx = ax_at(4.14, 1.16)
COL = {"jshc": "0.1", "escphc": "#7b5ea7", "panda": "#d1495b", "ur5e": "#2d7dd2", "hopper": "#9c6b30"}
steps = np.arange(1, 41)
for ds, dn in zip(DS, DN):
    c = PS["datasets"][ds]["crossing"]; p = np.array(c["pct"]); lo, hi = np.array(c["pct_ci"][0]), np.array(c["pct_ci"][1])
    cx.fill_between(steps, lo, hi, color=COL[ds], alpha=0.12, lw=0)
    cx.plot(steps, p, color=COL[ds], lw=0.9, label=dn)
    summary["crossing_" + ds] = {"value": p, "lo": lo, "hi": hi}
cx.axhline(0, color="0.3", lw=0.5)
cx.set_xlim(1, 40); cx.set_xticks([1, 10, 20, 30, 40])
cx.set_ylim(-32, 65)
cx.set_xlabel("prediction step $h$", fontsize=6.3, labelpad=0.5)
cx.set_ylabel("slow-only $-$ fast-only\n(% of AFTER-WM)", fontsize=6.3, labelpad=1)
cx.text(39.3, 8, "fast-only better", fontsize=6.0, ha="right", va="bottom", color="0.3")
cx.text(39.3, -31, "slow-only better", fontsize=6.0, ha="right", va="bottom", color="0.3")
cx.legend(loc="upper right", fontsize=6.0, frameon=False, ncol=2, handlelength=1.0, borderaxespad=0.2, labelspacing=0.15, columnspacing=0.6)
cx.spines[["top", "right"]].set_visible(False)
fig.text((4.14 + 1.16 / 2) / W, TT, "(c) Retrained with one correction", fontsize=6.8, ha="center", va="bottom")

dx = ax_at(5.98, 0.84)
r = [M["routing"]["cells"][ds] for ds in DS]
v = [x["value"] for x in r]; lo = [x["ci95"][0] for x in r]; hi = [x["ci95"][1] for x in r]
yy = np.arange(5)[::-1]
dx.barh(yy, v, 0.55, color=ROUTE, lw=0)
errbars(dx, yy, v, lo, hi, horizontal=True)
summary["route"] = {"value": v, "lo": lo, "hi": hi}
dx.axvline(0, color="0.3", lw=0.5)
dx.set_yticks(yy); dx.set_yticklabels(DN, fontsize=6.0)
dx.set_xlim(-2, 42); dx.set_xticks([0, 20, 40])
dx.set_xlabel("error increase (%)", fontsize=6.3, labelpad=0.5)
dx.spines[["top", "right"]].set_visible(False)
dx.tick_params(axis="y", length=0, pad=1.5)
fig.text((5.98 + 0.84 / 2 - 0.20) / W, TT, "(d) Routing w/o errors", fontsize=6.8, ha="center", va="bottom")

def header(x0, x1, text):
    fig.text(((x0 + x1) / 2) / W, GT, text, fontsize=7.0, ha="center", va="bottom", fontweight="bold")
    fig.add_artist(plt.Line2D([x0 / W, x1 / W], [GT - 0.012, GT - 0.012], color="0.45", lw=0.6))
header(0.05, 1.83, "Predictive correction")
header(1.93, 5.30, "Dual-timescale adaptation")
header(5.45, 6.84, "Multiple internal models")
save(fig, A.out, summary)
