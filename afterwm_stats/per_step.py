from __future__ import annotations

import argparse

import numpy as np

from afterwm_stats import bootstrap as BT
from afterwm_stats import load as L

K = 40
YCAP = 10.0


def five(e):
    return float(np.mean([e[:h].mean() for h in BT.HS]))


def curves(steps, models):
    out = {}
    for m in models:
        seeds = sorted(steps[m])
        A = np.stack([np.concatenate([steps[m][sd][s] for s in L.SPLITS]).mean(0) for sd in seeds])
        f5 = np.array([five(A[i]) for i in range(A.shape[0])])
        out[m] = {"seeds": seeds, "mean": A.mean(0), "sd": A.std(0, ddof=1), "multi_step_seed": f5,
                  "step40_seed": A[:, K - 1].copy()}
    return out


def ylim(cv):
    lo = min(float(c["mean"].min()) for c in cv.values()) / 1.15
    top = max(float(c["mean"].max()) for c in cv.values())
    hi = YCAP if top > YCAP else top * 1.2
    clipped = {m: int(np.argmax(c["mean"] > YCAP)) + 1 for m, c in cv.items() if c["mean"].max() > YCAP}
    return [lo, hi], clipped


def comparisons(qs, steps):
    out = {}
    for b in L.COMPARED:
        seeds = sorted(set(steps["afterwm"]) & set(steps[b]))
        r = BT.paired_curve({sd: steps["afterwm"][sd] for sd in seeds}, {sd: steps[b][sd] for sd in seeds}, L.SPLITS,
                            qs.draws, seeds)
        out[b] = {"pct": r["pct"], "pct_ci": r["pct_ci"], "seeds": seeds}
    return out


def crossing(qs, steps):
    seeds = sorted(set(steps["afterwm_slow_only"]) & set(steps["afterwm_fast_only"]) & set(steps["afterwm"]))
    r = BT.paired_curve({sd: steps["afterwm_slow_only"][sd] for sd in seeds}, {sd: steps["afterwm_fast_only"][sd] for sd in seeds},
                        L.SPLITS, qs.draws, seeds, vref={sd: steps["afterwm"][sd] for sd in seeds})
    return {"pct": r["pct"], "pct_ci": r["pct_ci"], "seeds": seeds}


def dataset(ds, root=None):
    qs = L.per_query(ds, root)
    steps = L.per_step(ds, root)
    cv = curves(steps, ("afterwm",) + L.COMPARED)
    lim, clipped = ylim(cv)
    strongest = min(L.COMPARED, key=lambda b: cv[b]["multi_step_seed"].mean())
    return {"curves": cv, "ylim": lim, "clipped": clipped, "comparisons": comparisons(qs, steps),
            "strongest": strongest, "crossing": crossing(qs, steps)}


def summary(res):
    n = lower = ci_below = 0
    not_lower = {}
    lowest_cells = {}
    from_step = {}
    k40 = {}
    for ds, r in res.items():
        afterwm = r["curves"]["afterwm"]["mean"]
        others = np.stack([r["curves"][b]["mean"] for b in L.COMPARED])
        lowest_cells[ds] = int((afterwm < others.min(0)).sum())
        for b, c in r["comparisons"].items():
            p, hi = np.asarray(c["pct"]), np.asarray(c["pct_ci"])[1]
            n += len(p)
            lower += int((p < 0).sum())
            ci_below += int((hi < 0).sum())
            bad = [int(k) for k in np.flatnonzero(p >= 0) + 1]
            if bad:
                not_lower.setdefault(ds, {})[b] = bad
        c = r["comparisons"][r["strongest"]]
        hi = np.asarray(c["pct_ci"])[1]
        above = np.flatnonzero(hi >= 0)
        from_step[ds] = int(above.max()) + 2 if len(above) else 1
        k40[ds] = float(-np.asarray(c["pct"])[K - 1])
    first_all_lowest = max(int(np.max(np.flatnonzero(
        np.asarray(res[ds]["curves"]["afterwm"]["mean"]) >= np.stack([res[ds]["curves"][b]["mean"] for b in L.COMPARED]).min(0)),
        initial=-1)) + 2 for ds in res)
    cross = {ds: {"step1": float(r["crossing"]["pct"][0]), "step40": float(r["crossing"]["pct"][K - 1])} for ds, r in res.items()}
    return {"comparisons": n, "afterwm_lower": lower, "ci_below_zero": ci_below, "points_not_lower": not_lower,
            "afterwm_lowest_cells": lowest_cells, "afterwm_lowest_from_step": first_all_lowest,
            "vs_strongest_ci_below_zero_from_step": max(from_step.values()), "vs_strongest_from_step": from_step,
            "vs_strongest_step40_reduction_pct": k40, "vs_strongest_step40_range_pct": [min(k40.values()), max(k40.values())],
            "clipped": {ds: r["clipped"] for ds, r in res.items() if r["clipped"]},
            "crossing": cross, "crossing_step1_range_pct": [min(v["step1"] for v in cross.values()),
                                                          max(v["step1"] for v in cross.values())],
            "crossing_step40_range_pct": [min(-v["step40"] for v in cross.values()), max(-v["step40"] for v in cross.values())]}


def run(root=None, datasets=L.DATASETS):
    res = {ds: dataset(ds, root) for ds in datasets}
    return {"datasets": res, "summary": summary(res)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-step errors and per-step comparisons (Fig. 3, Fig. 6c).")
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    L.write_json(a.out, run(a.outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
