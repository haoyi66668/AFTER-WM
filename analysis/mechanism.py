from __future__ import annotations

import argparse

import numpy as np

from afterwm_stats import bootstrap as BT
from afterwm_stats import load as L

INTERVENTIONS = ("no_correction", "zero_errors", "delay_4", "other_sequence")
REMOVAL_DATASETS = ("panda", "ur5e")


def _by(z, ds, key):
    n = L.nested(z, 4)
    return {sd: {s: n[ds][sd][s][key].astype(np.float64) for s in L.SPLITS} for sd in sorted(n.get(ds, {}))}


def interventions(root=None):
    z = L.analysis("interventions", root)
    out = {}
    for ds in L.DATASETS:
        qs = L.per_query(ds, root)
        base = _by(z, ds, "intact")
        seeds = sorted(base)
        out[ds] = {}
        for v in INTERVENTIONS:
            r = BT.paired(_by(z, ds, v), base, L.SPLITS, qs.draws, seeds)
            out[ds][v] = {"pct": r["pct"], "pct_ci95": r["pct_ci95"], "pct_seed_values": r["pct_seed_values"], "seeds": seeds}
    rng = {v: [min(out[ds][v]["pct"] for ds in out), max(out[ds][v]["pct"] for ds in out)] for v in INTERVENTIONS}
    worst = all(max(INTERVENTIONS, key=lambda v: out[ds][v]["pct"]) == "other_sequence" for ds in out)
    return {"cells": out, "range_pct": rng, "other_sequence_largest_everywhere": bool(worst),
            "all_positive": bool(min(r[0] for r in rng.values()) > 0)}


def _removal_rows(root):
    z = L.nested(L.analysis("removal", root), 5)
    return z


def removal(root=None):
    z = _removal_rows(root)
    res = {}
    for ds in L.DATASETS:
        qs = L.per_query(ds, root)
        draws = BT.make_draws({s: qs.parents[s] for s in L.SPLITS})
        seeds = sorted(z[ds])
        val = lambda v, key: {sd: {s: z[ds][sd][s][v][key].astype(np.float64) for s in L.SPLITS} for sd in seeds}
        res[ds] = {"seeds": seeds}
        for v in ("no_fast", "no_slow"):
            for key, name in (("h1", "step1"), ("h40", "step40")):
                p = BT.pct_seedwise(val(v, key), val("intact", key), L.SPLITS, draws, seeds)
                b = p["draw"][np.isfinite(p["draw"])]
                res[ds][f"{v}/{name}"] = [p["pct"], float(np.quantile(b, .025)), float(np.quantile(b, .975))]
        if ds in REMOVAL_DATASETS:
            lab = L.labels(ds, root)
            clean = {s: ~lab[s]["change_in_first_40"].astype(bool) for s in L.SPLITS}
            groups = {"early": {s: (lab[s]["phase"].astype(str) == "1-4") & clean[s] for s in L.SPLITS},
                      "late": {s: np.isin(lab[s]["phase"].astype(str), (">=60", "none")) & clean[s] for s in L.SPLITS}}
            for v in ("no_fast", "no_slow"):
                for g, w in groups.items():
                    p = BT.pct_seedwise(val(v, "ms"), val("intact", "ms"), L.SPLITS, draws, seeds,
                                        {s: w[s].astype(np.float64) for s in L.SPLITS})
                    res[ds][f"{v}/{g}"] = {"pct": p["pct"], "n_q": p["n_q"],
                                           "ci95": [float(np.nanquantile(p["draw"], .025)), float(np.nanquantile(p["draw"], .975))]}
    s = {k: [min(res[ds][k][0] for ds in L.DATASETS), max(res[ds][k][0] for ds in L.DATASETS)]
         for k in ("no_fast/step1", "no_fast/step40", "no_slow/step1", "no_slow/step40")}
    s["no_fast/step40_max_abs"] = max(abs(res[ds]["no_fast/step40"][0]) for ds in L.DATASETS)
    s["no_slow/step1_max_abs"] = max(abs(res[ds]["no_slow/step1"][0]) for ds in L.DATASETS)
    s["by_phase"] = {ds: {k: round(res[ds][k]["pct"], 10) for k in ("no_fast/early", "no_fast/late", "no_slow/early", "no_slow/late")}
                     for ds in REMOVAL_DATASETS}
    return {"cells": res, "summary": s}


def routing(root=None):
    z = L.nested(L.analysis("routing", root), 4)
    res = {}
    for ds in L.DATASETS:
        qs = L.per_query(ds, root)
        draws = BT.make_draws({s: qs.parents[s] for s in L.SPLITS})
        seeds = sorted(z[ds])
        a = {sd: {s: z[ds][sd][s]["frozen_routing"].astype(np.float64) for s in L.SPLITS} for sd in seeds}
        b = {sd: {s: z[ds][sd][s]["intact"].astype(np.float64) for s in L.SPLITS} for sd in seeds}
        p = BT.pct_seedwise(a, b, L.SPLITS, draws, seeds)
        res[ds] = BT.verdict(p["pct"], p["seed"], p["draw"], +1)
        res[ds]["seeds"] = seeds
    v = [res[ds]["value"] for ds in res]
    return {"cells": res, "range_pct": [min(v), max(v)], "all_positive": bool(min(v) > 0)}


def beyond_range(root=None):
    z = L.analysis("beyond_range", root)
    lab = L.labels("escphc_beyond", root)
    parents = {s: z[f"{s}/parent"].astype(str) for s in L.SPLITS}
    draws = BT.make_draws(parents)
    rows = L.nested({k: v for k, v in z.items() if k.count("/") == 3}, 4)
    w = {s: lab[s]["beyond_range"].astype(np.float64) for s in L.SPLITS}
    afterwm = rows["afterwm"]
    res = {}
    for b in L.COMPARED:
        seeds = sorted(set(afterwm) & set(rows[b]))
        va = {sd: {s: afterwm[sd][s]["ms"].astype(np.float64) for s in L.SPLITS} for sd in seeds}
        vb = {sd: {s: rows[b][sd][s]["ms"].astype(np.float64) for s in L.SPLITS} for sd in seeds}
        p = BT.pct_seedwise(va, vb, L.SPLITS, draws, seeds, w, finite_pairs=True)
        res[b] = BT.verdict(p["pct"], p["seed"], p["draw"], -1)
        res[b]["n_queries"] = p["n_q"]
    closest = max(L.COMPARED, key=lambda b: res[b]["value"])
    return {"cells": res, "n_queries": int(sum(w[s].sum() for s in L.SPLITS)),
            "beaten": sum(1 for b in L.COMPARED if res[b]["value"] < 0), "of": len(L.COMPARED), "closest": closest,
            "smallest_margin_pct": -res[closest]["value"]}


def run(root=None):
    return {"interventions": interventions(root), "removal": removal(root), "routing": routing(root),
            "beyond_range": beyond_range(root)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Test-time alterations of the errors, removal of one correction, routing test without errors and "
                                             "the test beyond the training range, from the stored outputs.")
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    L.write_json(a.out, run(a.outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
