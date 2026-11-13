from __future__ import annotations

import argparse

import numpy as np

from afterwm_stats import bootstrap as BT
from afterwm_stats import load as L

METRICS = {"multi_step": "ms", "step40": "h40"}


def model_table(qs, model, seeds=None):
    seeds = seeds or qs.seeds(model)
    out = {"seeds": list(seeds)}
    for name, key in METRICS.items():
        r = BT.model_stat(qs.values(model, key, seeds), L.SPLITS, qs.draws, seeds)
        out[name] = {"mean": r["mean"], "sd": r["sd"], "seed_values": r["seed_values"], "ci95": r["ci95"]}
    return out


def compare(qs, a, b, key="ms", denominator=None, seeds=None):
    if seeds is None:
        seeds = qs.seeds(a, b) if denominator is None else qs.seeds(a, b, denominator)
    ref = None if denominator is None else qs.values(denominator, key, seeds)
    r = BT.paired(qs.values(a, key, seeds), qs.values(b, key, seeds), L.SPLITS, qs.draws, seeds, vref=ref)
    r["seeds"] = list(seeds)
    r["seeds_a_lower"] = int(sum(1 for v in r["seed_values"] if v < 0))
    r["seeds_a_higher"] = int(sum(1 for v in r["seed_values"] if v > 0))
    return r


def main_table(root=None, datasets=L.DATASETS):
    res = {}
    for ds in datasets:
        qs = L.per_query(ds, root)
        rows = {m: model_table(qs, m) for m in ("afterwm",) + L.COMPARED}
        comps = {b: compare(qs, "afterwm", b) for b in L.COMPARED}
        strongest = min(L.COMPARED, key=lambda b: rows[b]["multi_step"]["mean"])
        res[ds] = {"rows": rows, "comparisons": comps, "strongest": strongest}
    return res


def main_summary(res):
    cells = [(ds, b, c) for ds, r in res.items() for b, c in r["comparisons"].items()]
    vs = {ds: res[ds]["comparisons"][res[ds]["strongest"]] for ds in res}
    lowest_both = all(all(r["rows"]["afterwm"][met]["mean"] < r["rows"][b][met]["mean"] for b in L.COMPARED)
                      for r in res.values() for met in METRICS)
    return {"n_comparisons": len(cells),
            "all_seeds_lower": sum(1 for _, _, c in cells if c["seeds_a_lower"] == c["n_seeds"]),
            "ci_below_zero": sum(1 for _, _, c in cells if c["ci95"][1] < 0),
            "afterwm_lowest_multi_step_and_step40_everywhere": bool(lowest_both),
            "strongest": {ds: res[ds]["strongest"] for ds in res},
            "vs_strongest_pct": {ds: vs[ds]["pct"] for ds in res},
            "vs_strongest_seeds": {ds: [vs[ds]["seeds_a_lower"], vs[ds]["n_seeds"]] for ds in res},
            "vs_strongest_range_pct": [min(-v["pct"] for v in vs.values()), max(-v["pct"] for v in vs.values())]}


def alfa(root=None):
    qs = L.per_query("alfa", root)
    tseeds = qs.seeds("afterwm")
    rows = {m: model_table(qs, m, tseeds if m == "afterwm" else qs.seeds(m)) for m in ("afterwm",) + L.COMPARED}
    comps = {b: compare(qs, "afterwm", b) for b in L.COMPARED}
    beaten = [b for b in L.COMPARED if comps[b]["pct"] < 0]
    closest = max(L.COMPARED, key=lambda b: comps[b]["pct"])
    lowest = {met: all(rows["afterwm"][met]["mean"] < rows[b][met]["mean"] for b in L.COMPARED) for met in METRICS}
    return {"rows": rows, "comparisons": comps, "beaten": len(beaten), "of": len(L.COMPARED), "closest": closest,
            "smallest_margin_pct": -comps[closest]["pct"], "closest_seeds": [comps[closest]["seeds_a_lower"],
                                                                          comps[closest]["n_seeds"]],
            "all_beaten_in_all_seeds": all(c["seeds_a_lower"] == c["n_seeds"] for c in comps.values()),
            "afterwm_lowest": lowest, "afterwm_seeds": tseeds}


def ablations(root=None, datasets=L.DATASETS):
    res = {}
    for ds in datasets:
        qs = L.per_query(ds, root)
        res[ds] = {}
        for v in L.ABLATIONS:
            r = compare(qs, v, "afterwm")
            res[ds][v] = {"pct": r["pct"], "pct_ci95": r["pct_ci95"], "seed_pct": r["pct_seed_values"],
                          "n_seeds_pos": r["seeds_a_higher"], "n_seeds": r["n_seeds"]}
    allv = [res[ds][v]["pct"] for ds in res for v in L.ABLATIONS]
    g1 = [res[ds]["afterwm_gamma1"]["pct"] for ds in res]
    return {"cells": res, "all_positive": bool(min(allv) > 0), "range_pct": [min(allv), max(allv)],
            "gamma1_range_pct": [min(g1), max(g1)]}


def single_correction(root=None, datasets=L.DATASETS):
    res = {}
    for ds in datasets:
        qs = L.per_query(ds, root)
        res[ds] = {}
        for v in ("afterwm_fast_only", "afterwm_slow_only"):
            for key, name in (("h1", "step1"), ("h40", "step40")):
                r = compare(qs, v, "afterwm", key=key)
                res[ds][f"{v}/{name}"] = {"pct": r["pct"], "pct_ci95": r["pct_ci95"], "seed_pct": r["pct_seed_values"]}
    f1 = [res[ds]["afterwm_fast_only/step1"]["pct"] for ds in res]
    f40 = [res[ds]["afterwm_fast_only/step40"]["pct"] for ds in res]
    s40 = [res[ds]["afterwm_slow_only/step40"]["pct"] for ds in res]
    return {"cells": res, "fast_only_step1_max_abs_pct": max(abs(x) for x in f1),
            "fast_only_step40_range_pct": [min(f40), max(f40)], "slow_only_step40_range_pct": [min(s40), max(s40)]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Statistics of Table 1, ALFA, the ablations and the models retrained with one correction, from the outputs.")
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    mt = main_table(a.outputs)
    L.write_json(a.out, {"main": mt, "main_summary": main_summary(mt), "alfa": alfa(a.outputs),
                         "ablations": ablations(a.outputs), "single_correction": single_correction(a.outputs)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
