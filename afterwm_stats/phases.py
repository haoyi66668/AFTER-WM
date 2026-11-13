from __future__ import annotations

import argparse

import numpy as np

from afterwm_stats import bootstrap as BT
from afterwm_stats import load as L

PHASES = {"escphc": ("G1", "G2", "G3", "G4", "G5")}
DEFAULT = ("1-4", "5-19", "20-59", ">=60", "none")
TABLE_ORDER = ("rssm", "tdmpc2", "tdmpc2_h40", "mts3", "grbal", "rebal", "mole", "mbcd", "gpmm", "mlp")


def phase_names(ds):
    return PHASES.get(ds, DEFAULT)


def dataset(ds, root=None):
    qs = L.per_query(ds, root)
    lab = L.labels(ds, root)
    for s in L.SPLITS:
        if not np.array_equal(lab[s]["query_id"].astype(str), qs.query_id[s]):
            raise ValueError(f"{ds} {s}: label order differs from the query order")
    out = {}
    for b in L.COMPARED:
        seeds = qs.seeds("afterwm", b)
        va, vb = qs.values("afterwm", "ms", seeds), qs.values(b, "ms", seeds)
        out[b] = {}
        for p in phase_names(ds):
            w = {s: (lab[s]["phase"].astype(str) == p).astype(np.float64) for s in L.SPLITS}
            r = BT.paired(va, vb, L.SPLITS, qs.draws, seeds, weights=w)
            out[b][p] = {"pct": r["pct"], "pct_ci95": r["pct_ci95"], "pct_seed_values": r["pct_seed_values"],
                         "seeds_afterwm_lower": int(sum(1 for v in r["pct_seed_values"] if v < 0)), "n_seeds": r["n_seeds"],
                         "n_queries": r["n_queries"]}
    return out


def summary(res):
    cells = [(ds, b, p, c) for ds, r in res.items() for b, d in r.items() for p, c in d.items()]
    fig = {}
    for ds in ("jshc", "panda", "ur5e", "hopper"):
        if ds not in res:
            continue
        row = []
        for p in phase_names(ds):
            shown = [(b, float("%.1f" % res[ds][b][p]["pct"]), res[ds][b][p]["seeds_afterwm_lower"]) for b in TABLE_ORDER]
            b, v, k = max(shown, key=lambda t: t[1])
            row.append({"phase": p, "closest": b, "min_reduction_pct": -v, "seeds_afterwm_lower": k})
        fig[ds] = row
    p1 = [fig[ds][0]["min_reduction_pct"] for ds in fig]
    esc = [max(res["escphc"][b][p]["pct"] for b in L.COMPARED) for p in phase_names("escphc")] if "escphc" in res else []
    return {"cells": len(cells), "afterwm_lower": sum(1 for *_, c in cells if c["pct"] < 0),
            "ci_below_zero": sum(1 for *_, c in cells if c["pct_ci95"][1] < 0),
            "ci_not_below_zero": [f"{ds} {b} {p}" for ds, b, p, c in cells if not c["pct_ci95"][1] < 0],
            "min_seeds_afterwm_lower": min(c["seeds_afterwm_lower"] for *_, c in cells),
            "fig1b": fig, "phase1_min_reduction_pct": min(p1) if p1 else None,
            "escphc_all_phases_positive": bool(esc and max(esc) < 0)}


def run(root=None, datasets=L.DATASETS):
    res = {ds: dataset(ds, root) for ds in datasets}
    return {"cells": res, "summary": summary(res)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Change-phase comparisons (250 cells, Fig. 1b).")
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    L.write_json(a.out, run(a.outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
