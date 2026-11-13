from __future__ import annotations

import argparse

from afterwm_stats import bootstrap as BT
from afterwm_stats import load as L

MODELS = ("rebal", "mts3", "mlp", "tdmpc2", "tdmpc2_h40")
SETTINGS = (("context_32", "escphc_ctx32", "afterwm"), ("default", "escphc", "afterwm"), ("context_128", "escphc_ctx128", "afterwm"),
            ("gamma_0.05", "escphc", "afterwm_gamma_0.05"), ("gamma_0.5", "escphc", "afterwm_gamma_0.5"),
            ("gamma_1", "escphc", "afterwm_gamma1"))


def cell(qs, a, b):
    seeds = qs.seeds(a, b)
    r = BT.paired(qs.values(a, "ms", seeds), qs.values(b, "ms", seeds), L.SPLITS, qs.draws, seeds)
    return {"pct": r["pct"], "pct_ci95": r["pct_ci95"], "pct_seed_values": r["pct_seed_values"],
            "seeds_afterwm_lower": int(sum(1 for v in r["pct_seed_values"] if v < 0)), "n_seeds": r["n_seeds"]}


def run(root=None):
    sets = {}
    res = {}
    for name, qset, afterwm in SETTINGS:
        qs = sets.setdefault(qset, L.per_query(qset, root))
        res[name] = {b: cell(qs, afterwm, b) for b in MODELS}
    other = [c for name, d in res.items() if name != "default" for c in d.values()]
    return {"cells": res, "n_other": len(other), "all_lower_all_seeds": all(c["seeds_afterwm_lower"] == c["n_seeds"] for c in other),
            "reduction_range_pct": [min(-c["pct"] for c in other), max(-c["pct"] for c in other)]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sensitivity on ESCP-HC (Fig. 7b).")
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    L.write_json(a.out, run(a.outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
