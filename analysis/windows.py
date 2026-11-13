from __future__ import annotations

import argparse

import numpy as np

from afterwm_stats import bootstrap as BT
from afterwm_stats import load as L

ARMS = ("panda", "ur5e", "hopper")


def test_draws(qs):
    return BT.make_draws({s: qs.parents[s] for s in L.SPLITS})


def window(ds, label, root=None):
    qs = L.per_query(ds, root)
    lab = L.labels(ds, root)
    for s in L.SPLITS:
        if not np.array_equal(lab[s]["query_id"].astype(str), qs.query_id[s]):
            raise ValueError(f"{ds} {s}: label order differs from the query order")
    w = {s: lab[s][label].astype(np.float64) for s in L.SPLITS}
    draws = test_draws(qs)
    out = {}
    for b in L.COMPARED:
        seeds = qs.seeds("afterwm", b)
        p = BT.pct_seedwise(qs.values("afterwm", "ms", seeds), qs.values(b, "ms", seeds), L.SPLITS, draws, seeds, w)
        r = BT.verdict(p["pct"], p["seed"], p["draw"], -1)
        r["n_queries"] = p["n_q"]
        out[b] = r
    return {"n_queries": int(sum(w[s].sum() for s in L.SPLITS)), "res": out}


def summarize(res):
    cells = [(ds, b, r) for ds, d in res.items() for b, r in d["res"].items()]
    worst = max(cells, key=lambda t: t[2]["value"])
    beaten = [b for b in L.COMPARED if all(res[ds]["res"][b]["value"] < 0 for ds in res)]
    return {"beaten": len(beaten), "of": len(L.COMPARED), "smallest_margin_pct": -worst[2]["value"],
            "closest": worst[1], "closest_dataset": worst[0]}


def run(root=None):
    nc = {ds: window(ds, "no_change", root) for ds in L.DATASETS}
    tr = {ds: window(ds, "transient_only", root) for ds in ARMS}
    return {"no_change": nc, "no_change_summary": summarize(nc), "transient": tr, "transient_summary": summarize(tr)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Queries without a change and queries with only transient disturbances.")
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    L.write_json(a.out, run(a.outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
