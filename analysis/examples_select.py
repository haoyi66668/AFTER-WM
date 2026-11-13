from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from afterwm_stats import aggregate as AG
from afterwm_stats import load as L

CHANGE = ("friction", "gain", "payload")
SPLIT_FILE = {"test": "test", "test_b": "testB"}
SEED = 20260910
ROBOTS = ("panda", "ur5e")


def change_mask(arrays, keys=CHANGE):
    ch = np.zeros(arrays[keys[0]].shape[:2], bool)
    for k in keys:
        x = arrays[k].reshape(arrays[k].shape[0], arrays[k].shape[1], -1)
        ch[:, 1:] |= (x[:, 1:] != x[:, :-1]).any(-1)
    return ch


def load_split(data_root, ds, split):
    with np.load(os.path.join(data_root, ds, f"{ds}_{SPLIT_FILE[split]}.npz"), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def compared_model(qs):
    pct = {b: AG.compare(qs, "afterwm", b)["pct"] for b in L.COMPARED}
    return max(L.COMPARED, key=lambda b: pct[b])


def select(ds, data_root, outputs=None, seed=SEED):
    qs = L.per_query(ds, outputs)
    comp = compared_model(qs)
    z = np.load(os.path.join(L.outputs_root(outputs), "per_query", f"{ds}.npz"), allow_pickle=False)
    rows = []
    data = {}
    for s in L.SPLITS:
        data[s] = load_split(data_root, ds, s)
        ch = change_mask(data[s])
        canon = np.asarray(z[f"{s}/canonical"], bool)
        qid = qs.query_id[s]
        e = np.array([int(q.split(":")[1]) for q in qid])
        t = np.array([int(q.split(":")[2]) for q in qid])
        grp = np.array([ch[ei, max(1, ti - 20):ti + 10].any() for ei, ti in zip(e, t)])
        Ft, Fc = qs.runs["afterwm"][seed][s]["ms"], qs.runs[comp][seed][s]["ms"]
        for i in np.flatnonzero(canon & np.isfinite(Ft) & np.isfinite(Fc)):
            rows.append({"split": s, "idx": int(i), "query_id": str(qid[i]), "e": int(e[i]), "t0": int(t[i]),
                         "group_a": bool(grp[i]), "multi_step": {"afterwm": float(Ft[i]), "compared": float(Fc[i])},
                         "d": float(Fc[i] - Ft[i])})
    B = sorted(rows, key=lambda r: (-r["d"], L.SPLITS.index(r["split"]), r["e"], r["t0"]))
    A = [r for r in B if r["group_a"]]
    k = math.ceil(len(A) / 2)
    M = dict(A[k - 1])
    M["rank"], M["n_candidates"] = k, len(A)
    arr = data[M["split"]]
    chg = np.flatnonzero(change_mask(arr)[M["e"], :M["t0"]])
    c = int(chg[-1])
    M["steps_since_change"] = M["t0"] - c
    M["change"] = {}
    for key in CHANGE:
        x = arr[key][M["e"]]
        M["change"][key] = {"before": np.asarray(x[c - 1], np.float64).tolist(), "after": np.asarray(x[c], np.float64).tolist()}
    return {"dataset": ds, "compared": comp, "seed": seed, "selected": M}


def facts(sel, ex):
    out = {}
    for ds in sel:
        fr = ex[ds]["frames"]
        early = [fr[f"{r}|k{k}"]["mean_cm"] for r in ("afterwm", "compared") for k in (1, 5)]
        out[ds] = {"compared": sel[ds]["compared"], "query_id": sel[ds]["selected"]["query_id"],
                   "rank": sel[ds]["selected"]["rank"], "n_candidates": sel[ds]["selected"]["n_candidates"],
                   "steps_since_change": sel[ds]["selected"]["steps_since_change"],
                   "change": sel[ds]["selected"]["change"],
                   "max_mean_link_cm_steps_1_5": max(early),
                   "step40_cm": {"afterwm": fr["afterwm|k40"]["mean_cm"], "compared": fr["compared|k40"]["mean_cm"]},
                   "step40_label": {"afterwm": fr["afterwm|k40"]["label"], "compared": fr["compared|k40"]["label"]},
                   "afterwm_less_than_half": bool(fr["afterwm|k40"]["mean_cm"] < fr["compared|k40"]["mean_cm"] / 2),
                   "matches_released_selection": sel[ds]["selected"]["query_id"] == ex[ds]["selected"]["query_id"]
                   and sel[ds]["compared"] == ex[ds]["compared"]}
    return out


def load_examples(outputs=None):
    out = {}
    for ds in ROBOTS:
        with open(os.path.join(L.outputs_root(outputs), "examples", f"{ds}.json")) as f:
            out[ds] = json.load(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Robot-arm example selection (median case of the queries with a change near "
                                             "the prediction start) and the example facts.")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sel = {ds: select(ds, a.data_root, a.outputs) for ds in ROBOTS}
    res = {"selection": sel, "facts": facts(sel, load_examples(a.outputs))}
    L.write_json(a.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
