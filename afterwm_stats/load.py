from __future__ import annotations

import json
import os

import numpy as np

from afterwm_stats import bootstrap as BT

DATASETS = ("jshc", "escphc", "panda", "ur5e", "hopper")
SPLITS = ("test", "test_b")
SEEDS = (20260910, 20270913, 20280916, 20290919, 20300922)
COMPARED = ("rssm", "tdmpc2", "tdmpc2_h40", "mlp", "grbal", "rebal", "mole", "mbcd", "gpmm", "mts3")
ABLATIONS = ("afterwm_slow_only", "afterwm_fast_only", "afterwm_no_error_input", "afterwm_gamma1")
NAME = {"rssm": "DreamerV3-style RSSM", "tdmpc2": "TD-MPC2", "tdmpc2_h40": "TD-MPC2-40", "mlp": "MLP", "grbal": "GrBAL",
        "rebal": "ReBAL", "mole": "MOLe", "mbcd": "MBCD", "gpmm": "GPMM", "mts3": "MTS3", "afterwm": "AFTER-WM (ours)"}
SHORT = {**NAME, "rssm": "RSSM", "afterwm": "AFTER-WM"}
DATASET_NAME = {"jshc": "JSHC", "escphc": "ESCP-HC", "panda": "Panda", "ur5e": "UR5e", "hopper": "Hopper", "alfa": "ALFA"}


def outputs_root(root=None):
    return root or os.environ.get("AFTERWM_OUTPUTS") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


class QuerySet:
    def __init__(self, path):
        z = _npz(path)
        self.parents = {s: z[f"{s}/parent"].astype(str) for s in ("val",) + SPLITS}
        self.query_id = {s: z[f"{s}/query_id"].astype(str) for s in SPLITS}
        self.runs = {}
        for k, v in z.items():
            parts = k.split("/")
            if len(parts) == 4:
                m, sd, s, key = parts
                self.runs.setdefault(m, {}).setdefault(int(sd), {}).setdefault(s, {})[key] = np.asarray(v, np.float64)
        self._draws = None

    @property
    def draws(self):
        if self._draws is None:
            self._draws = BT.make_draws(self.parents)
        return self._draws

    def seeds(self, *models):
        out = None
        for m in models:
            s = set(self.runs.get(m, {}))
            out = s if out is None else out & s
        return sorted(out or [])

    def values(self, model, key, seeds=None):
        seeds = seeds or self.seeds(model)
        return {sd: {s: self.runs[model][sd][s][key] for s in SPLITS} for sd in seeds}


def per_query(name, root=None):
    return QuerySet(os.path.join(outputs_root(root), "per_query", f"{name}.npz"))


def per_step(ds, root=None):
    z = _npz(os.path.join(outputs_root(root), "per_step", f"{ds}.npz"))
    out = {}
    for k, v in z.items():
        m, sd, s = k.split("/")
        out.setdefault(m, {}).setdefault(int(sd), {})[s] = np.asarray(v, np.float64)
    return out


def labels(ds, root=None):
    z = _npz(os.path.join(outputs_root(root), "labels", f"labels_{ds}.npz"))
    out = {}
    for k, v in z.items():
        s, key = k.split("|", 1)
        out.setdefault(s, {})[key] = v
    return out


def analysis(name, root=None):
    return _npz(os.path.join(outputs_root(root), "analysis", f"{name}.npz"))


def analysis_json(name, root=None):
    with open(os.path.join(outputs_root(root), "analysis", f"{name}.json")) as f:
        return json.load(f)


def nested(z, depth):
    out = {}
    for k, v in z.items():
        parts = k.split("/")
        if len(parts) != depth:
            continue
        d = out
        for p in parts[:-1]:
            d = d.setdefault(int(p) if p.isdigit() else p, {})
        d[parts[-1]] = np.asarray(v)
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))
