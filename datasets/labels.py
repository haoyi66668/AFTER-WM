from __future__ import annotations

import argparse
import json
import os

import numpy as np

STARTS = (64, 96, 128, 160, 192)
CONTEXT = 64
HORIZON = 40
TRANSIENT_LOOKBACK = 24
SINCE_BINS = (("1-4", 1, 4), ("5-19", 5, 19), ("20-59", 20, 59), (">=60", 60, 10 ** 9))
PHASES = {"escphc": ("G1", "G2", "G3", "G4", "G5")}
DEFAULT_PHASES = ("1-4", "5-19", "20-59", ">=60", "none")
TRAIN_RANGE = (1.5 ** -3, 1.5 ** 3)
TRAIN_RANGE_TOL = 1e-6
CHANGE_KEYS = {"jshc": ("fail_joint",), "escphc": ("grav_mult", "damp_mult", "mass_mult"),
               "escphc_beyond": ("grav_mult", "damp_mult", "mass_mult")}
ARM_DATASETS = ("panda", "ur5e", "hopper")
SPLIT_TAG = {"test": "TEST", "test_b": "TEST_B"}
SPLIT_FILE = {"train": "train", "val": "val", "test": "test", "test_b": "testB"}
FILE_PREFIX = {"escphc_beyond": "escphc"}
DATASETS = ("jshc", "escphc", "panda", "ur5e", "hopper", "escphc_beyond")


def data_file(data_root, dataset, split):
    return os.path.join(data_root, dataset, f"{FILE_PREFIX.get(dataset, dataset)}_{SPLIT_FILE[split]}.npz")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def query_grid(n_episodes):
    e = np.repeat(np.arange(n_episodes), len(STARTS)).astype(np.int64)
    t = np.tile(np.asarray(STARTS), n_episodes).astype(np.int64)
    return e, t


def query_ids(split, e, t):
    return np.asarray([f"{SPLIT_TAG[split]}:{a}:{b}" for a, b in zip(e.tolist(), t.tolist())])


def change_mask(dataset, arrays):
    keys = CHANGE_KEYS.get(dataset, ("friction", "gain", "payload"))
    ch = np.zeros(arrays[keys[0]].shape, bool)
    for k in keys:
        x = arrays[k]
        ch[:, 1:] |= x[:, 1:] != x[:, :-1]
    if dataset == "jshc":
        stored = arrays["changed"].astype(bool)
        if not np.array_equal(stored, ch):
            raise ValueError("changed is not the change indicator of fail_joint")
    return ch


def since_and_window(ch, e, t, ks=(25, 40)):
    since = np.full(len(e), -1, np.int64)
    win = {k: np.zeros(len(e), bool) for k in ks}
    for i, (a, b) in enumerate(zip(e.tolist(), t.tolist())):
        idx = np.flatnonzero(ch[a, :b])
        if len(idx):
            since[i] = b - int(idx[-1])
        for k in ks:
            win[k][i] = bool(ch[a, b:b + k].any())
    return since, win


def since_bin(since):
    out = np.full(len(since), "none", dtype=object)
    for name, lo, hi in SINCE_BINS:
        out[(since >= lo) & (since <= hi)] = name
    return out.astype(str)


def escphc_phase(arrays, e, t):
    starts = arrays["eval_starts"].astype(np.int64)
    col = np.searchsorted(starts, t)
    if not np.all(starts[np.minimum(col, len(starts) - 1)] == t):
        raise ValueError("query start not in eval_starts")
    g = arrays["since_group"].astype(np.int64)[e, col]
    if not set(np.unique(g).tolist()) <= {1, 2, 3, 4, 5}:
        raise ValueError("since_group outside 1..5")
    return np.asarray([f"G{x}" for x in g]), arrays["since_steps"].astype(np.int64)[e, col]


def window_any(mask, e, t, lo, hi):
    return np.asarray([bool(mask[a, b + lo:b + hi].any()) for a, b in zip(e.tolist(), t.tolist())])


def transient_mask(arrays):
    pulse = (arrays["friction_eff"] != arrays["friction"]) | (arrays["gain_eff"] != arrays["gain"])
    impact = np.any(arrays["impulse"] != 0, axis=-1)
    return pulse | impact


def beyond_range(arrays, e, t):
    lo, hi = TRAIN_RANGE[0] - TRAIN_RANGE_TOL, TRAIN_RANGE[1] + TRAIN_RANGE_TOL
    out = np.zeros(len(e), bool)
    for i, (a, b) in enumerate(zip(e.tolist(), t.tolist())):
        for k in ("grav_mult", "damp_mult", "mass_mult"):
            x = arrays[k][a, b - CONTEXT:b + HORIZON].astype(np.float64)
            if np.any(x < lo) or np.any(x > hi):
                out[i] = True
    return out


def split_labels(dataset, arrays, split):
    n = arrays["obs"].shape[0]
    e, t = query_grid(n)
    out = {"query_id": query_ids(split, e, t), "e": e, "t": t}
    ch = change_mask(dataset, arrays)
    if dataset == "escphc_beyond":
        out["beyond_range"] = beyond_range(arrays, e, t)
        return out
    if dataset == "escphc":
        out["phase"], out["since"] = escphc_phase(arrays, e, t)
    else:
        since, win = since_and_window(ch, e, t)
        out["phase"], out["since"] = since_bin(since), since
        out["change_in_first_25"], out["change_in_first_40"] = win[25], win[40]
    out["no_change"] = ~window_any(ch, e, t, -CONTEXT, HORIZON)
    if dataset in ARM_DATASETS:
        tr = transient_mask(arrays)
        out["transient_only"] = out["no_change"] & window_any(tr, e, t, -TRANSIENT_LOOKBACK, HORIZON)
    return out


def dataset_labels(data_root, dataset, splits=("test", "test_b")):
    return {s: split_labels(dataset, load(data_file(data_root, dataset, s)), s) for s in splits}


def counts(dataset, lab):
    res = {}
    allq = {k: np.concatenate([lab[s][k] for s in lab]) for k in lab[next(iter(lab))] if k not in ("e", "t")}
    res["queries"] = int(len(allq["query_id"]))
    if "phase" in allq:
        names = PHASES.get(dataset, DEFAULT_PHASES)
        res["phase"] = {p: int(np.sum(allq["phase"] == p)) for p in names}
    for k in ("no_change", "transient_only", "beyond_range"):
        if k in allq:
            res[k] = int(allq[k].sum())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analysis-only labels of the test queries (never given to any model).")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for ds in a.datasets.split(","):
        lab = dataset_labels(a.data_root, ds)
        arrays = {f"{s}|{k}": v for s, d in lab.items() for k, v in d.items()}
        path = os.path.join(a.out, f"labels_{ds}.npz")
        if os.path.exists(path):
            raise SystemExit(f"refusing to overwrite {path}")
        np.savez_compressed(path, **arrays)
        summary[ds] = counts(ds, lab)
    with open(os.path.join(a.out, "LABEL_COUNTS.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
