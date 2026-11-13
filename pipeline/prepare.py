from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import io
import json
import socket
import sys
import tempfile
import time

import numpy as np

from pipeline import data as D
from pipeline import evaluate as E
from pipeline import registry as P
from pipeline import run_io as R

SPLITS = ("val", "test", "test_b")


def stats_bytes(stats) -> bytes:
    with tempfile.TemporaryDirectory(dir=P.PREP_DIR) as td:
        p = os.path.join(td, "s.json")
        stats.save(p)
        return open(p, "rb").read()


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def write_once(path: str, payload: bytes, check_only: bool) -> str:
    new_sha = R.sha256_bytes(payload)
    if os.path.exists(path):
        old_sha = R.sha256_file(path)
        if old_sha != new_sha:
            raise RuntimeError(f"{path} exists with sha {old_sha} but recomputed {new_sha}; refusing to overwrite")
        return "identical"
    if check_only:
        raise RuntimeError(f"{path} missing (check mode)")
    R.atomic_write_bytes(path, payload)
    os.chmod(path, 0o444)
    return "written"


def build_all() -> dict:
    files = {}
    stats = D.train_stats()
    files["TRAIN_STATS.json"] = stats_bytes(stats)
    n_train = int(D.load_visible("train")[0].shape[0])
    sched_info = {}
    for seed in P.SEEDS:
        ep, tt = D.make_schedule(seed, n_train)
        files[f"SCHEDULE_{seed}.npz"] = npz_bytes(episode=ep, t=tt, seed=np.asarray(seed, np.int64))
        sched_info[str(seed)] = {"episode_sha256": R.array_sha256(ep), "t_sha256": R.array_sha256(tt),
                                 "shape": list(ep.shape), "t_min": int(tt.min()), "t_max": int(tt.max())}
    dup = {}
    for split in SPLITS:
        m = E.query_manifest(split)
        files[f"QUERY_MANIFEST_{split}.npz"] = npz_bytes(**{k: np.asarray(v) for k, v in m.items()})
        dup[split] = {"queries": int(len(m["e"])), "parents": int(len(np.unique(m["parent"]))),
                      "manifest_sha256": str(m["manifest_sha256"]),
                      **{f"unique_in_H{h}": int(len(np.unique(m[f"gid_in_H{h}"]))) for h in P.HORIZONS},
                      **{f"unique_full_H{h}": int(len(np.unique(m[f"gid_full_H{h}"]))) for h in P.HORIZONS}}
    common = np.asarray(stats.obs_std, np.float64)
    manifest = {
        "schema": "afterwm_prepared_v1",
        "files_sha256": {k: R.sha256_bytes(v) for k, v in sorted(files.items())},
        "data_sha256": {s: R.sha256_file(D.split_path(s)) for s in ("train", "val", "test", "test_b")},
        "expected_data_sha256": P.SPLIT_SHA256,
        "stats_sha256": stats.sha256(), "raw_scale": float(np.mean(common ** 2)),
        "obs_std": stats.obs_std, "schedules": sched_info, "query_manifests": dup,
        "protocol": {"CONTEXT": P.CONTEXT, "MAX_H": P.MAX_H, "HORIZONS": list(P.HORIZONS),
                     "EVAL_STARTS": list(P.EVAL_STARTS), "TRAIN_T_MIN": P.TRAIN_T_MIN,
                     "TRAIN_T_MAX": P.TRAIN_T_MAX, "SEEDS": list(P.SEEDS), "BATCH": P.BATCH,
                     "MAX_STEPS": P.MAX_STEPS},
    }
    files["PREPARE_MANIFEST.json"] = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Write the prepared query manifests, training schedules and TRAIN statistics of a dataset.', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only; never write")
    P.add_cli_options(ap)
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(P.PREP_DIR, exist_ok=True)
    try:
        files = build_all()
        status = {name: write_once(os.path.join(P.PREP_DIR, name), payload, args.check)
                  for name, payload in files.items()}
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        return 2
    for name in sorted(files):
        print(f"{R.sha256_file(os.path.join(P.PREP_DIR, name))}  {name}  [{status[name]}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
