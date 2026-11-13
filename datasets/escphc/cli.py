from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import escp as E
from . import gen as G


def _w(path, obj):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def cmd_collect(a):
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (never overwritten)")
    out.mkdir(parents=True)
    splits = a.splits.split(",")
    env, applier = G.make_env()
    man = {"protocol": G.protocol(a.namespace), "runtime": G.runtime_identity(), "code_files": G.code_files(),
           "pools": G.pools_json(a.namespace), "splits": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for s in splits:
        arrays, audit = G.collect_split(a.namespace, s, env, applier, replay=True, log=lambda m: print(m, flush=True))
        f = out / G.FILE_OF[s]
        G.save_npz_det(f, arrays)
        man["splits"][s] = {"file": f.name, "file_sha256": G.sha256_file(f), "n": int(arrays["obs"].shape[0]),
                            "arrays": {k: G.array_sha256(v) for k, v in sorted(arrays.items())}, "audit": audit}
        print(json.dumps({s: man["splits"][s]["audit"]}), flush=True)
    man["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _w(out / "MANIFEST.json", man)
    print("COLLECT_DONE", out)
    return 0


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cmd_hashes(a):
    d = Path(a.data)
    res = {}
    for s in G.SPLIT_ORDER:
        f = d / G.FILE_OF[s]
        if not f.exists():
            continue
        arr = _load(f)
        res[s] = {"file_sha256": G.sha256_file(f), "arrays": {k: G.array_sha256(v) for k, v in sorted(arr.items())}}
    man = json.loads((d / "MANIFEST.json").read_text()) if (d / "MANIFEST.json").exists() else {}
    res["_pools_sha256"] = hashlib.sha256(json.dumps(man.get("pools"), sort_keys=True).encode()).hexdigest()
    res["_audits"] = {s: v.get("audit") for s, v in (man.get("splits") or {}).items()}
    _w(a.out, res)
    print(json.dumps({s: v["file_sha256"][:16] for s, v in res.items() if not s.startswith("_")}))
    return 0


def cmd_xcheck(a):
    A, B = json.loads(Path(a.a).read_text()), json.loads(Path(a.b).read_text())
    diffs = []
    keys = sorted(set(A) | set(B))
    for s in keys:
        if s == "_audits":
            strip = lambda d: {k: {kk: vv for kk, vv in v.items() if kk != "seconds"} for k, v in (d or {}).items()}
            if strip(A.get(s)) != strip(B.get(s)):
                diffs.append(s)
            continue
        if s.startswith("_"):
            if A.get(s) != B.get(s):
                diffs.append(s)
            continue
        if s not in A or s not in B:
            diffs.append(f"{s}: missing")
            continue
        for k in sorted(set(A[s]["arrays"]) | set(B[s]["arrays"])):
            if A[s]["arrays"].get(k) != B[s]["arrays"].get(k):
                diffs.append(f"{s}.{k}")
        if A[s]["file_sha256"] != B[s]["file_sha256"]:
            diffs.append(f"{s}: file")
    res = {"a": a.a, "b": a.b, "identical": not diffs, "differences": diffs,
           "n_arrays_compared": sum(len(A[s]["arrays"]) for s in keys if not s.startswith("_") and s in A)}
    _w(a.out, res)
    print(json.dumps(res))
    return 0 if not diffs else 4


def cmd_audit(a):
    d = Path(a.data)
    man = json.loads((d / "MANIFEST.json").read_text())
    ns = man["protocol"]["namespace"]
    L = float(man["protocol"]["escp"]["log_scale_limit"])
    if a.log_scale_limit is not None and float(a.log_scale_limit) != L:
        raise SystemExit(f"--log-scale-limit {a.log_scale_limit} != manifest {L}")
    G.set_log_scale_limit(L)
    data = {s: _load(d / G.FILE_OF[s]) for s in G.SPLIT_ORDER}
    hard, desc = {}, {}
    exp = {s: G.SPLITS[s][0] for s in G.SPLIT_ORDER}
    hard["1_sizes_shapes"] = all(data[s]["obs"].shape == (exp[s], G.T + 1, G.OBS_DIM) and data[s]["act"].shape == (
        exp[s], G.T, G.ACT_DIM) and data[s]["obs"].dtype == np.float32 and data[s]["act"].dtype == np.float32
        for s in data)
    hard["2_finite"] = all(np.isfinite(data[s]["obs"]).all() and np.isfinite(data[s]["act"]).all() for s in data)
    hard["3_action_range"] = all(float(np.abs(data[s]["act"]).max()) <= G.ACTION_LIMIT for s in data)
    hard["4_parents"] = all(len(set(data[s]["parent"].tolist())) == exp[s] // G.PARENT_BLOCK and all(
        (np.unique(data[s]["parent"], return_counts=True)[1] == G.PARENT_BLOCK)) for s in data)
    trp, tep = G.pools(ns)
    pj = man["pools"]
    hard["5a_pools_recomputed"] = (G.pools_json(ns) == pj)
    ok5 = True
    for s in data:
        pool = trp if G.SPLITS[s][1] == "train" else tep
        for i in range(exp[s]):
            dd = G.episode_design(ns, s, i)
            lab = G.labels(dd, pool)
            z = data[s]
            if not (z["offset"][i] == dd["offset"] and list(z["task_list"][i]) == dd["task_list"] and z["kind"][i] == dd[
                    "kind"] and z["pool"][i] == dd["pool"] and int(z["plan_seed"][i]) == dd["plan_seed"] and int(
                    z["reset_seed"][i]) == dd["reset_seed"]):
                ok5 = False
            for k, v in lab.items():
                if not np.array_equal(z[k][i], v):
                    ok5 = False
            acts = G.episode_actions(dd)
            if not np.array_equal(z["act"][i], acts[dd["offset"]: dd["offset"] + G.T]):
                ok5 = False
    hard["5b_labels_and_actions_recomputed"] = ok5
    sch = E.Schedule()
    cur = None
    ok6 = True
    for j in range(1000):
        ind = sch.before_transition()
        if ind is not None:
            cur = ind
        if (cur if cur is not None else 0) != E.list_index_of_transition(j):
            ok6 = False
    hard["6_schedule_literal_escp"] = ok6
    hard["7_replay_and_readback"] = all(man["splits"][s]["audit"]["replay_exact"] == exp[s] and man["splits"][s]["audit"][
        "applied_matches_labels"] == exp[s] and man["splits"][s]["audit"]["warnings"] == 0 for s in data)
    hard["8_hashes_match_manifest"] = all(G.sha256_file(d / G.FILE_OF[s]) == man["splits"][s]["file_sha256"] and {
        k: G.array_sha256(v) for k, v in sorted(data[s].items())} == man["splits"][s]["arrays"] for s in data)
    hs = {s: {hashlib.sha256(np.ascontiguousarray(o).tobytes()).hexdigest() for o in data[s]["obs"]} for s in data}
    inter = {f"{x}&{y}": len(hs[x] & hs[y]) for i, x in enumerate(G.SPLIT_ORDER) for y in G.SPLIT_ORDER[i + 1:]}
    within_dupes = {s: exp[s] - len(hs[s]) for s in data}
    tr_set = {json.dumps(t, sort_keys=True) for t in pj["train"]}
    te_set = {json.dumps(t, sort_keys=True) for t in pj["test"]}
    hard["9_disjoint"] = all(v == 0 for v in inter.values()) and all(v == 0 for v in within_dupes.values()) and not (
        tr_set & te_set) and len(tr_set) == 40 and len(te_set) == 40
    std = data["train"]["obs"].reshape(-1, G.OBS_DIM).std(0)
    hard["10_train_std_min"] = bool(std.min() >= 1e-3)
    desc["train_obs_std"] = [float(x) for x in std]
    desc["train_act_std"] = [float(x) for x in data["train"]["act"].reshape(-1, G.ACT_DIM).std(0)]
    desc["content_hash_intersections"] = inter
    for s in data:
        z = data[s]
        pitch = z["obs"][:, :, 1]
        sg = z["since_group"].reshape(-1)
        desc[s] = {
            "n": int(z["obs"].shape[0]),
            "kinds": {k: int(v) for k, v in zip(*np.unique(z["kind"], return_counts=True))},
            "offset_min_med_max": [int(z["offset"].min()), float(np.median(z["offset"])), int(z["offset"].max())],
            "effective_changes_per_window": {int(k): int(v) for k, v in zip(*np.unique(z["chg_eff"].sum(1), return_counts=True))},
            "since_group_counts_G1_G5": [int((sg == g).sum()) for g in range(1, 6)],
            "flipped_fraction_abs_pitch_gt_pi_over_2": float((np.abs(pitch) > np.pi / 2).any(1).mean()),
            "max_abs_obs": float(np.abs(z["obs"]).max()),
            "max_abs_velocity_dims": float(np.abs(z["obs"][:, :, 8:]).max()),
            "x_velocity_mean_sd": [float(z["obs"][:, :, 8].mean()), float(z["obs"][:, :, 8].std())],
            "torso_height_min_max": [float(z["obs"][:, :, 0].min()), float(z["obs"][:, :, 0].max())],
            "mult_ranges": {k: [float(z[k].min()), float(z[k].max())] for k in ("mass_mult", "damp_mult", "grav_mult")},
            "distinct_tasks_used": int(len(np.unique(z["task_id"]))),
        }
        if s != "train":
            lo, hi = np.quantile(data["train"]["obs"].reshape(-1, G.OBS_DIM), [0.001, 0.999], axis=0)
            o = z["obs"].reshape(-1, G.OBS_DIM)
            desc[s]["frac_outside_train_q001_q999"] = float(((o < lo) | (o > hi)).any(1).mean())
    res = {"hard": hard, "all_hard_pass": all(hard.values()), "description": desc, "data": str(d),
           "manifest_sha256": G.sha256_file(d / "MANIFEST.json")}
    _w(a.out, res)
    print(json.dumps({"hard": hard, "all_hard_pass": res["all_hard_pass"]}))
    return 0 if res["all_hard_pass"] else 4


def cmd_probe(a):
    env, applier = G.make_env()
    trp, tep = G.pools(a.namespace)
    stats = {"namespace": a.namespace, "n": a.n, "per_split": {}}
    t0 = time.time()
    for s in ("train", "test"):
        pool = trp if s == "train" else tep
        flips, maxabs, jumps, vx = 0, 0.0, [], []
        for i in range(a.n):
            d = G.episode_design(a.namespace, s, i)
            acts = G.episode_actions(d)
            obs, applied, w = G.simulate(env, applier, d, pool, acts)
            lab = G.labels(d, pool)
            flips += int((np.abs(obs[:, 1]) > np.pi / 2).any())
            maxabs = max(maxabs, float(np.abs(obs).max()))
            vx.append(float(obs[:, 8].mean()))
            d2 = dict(d)
            d2["task_list"] = [d["task_list"][0]] * E.LIST_LEN
            obs_cf, _, _ = G.simulate(env, applier, d2, pool, acts)
            c = int(lab["chg_state"][0])
            if c + 20 <= G.T and lab["chg_eff"][0]:
                jumps.append(float(np.mean((obs[c:c + 20] - obs_cf[c:c + 20]) ** 2)))
        stats["per_split"][s] = {"flipped_fraction": flips / a.n, "max_abs_obs": maxabs,
                                 "vx_mean_of_means": float(np.mean(vx)),
                                 "effect_mse_20_steps_after_first_change_vs_no_change": {
                                     "median": float(np.median(jumps)) if jumps else None,
                                     "q10": float(np.quantile(jumps, 0.1)) if jumps else None,
                                     "q90": float(np.quantile(jumps, 0.9)) if jumps else None, "n": len(jumps)}}
    stats["seconds"] = round(time.time() - t0, 1)
    stats["runtime"] = G.runtime_identity()
    _w(a.out, stats)
    print(json.dumps(stats["per_split"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect"); p.add_argument("--out", required=True); p.add_argument("--namespace", default=G.NAMESPACE)
    p.add_argument("--splits", default=",".join(G.SPLIT_ORDER))
    p.add_argument("--log-scale-limit", type=float, default=3.0)
    p = sub.add_parser("hashes"); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("xcheck"); p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("audit"); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--log-scale-limit", type=float, default=None)
    p = sub.add_parser("probe"); p.add_argument("--namespace", required=True); p.add_argument("--n", type=int, default=24)
    p.add_argument("--out", required=True)
    p.add_argument("--log-scale-limit", type=float, default=3.0)
    a = ap.parse_args(argv)
    if a.cmd in ("collect", "probe"):
        G.set_log_scale_limit(a.log_scale_limit)
    if a.cmd in ("collect", "probe") and a.namespace == G.NAMESPACE and a.cmd == "probe":
        raise SystemExit("the probe never uses the official namespace")
    return {"collect": cmd_collect, "hashes": cmd_hashes, "xcheck": cmd_xcheck, "audit": cmd_audit, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
