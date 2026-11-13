from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import gen

SPLIT_FILES = {"train": "panda_train.npz", "val": "panda_val.npz", "test": "panda_test.npz", "test_b": "panda_testB.npz"}
EXPECTED = {"train": (24, 288), "val": (12, 144), "test": (12, 144), "test_b": (12, 144)}
REDRAW_MAX = 0.25
EFFECT_MIN_REL = 0.25
EFFECT_MAX_REL = 4.0
MIN_TRAIN_STD = 0.01
D_REF_TRAIN = 0.13336802669316572
OBS_GROUPS = {"joint_pos": list(range(0, 7)), "joint_vel": list(range(7, 14))}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def d_statistic(d, sd, seq="persistent", lo=41, hi=121):
    obs = d["obs"].astype(np.float64)
    keys = {(str(p), int(c), str(s)): i for i, (p, c, s) in enumerate(zip(d["parent"], d["cohort"], d["sequence"]))}
    vals = []
    for (p, c, s), i in keys.items():
        if s != seq:
            continue
        j = keys[(p, c, "null")]
        vals.append(float((((obs[i, lo:hi] - obs[j, lo:hi]) / sd) ** 2).mean()))
    return float(np.median(vals)) if vals else float("nan"), vals


def _sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _count(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def check_split(name, d, manifest, protocol, half=None, lo_hi=None):
    out = {"split": name}
    n_par, n_ep = EXPECTED[name]
    parents = sorted(set(map(str, d["parent"])))
    out["counts"] = {"episodes": int(d["obs"].shape[0]), "parents": len(parents),
                     "ok": int(d["obs"].shape[0]) == 12 * len(parents),
                     "formal_size_ok": int(d["obs"].shape[0]) == n_ep and len(parents) == n_par}
    out["shapes_ok"] = bool(d["obs"].shape[1:] == (protocol["T"] + 1, protocol["obs_dim"])
                            and d["act"].shape[1:] == (protocol["T"], protocol["act_dim"])
                            and d["wrench"].shape[1:] == (protocol["T"], 3) and d["impulse"].shape[1:] == (protocol["T"], 3))
    keys = {}
    for i, (p, c, s) in enumerate(zip(d["parent"], d["cohort"], d["sequence"])):
        keys.setdefault((str(p), int(c)), {})[str(s)] = i
    out["pairing_ok"] = bool(all(sorted(v) == sorted(gen.SEQUENCES) for v in keys.values()) and len(keys) == 2 * len(parents))
    t0 = protocol["segments"][1]
    out["pre_switch_identical_ok"] = bool(all(
        np.array_equal(d["obs"][v[s], :t0 + 1], d["obs"][v["null"], :t0 + 1])
        and np.array_equal(d["act"][v[s]], d["act"][v["null"]]) for v in keys.values() for s in v))
    out["finite_ok"] = bool(np.isfinite(d["obs"]).all() and np.isfinite(d["act"]).all())
    out["action_range_ok"] = bool(np.abs(d["act"]).max() <= protocol["action_limit"] + 1e-7)
    if lo_hi is not None:
        q = d["obs"][:, :, :7].astype(np.float64)
        out["joint_range_ok"] = bool((q >= lo_hi[0][None, None, :]).all() and (q <= lo_hi[1][None, None, :]).all())
        out["joint_margin_min_rad"] = float(min((q - lo_hi[0][None, None, :]).min(), (lo_hi[1][None, None, :] - q).min()))
    designs = {p["parent_id"]: p for p in manifest["parents"]}
    so = int(manifest["seed_offset"])
    lab_ok = True
    for (p, c), v in keys.items():
        des = designs[p]
        att = int(d["plan_attempt"][v["null"]])
        labels, wrench, imps = gen.disturbance_labels(protocol, des, c, att, so)
        acts, _ = gen.action_plan(protocol, des["cohort_kinds"][c], des["index"], c, att, half, so)
        for s, i in v.items():
            lab = labels[s]
            for k in ("friction", "gain", "payload", "friction_eff", "gain_eff"):
                if not np.array_equal(d[k][i], lab[k].astype(np.float32)):
                    lab_ok = False
            if not (np.array_equal(d["seg"][i], lab["seg"]) and np.array_equal(d["impulse"][i], imps)
                    and np.array_equal(d["wrench"][i], wrench.astype(np.float32)) and np.array_equal(d["act"][i], acts)):
                lab_ok = False
            if int(d["plan_attempt"][i]) != att:
                lab_ok = False
    out["labels_recomputed_ok"] = bool(lab_ok)
    cohorts = [(p["parent_id"], c["cohort"], c["attempt"], len(p.get("anchor_rejections", [])))
               for p in manifest["parents"] if p["split"] == name for c in p["cohorts"]]
    redrawn = sum(1 for _, _, a, r in cohorts if a > 0 or r > 0)
    frac = redrawn / max(1, len(cohorts))
    anchor_rej = [r for p in manifest["parents"] if p["split"] == name for r in p.get("anchor_rejections", [])]
    out["redraw"] = {"cohorts": len(cohorts), "redrawn": redrawn, "fraction": frac,
                     "max_attempt": max(a for _, _, a, _ in cohorts), "anchor_redraws": len(anchor_rej),
                     "parents_with_anchor_redraw": sorted({r["parent_id"] for r in anchor_rej}), "ok": frac <= REDRAW_MAX}
    pids = {p for p, _, _, _ in cohorts}
    rej = [r for r in manifest["rejections"] if r["parent_id"] in pids]
    out["rejections"] = {"n": len(rej), "by_reason": _count(r["reason"] for r in rej),
                         "by_kind": _count(r["kind"] for r in rej), "by_sequence": _count(r["sequence"] for r in rej)}
    return out, keys


def full_replay(d, manifest, protocol, adapter):
    designs = {p["parent_id"]: p for p in manifest["parents"]}
    so = int(manifest["seed_offset"])
    n_ok, bad = 0, []
    snaps = {}
    for i in range(d["obs"].shape[0]):
        p, c, s = str(d["parent"][i]), int(d["cohort"][i]), str(d["sequence"][i])
        des = designs[p]
        if p not in snaps:
            snap, anc = gen.discover_anchor(adapter, protocol, des, so, start_attempt=int(des["anchor"]["attempt"]))
            if anc["state_sha256"] != des["anchor"]["state_sha256"] or anc["attempt"] != des["anchor"]["attempt"]:
                bad.append({"episode": i, "why": "anchor differs"})
                continue
            snaps[p] = snap
        att = int(d["plan_attempt"][i])
        labels, wrench, imps = gen.disturbance_labels(protocol, des, c, att, so)
        lab = labels[s]
        o, viol = gen.rollout(adapter, snaps[p], d["act"][i], lab["friction_eff"], lab["gain_eff"], lab["payload"],
                              wrench, imps)
        if viol is not None or not np.array_equal(o, d["obs"][i]):
            bad.append({"episode": i, "parent": p, "cohort": c, "sequence": s, "violation": viol,
                        "obs_equal": bool(np.array_equal(o, d["obs"][i]))})
        else:
            n_ok += 1
    return {"episodes": int(d["obs"].shape[0]), "ok": n_ok, "bad": bad[:20], "all_ok": not bad}


def audit_dir(main_data, testb_data=None, replay_all=True):
    main_data = Path(main_data)
    man = json.loads((main_data / "PANDA_MANIFEST.json").read_text())
    protocol = man["protocol"]
    adapter = gen.make_adapter(protocol)
    half = adapter.half
    lo_hi = (adapter.lo, adapter.hi)
    splits = {s: load(main_data / f) for s, f in (("train", "panda_train.npz"), ("val", "panda_val.npz"),
                                                    ("test", "panda_test.npz")) if (main_data / f).exists()}
    mans = {s: man for s in splits}
    if testb_data is not None:
        tb = Path(testb_data)
        man_b = json.loads((tb / "PANDA_MANIFEST.json").read_text())
        if json.dumps(man_b["protocol"], sort_keys=True) != json.dumps(protocol, sort_keys=True):
            raise ValueError("TEST_B was generated with a different protocol")
        splits["test_b"] = load(tb / "panda_test.npz")
        mans["test_b"] = man_b
    res = {"protocol_schema": protocol["schema"], "namespace": protocol["namespace"], "splits": {},
           "env_model_sha256": adapter.model_sha256}
    for s, d in splits.items():
        r, _ = check_split("test" if s == "test_b" else s, d, mans[s], protocol, half, lo_hi)
        r["split"] = s
        if replay_all:
            r["full_replay"] = full_replay(d, mans[s], protocol, adapter)
            r["health_ok"] = r["full_replay"]["all_ok"]
        res["splits"][s] = r
    res["replay_audit_ok"] = all(x["exact"] for m in {id(v): v for v in mans.values()}.values() for x in m["replay_audit"])
    anchors, plans, seqs = {}, {}, {}
    for s, d in splits.items():
        anchors[s] = {_sha(o[0]) for o in d["obs"]}
        plans[s] = {_sha(a) for a in d["act"]}
        seqs[s] = {_sha(o) for o in d["obs"]}
    inter = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter[f"{a}|{b}"] = {"anchor": len(anchors[a] & anchors[b]), "plan": len(plans[a] & plans[b]),
                                 "obs": len(seqs[a] & seqs[b])}
    res["cross_split_intersections"] = inter
    res["disjoint_ok"] = all(v["anchor"] == 0 and v["plan"] == 0 and v["obs"] == 0 for v in inter.values())
    if "train" in splits:
        tr = splits["train"]
        sd = tr["obs"].reshape(-1, tr["obs"].shape[-1]).astype(np.float64).std(0)
        D, _ = d_statistic(tr, sd)
        d_ref = D_REF_TRAIN
        res["effect"] = {"D_panda_train": D, "D_ref_train": d_ref, "ratio": D / d_ref,
                         "ok": D >= EFFECT_MIN_REL * d_ref, "too_large": D > EFFECT_MAX_REL * d_ref,
                         "by_split": {s: d_statistic(x, sd)[0] for s, x in splits.items()}}
        kinds = {(p["parent_id"], c["cohort"]): c["kind"] for p in man["parents"] for c in p["cohorts"]}
        by_kind = {}
        obs = tr["obs"].astype(np.float64)
        keys = {(str(p), int(c), str(s)): i for i, (p, c, s) in enumerate(zip(tr["parent"], tr["cohort"], tr["sequence"]))}
        for (p, c, s), i in keys.items():
            if s != "persistent":
                continue
            j = keys[(p, c, "null")]
            by_kind.setdefault(kinds[(p, c)], []).append(float((((obs[i, 41:121] - obs[j, 41:121]) / sd) ** 2).mean()))
        res["effect"]["by_kind_median"] = {k: float(np.median(v)) for k, v in by_kind.items()}
        r = tr["obs"].reshape(-1, tr["obs"].shape[-1]).astype(np.float64)
        lo, hi = np.percentile(r, 0.1, axis=0), np.percentile(r, 99.9, axis=0)
        res["out_of_train_range"] = {s: float((((x["obs"].reshape(-1, r.shape[1]) < lo) |
                                               (x["obs"].reshape(-1, r.shape[1]) > hi)).mean()))
                                     for s, x in splits.items()}
        res["obs_train_std"] = sd.tolist()
        res["train_std_ok"] = bool(sd.min() >= MIN_TRAIN_STD)
        res["joint_velocity_absmax"] = {s: np.abs(x["obs"][:, :, 7:]).max(axis=(0, 1)).astype(float).tolist()
                                        for s, x in splits.items()}
    hard = ["counts", "shapes_ok", "pairing_ok", "pre_switch_identical_ok", "finite_ok", "action_range_ok",
            "joint_range_ok", "labels_recomputed_ok"] + (["health_ok"] if replay_all else [])
    split_ok = all(all((v[h]["ok"] if isinstance(v[h], dict) else v[h]) for h in hard) for v in res["splits"].values())
    res["hard_checks_ok"] = bool(split_ok and res["replay_audit_ok"] and res["disjoint_ok"]
                                 and res.get("train_std_ok", True))
    res["full_replay_done"] = bool(replay_all)
    res["formal_sizes_ok"] = all(v["counts"]["formal_size_ok"] for v in res["splits"].values()) and len(res["splits"]) == 4
    res["redraw_ok"] = all(v["redraw"]["ok"] for v in res["splits"].values())
    res["effect_ok"] = bool(res.get("effect", {}).get("ok", False))
    res["passed"] = bool(res["hard_checks_ok"] and res["formal_sizes_ok"] and res["redraw_ok"] and res["effect_ok"]
                         and replay_all)
    return res
