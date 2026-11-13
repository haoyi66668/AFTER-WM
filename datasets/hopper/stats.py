from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import gen

SPLIT_FILES = {"train": "hopper_train.npz", "val": "hopper_val.npz", "test": "hopper_test.npz",
               "test_b": "hopper_testB.npz"}
EXPECTED = {"train": (24, 288), "val": (12, 144), "test": (12, 144), "test_b": (12, 144)}
REDRAW_MAX = 0.25
EFFECT_MIN_REL = 0.25
EFFECT_MAX_REL = 4.0
MIN_TRAIN_STD = 0.002
D_REF_TRAIN = 0.13336802669316572
OBS_GROUPS = {"z": [0], "pitch": [1], "joint_pos": [2, 3, 4], "lin_vel": [5, 6], "pitch_rate": [7],
              "joint_vel": [8, 9, 10]}


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


def check_split(name, d, manifest, protocol):
    out = {"split": name}
    n_par, n_ep = EXPECTED[name]
    parents = sorted(set(map(str, d["parent"])))
    out["counts"] = {"episodes": int(d["obs"].shape[0]), "parents": len(parents),
                     "ok": int(d["obs"].shape[0]) == 12 * len(parents),
                     "formal_size_ok": int(d["obs"].shape[0]) == n_ep and len(parents) == n_par}
    out["shapes_ok"] = bool(d["obs"].shape[1:] == (protocol["T"] + 1, protocol["obs_dim"])
                            and d["act"].shape[1:] == (protocol["T"], protocol["act_dim"])
                            and d["friction"].shape[1:] == (protocol["T"],) and d["impulse"].shape[1:] == (protocol["T"], 3))
    keys = {}
    for i, (p, c, s) in enumerate(zip(d["parent"], d["cohort"], d["sequence"])):
        keys.setdefault((str(p), int(c)), {})[str(s)] = i
    out["pairing_ok"] = bool(all(sorted(v) == sorted(gen.SEQUENCES) for v in keys.values())
                             and len(keys) == 2 * len(parents))
    t0 = protocol["segments"][1]
    out["pre_switch_identical_ok"] = bool(all(np.array_equal(d["obs"][v[s], :t0 + 1], d["obs"][v["null"], :t0 + 1])
                                              and np.array_equal(d["act"][v[s]], d["act"][v["null"]])
                                              for v in keys.values() for s in v))
    out["finite_ok"] = bool(np.isfinite(d["obs"]).all() and np.isfinite(d["act"]).all())
    env = manifest["env"]
    lo = np.asarray(env["actuator_ctrlrange"], np.float64)[:, 0]
    hi = np.asarray(env["actuator_ctrlrange"], np.float64)[:, 1]
    q = lo + 0.5 * (d["act"].astype(np.float64) + 1.0) * (hi - lo)
    box = np.asarray(protocol["offset_box_rad"], np.float64)
    home = gen.home_pose(protocol)
    out["action_range_ok"] = bool(np.abs(d["act"]).max() <= 1.0 and (np.abs(q - home) <= box + 1e-5).all())
    out["action_offset_max_rad"] = np.abs(q - home).reshape(-1, 3).max(0).tolist()
    bad = [i for i in range(d["obs"].shape[0]) if gen.health_check(d["obs"][i], protocol) is not None]
    out["health_ok"] = not bad
    out["health_violations"] = bad[:20]
    out["friction_cap_ok"] = bool(d["friction_eff"].max() <= protocol["friction_eff_max"] + 1e-6)
    designs = {p["parent_id"]: p for p in manifest["parents"]}
    so = int(manifest["seed_offset"])
    lab_ok = True
    for (p, c), v in keys.items():
        des = designs[p]
        att = int(d["plan_attempt"][v["null"]])
        labels, imps = gen.disturbance_labels(protocol, des, c, att, so)
        acts, _ = gen.action_plan(protocol, des["cohort_kinds"][c], des["index"], c, att, lo.astype(np.float32).astype(np.float64),
                                  hi.astype(np.float32).astype(np.float64), so)
        for s, i in v.items():
            lab = labels[s]
            for k in ("friction", "gain", "payload", "friction_eff", "gain_eff"):
                if not np.array_equal(d[k][i], lab[k].astype(np.float32)):
                    lab_ok = False
            if not np.array_equal(d["seg"][i], lab["seg"]) or not np.array_equal(d["impulse"][i], imps):
                lab_ok = False
            if not np.array_equal(d["act"][i], acts):
                lab_ok = False
    out["labels_recomputed_ok"] = bool(lab_ok)
    cohorts = [(p["parent_id"], c["cohort"], c["attempt"], len(p.get("anchor_rejections", [])))
               for p in manifest["parents"] if p["split"] == name for c in p["cohorts"]]
    redrawn = sum(1 for _, _, a, r in cohorts if a > 0 or r > 0)
    frac = redrawn / max(1, len(cohorts))
    anchor_rej = [r for p in manifest["parents"] if p["split"] == name for r in p.get("anchor_rejections", [])]
    out["redraw"] = {"cohorts": len(cohorts), "redrawn": redrawn, "fraction": frac,
                     "max_attempt": max(a for _, _, a, _ in cohorts) if cohorts else None,
                     "anchor_redraws": len(anchor_rej),
                     "parents_with_anchor_redraw": sorted({r["parent_id"] for r in anchor_rej}),
                     "ok": frac <= REDRAW_MAX}
    pset = {p for p, _, _, _ in cohorts}
    rej = [r for r in manifest["rejections"] if r["parent_id"] in pset]
    out["rejections"] = {"n": len(rej), "by_reason": _count(r["reason"] for r in rej),
                         "by_kind": _count(r["kind"] for r in rej), "by_sequence": _count(r["sequence"] for r in rej)}
    obs = d["obs"].astype(np.float64)
    travel = np.abs(obs[:, 1:, 5].sum(1) * protocol["dt"])
    out["physics"] = {"z_min": float(obs[:, :, 0].min()), "z_p1": float(np.percentile(obs[:, :, 0], 1)),
                      "z_median": float(np.median(obs[:, :, 0])), "z_max": float(obs[:, :, 0].max()),
                      "pitch_min": float(obs[:, :, 1].min()), "pitch_max": float(obs[:, :, 1].max()),
                      "pitch_abs_p99": float(np.percentile(np.abs(obs[:, :, 1]), 99)),
                      "x_speed_abs_median": float(np.median(np.abs(obs[:, 1:, 5]))),
                      "x_speed_abs_p99": float(np.percentile(np.abs(obs[:, 1:, 5]), 99)),
                      "x_travel_abs_median_m": float(np.median(travel)),
                      "joint_speed_abs_max": float(np.abs(obs[:, :, 8:11]).max()),
                      "qvel_abs_gt10_fraction": float((np.abs(obs[:, :, 5:]) > 10).mean())}
    out["regimes"] = _count(p["contact_regime"] for p in manifest["parents"] if p["split"] == name)
    return out, keys


def audit_dir(main_data, testb_data=None):
    main_data = Path(main_data)
    man = json.loads((main_data / "HOPPER_MANIFEST.json").read_text())
    protocol = man["protocol"]
    splits = {s: load(main_data / f) for s, f in (("train", "hopper_train.npz"), ("val", "hopper_val.npz"),
                                                    ("test", "hopper_test.npz")) if (main_data / f).exists()}
    mans = {s: man for s in splits}
    if testb_data is not None:
        tb = Path(testb_data)
        man_b = json.loads((tb / "HOPPER_MANIFEST.json").read_text())
        if json.dumps(man_b["protocol"], sort_keys=True) != json.dumps(protocol, sort_keys=True):
            raise ValueError("TEST_B was generated with a different protocol")
        splits["test_b"] = load(tb / "hopper_test.npz")
        mans["test_b"] = man_b
    res = {"protocol_schema": protocol["schema"], "namespace": protocol["namespace"], "splits": {}}
    for s, d in splits.items():
        r, _ = check_split("test" if s == "test_b" else s, d, mans[s], protocol)
        r["split"] = s
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
        res["effect"] = {"D_hopper_train": D, "D_ref_train": d_ref, "ratio": D / d_ref,
                         "ok": D >= EFFECT_MIN_REL * d_ref, "too_large": D > EFFECT_MAX_REL * d_ref,
                         "by_split": {s: d_statistic(x, sd)[0] for s, x in splits.items()}}
        kinds = {}
        for p in man["parents"]:
            for c in p["cohorts"]:
                kinds[(p["parent_id"], c["cohort"])] = c["kind"]
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
        res["train_std_min"] = float(sd.min())
        res["train_std_ok"] = bool(sd.min() >= MIN_TRAIN_STD)
    hard = ["counts", "shapes_ok", "pairing_ok", "pre_switch_identical_ok", "finite_ok", "action_range_ok", "health_ok",
            "friction_cap_ok", "labels_recomputed_ok"]
    split_ok = all(all((v[h]["ok"] if isinstance(v[h], dict) else v[h]) for h in hard) for v in res["splits"].values())
    res["hard_checks_ok"] = bool(split_ok and res["replay_audit_ok"] and res["disjoint_ok"]
                                 and res.get("train_std_ok", True))
    res["formal_sizes_ok"] = all(v["counts"]["formal_size_ok"] for v in res["splits"].values()) and len(res["splits"]) == 4
    res["redraw_ok"] = all(v["redraw"]["ok"] for v in res["splits"].values())
    res["effect_ok"] = bool(res.get("effect", {}).get("ok", False))
    res["passed"] = bool(res["hard_checks_ok"] and res["formal_sizes_ok"] and res["redraw_ok"] and res["effect_ok"])
    return res
