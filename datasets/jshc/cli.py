from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from . import config as C
from . import gen as G


def _counts(s):
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k] = int(v)
    return out


def cmd_collect(a):
    proto = C.make_protocol(a.namespace)
    m = G.collect(a.out, proto, _counts(a.counts), seed_offset=a.seed_offset, workers=a.workers,
                  log=lambda x: print(x, flush=True))
    print(json.dumps({"done": True, "out": a.out, "files": {k: v["sha256"] for k, v in m["files"].items()},
                      "checks": m["checks"], "elapsed_s": m["elapsed_s"]}))
    return 0 if all(v for v in m["checks"].values()) else 3


def _hash_record(raw):
    mgen = G._mgen()
    raw = Path(raw)
    man = json.load(open(raw / "JSHC_MANIFEST.json"))
    rec = {"arrays": {}, "parents": []}
    for split, f in sorted(man["files"].items()):
        z = G.load_npz(raw / f["path"])
        rec["arrays"][split] = {k: mgen.array_sha256(v) for k, v in sorted(z.items())}
    for p in man["parents"]:
        rec["parents"].append({"parent_id": p["parent_id"], "anchor": p["anchor"],
                               "cohorts": [{"plan_seed": c["plan_seed"], "draws": [s["draws"] for s in c["sequences"]],
                                            "phases": [s["phase"] for s in c["sequences"]],
                                            "dev": [s["dev_sq_by_segment"] for s in c["sequences"]]} for c in p["cohorts"]],
                               "replay_exact": p["replay_exact"], "nominal_equals_base_rollout": p["nominal_equals_base_rollout"]})
    rec["code_sha256"] = man["code_sha256"]
    rec["checks"] = man["checks"]
    rec["runtime"] = man["runtime"]
    return rec


def cmd_hashes(a):
    rec = _hash_record(a.raw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(a.out):
        raise SystemExit(f"exists: {a.out}")
    json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({"out": a.out, "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}))
    return 0


def cmd_xcheck(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    diffs = []
    for key in ("arrays", "parents", "code_sha256", "checks"):
        if A.get(key) != B.get(key):
            diffs.append(key)
    detail = {}
    if "arrays" in diffs:
        for split in sorted(set(A["arrays"]) | set(B["arrays"])):
            for k in sorted(set(A["arrays"].get(split, {})) | set(B["arrays"].get(split, {}))):
                if A["arrays"].get(split, {}).get(k) != B["arrays"].get(split, {}).get(k):
                    detail.setdefault(split, []).append(k)
    res = {"passed": not diffs, "differs": diffs, "array_detail": detail, "a": a.a, "b": a.b,
           "a_host": A["runtime"].get("hostname"), "b_host": B["runtime"].get("hostname"),
           "a_cpu_env": A["runtime"].get("env"), "b_cpu_env": B["runtime"].get("env")}
    if os.path.exists(a.out):
        raise SystemExit(f"exists: {a.out}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
    return 0 if res["passed"] else 4


def _load_splits(main, testb):
    main, testb = Path(main), Path(testb)
    mm = json.load(open(main / "JSHC_MANIFEST.json"))
    mb = json.load(open(testb / "JSHC_MANIFEST.json"))
    data = {s: G.load_npz(main / f"jshc_{s}.npz") for s in ("train", "val", "test")}
    data["test_b"] = G.load_npz(testb / "jshc_test.npz")
    return data, mm, mb


def cmd_audit(a):
    mgen = G._mgen()
    data, mm, mb = _load_splits(a.main, a.testb)
    proto = mm["protocol"]
    hard, info = {}, {}
    want = {"train": (288, 24), "val": (144, 12), "test": (144, 12), "test_b": (144, 12)}
    hard["1_counts"] = all(len(d["obs"]) == want[s][0] and len(set(d["parent"].tolist())) == want[s][1]
                           for s, d in data.items())
    hard["2_shapes"] = all(d["obs"].shape[1:] == (301, 17) and d["act"].shape[1:] == (300, 6) and
                           d["fail_joint"].shape[1:] == (300,) and d["changed"].shape[1:] == (300,) and
                           d["obs"].dtype == np.float32 and d["act"].dtype == np.float32 for d in data.values())
    ok3 = True
    for d in data.values():
        keys = list(zip(d["parent"].tolist(), d["cohort"].tolist()))
        for key in set(keys):
            idx = [i for i, k in enumerate(keys) if k == key]
            ok3 &= sorted(d["sequence"][idx].tolist()) == list(C.SEQUENCE_IDS)
            ok3 &= all(np.array_equal(d["act"][idx[0]], d["act"][i]) for i in idx)
            ok3 &= all(np.array_equal(d["obs"][idx[0], 0], d["obs"][i, 0]) for i in idx)
    hard["3_structure_6_sequences_shared_plan_and_anchor"] = bool(ok3)
    hard["4_finite"] = all(np.isfinite(d["obs"]).all() and np.isfinite(d["act"]).all() for d in data.values())
    hard["5_action_range"] = all(float(np.abs(d["act"]).max()) <= proto["action_limit"] + 1e-7 for d in data.values())
    seg_ok = True
    for d in data.values():
        f = d["fail_joint"]
        seg_ok &= bool(np.isin(f, C.CANDIDATE_JOINTS).all())
        seg_ok &= bool(((d["phase"] >= 0) & (d["phase"] < 100)).all())
        for i in range(len(f)):
            b = G.switch_times(proto, int(d["phase"][i])) + [300]
            seg_ok &= all(bool((f[i, lo:hi] == f[i, lo]).all()) for lo, hi in zip(b[:-1], b[1:]))
            seg_ok &= set(np.nonzero(d["changed"][i])[0].tolist()) <= set(b[1:-1])
    hard["6_joint_labels_valid_and_piecewise_constant"] = seg_ok
    lab_ok = True
    for man, splits in ((mm, ("train", "val", "test")), (mb, ("test_b",))):
        off = man["seed_offset"]
        designs = {p["parent_id"]: p for p in man["parents"]}
        for s in splits:
            d = data[s]
            for i in range(len(d["obs"])):
                p = designs[str(d["parent"][i])]
                c = int(d["cohort"][i])
                q = C.SEQUENCE_IDS.index(str(d["sequence"][i]))
                fail, draws, phase, _ = G.joint_schedule(proto, p["index"], c, q, off)
                lab_ok &= phase == int(d["phase"][i])
                lab_ok &= bool(np.array_equal(fail, d["fail_joint"][i])) and bool(
                    np.array_equal(G.changed_flags(fail), d["changed"][i]))
                if q == 0:
                    acts, _ = mgen.action_plan(proto, p["cohort_kinds"][c], p["index"], c, off)
                    lab_ok &= bool(np.array_equal(acts, d["act"][i]))
    hard["7_labels_and_plans_recomputed"] = bool(lab_ok)
    hard["8_replay_and_nominal_checks"] = all(m["checks"]["replay_exact_all"] and m["checks"]["nominal_equals_base_rollout_all"]
                                              for m in (mm, mb))
    def hset(arr):
        return {hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in arr}
    anchors = {"main": {}, "test_b": set()}
    for p in mm["parents"]:
        anchors["main"].setdefault(p["split"], set()).add(p["anchor"]["state_sha256"])
    anchors["test_b"] = {p["anchor"]["state_sha256"] for p in mb["parents"]}
    groups = {"train": anchors["main"].get("train", set()), "val": anchors["main"].get("val", set()),
              "test": anchors["main"].get("test", set()), "test_b": anchors["test_b"]}
    acts = {s: hset(d["act"]) for s, d in data.items()}
    obsh = {s: hset(d["obs"]) for s, d in data.items()}
    inter = 0
    names = list(data)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            inter += len(groups[x] & groups[y]) + len(acts[x] & acts[y]) + len(obsh[x] & obsh[y])
    hard["9_split_disjointness"] = inter == 0
    std = data["train"]["obs"].reshape(-1, 17).astype(np.float64).std(0)
    hard["10_train_std_floor_0.005"] = bool((std >= 0.005).all())
    info["train_obs_std"] = std.tolist()
    for s, d in data.items():
        f = d["fail_joint"]
        info[f"{s}_joint_counts"] = {str(j): int((f == j).sum()) for j in C.CANDIDATE_JOINTS}
        nsch = sum(len(G.switch_times(proto, int(p))) - 1 for p in d["phase"])
        info[f"{s}_scheduled_switches"] = int(nsch)
        info[f"{s}_actual_change_fraction_of_scheduled"] = float(d["changed"].sum() / max(nsch, 1))
        info[f"{s}_flip_fraction_max_abs_pitch_gt_pi_over_2"] = float((np.abs(d["obs"][:, :, 1]).max(1) > np.pi / 2).mean())
        lo, hi = np.percentile(data["train"]["obs"].reshape(-1, 17), [0.1, 99.9], axis=0)
        if s in ("test", "test_b"):
            o = d["obs"].reshape(-1, 17)
            info[f"{s}_outside_train_0.1_99.9_fraction"] = float(((o < lo) | (o > hi)).mean())
    var = std ** 2
    eff = {str(j): [] for j in C.CANDIDATE_JOINTS}
    for man in (mm, mb):
        for p in man["parents"]:
            for c in p["cohorts"]:
                for sq in c["sequences"]:
                    for k, dev in enumerate(sq["dev_sq_by_segment"]):
                        eff[str(sq["draws"][k])].append(float(np.mean(np.asarray(dev) / var)))
    info["effect_vs_nominal_by_joint_median"] = {j: (float(np.median(v)) if v else None) for j, v in eff.items()}
    for s in ("test", "test_b"):
        ch = data[s]["changed"]
        info[f"{s}_origin_change_within_25"] = float(np.mean([ch[:, o:o + 25].any(1).mean() for o in (64, 96, 128, 160, 192)]))
        info[f"{s}_origin_change_within_40"] = float(np.mean([ch[:, o:o + 40].any(1).mean() for o in (64, 96, 128, 160, 192)]))
    res = {"all_hard_pass": all(hard.values()), "hard": hard, "info": info,
           "files": {s: mm["files"].get(s, {}).get("sha256") for s in ("train", "val", "test")} | {"test_b": mb["files"]["test"]["sha256"]}}
    if os.path.exists(a.out):
        raise SystemExit(f"exists: {a.out}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"all_hard_pass": res["all_hard_pass"], "hard": hard}))
    return 0 if res["all_hard_pass"] else 5


def cmd_assemble(a):
    mgen = G._mgen()
    main, testb, out = Path(a.main), Path(a.testb), Path(a.out)
    if out.exists():
        raise SystemExit(f"exists: {out}")
    out.mkdir(parents=True)
    pairs = [(main / "jshc_train.npz", "jshc_train.npz"), (main / "jshc_val.npz", "jshc_val.npz"),
             (main / "jshc_test.npz", "jshc_test.npz"), (testb / "jshc_test.npz", "jshc_testB.npz"),
             (main / "JSHC_MANIFEST.json", "JSHC_MANIFEST.json"), (testb / "JSHC_MANIFEST.json", "JSHC_MANIFEST_testB.json")]
    for src, dst in pairs:
        shutil.copy2(src, out / dst)
    card = {"dataset": "JSHC-v1 (HalfCheetah disabled-joint switching; GrBAL/MOLe/MBCD protocol; data generated by us)",
            "files": {}, "namespace": C.NAMESPACE, "obs_dim": 17, "act_dim": 6, "dt": 0.05, "T": 300}
    for _, dst in pairs:
        card["files"][dst] = mgen.sha256_file(out / dst)
    json.dump(card, open(out / "DATASET_CARD.json", "w"), indent=1, sort_keys=True)
    with open(out / "SHA256SUMS.txt", "w") as f:
        for name in sorted(os.listdir(out)):
            if name != "SHA256SUMS.txt":
                f.write(f"{mgen.sha256_file(out / name)}  {name}\n")
    for p in out.iterdir():
        os.chmod(p, 0o444)
    os.chmod(out, 0o555)
    print(open(out / "SHA256SUMS.txt").read())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="JSHC generator: collect, hashes, xcheck, audit, assemble.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--counts", required=True)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--namespace", default=C.NAMESPACE)
    p = sub.add_parser("hashes")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("xcheck")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--main", required=True)
    p.add_argument("--testb", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("assemble")
    p.add_argument("--main", required=True)
    p.add_argument("--testb", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"collect": cmd_collect, "hashes": cmd_hashes, "xcheck": cmd_xcheck, "audit": cmd_audit,
            "assemble": cmd_assemble}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
