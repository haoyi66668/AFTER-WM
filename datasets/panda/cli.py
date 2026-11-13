from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _counts(text):
    if not text:
        return None
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def _logger(path):
    path = Path(path)
    t0 = time.monotonic()

    def log(stage, **fields):
        from simcore.gen import _jsonable
        row = {"stage": stage, "t": round(time.monotonic() - t0, 1), **_jsonable(fields)}
        text = json.dumps(row, ensure_ascii=False)
        print(text, flush=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    return log


def source_hashes():
    root = Path(__file__).resolve().parent.parent
    files = sorted([p for p in (root / "panda").glob("*.py")] + [root / "panda_model" / "mujoco_menagerie_6da8b971" / "SHA256SUMS.txt"] +
                   [root / "simcore" / "gen.py", root / "simcore" / "__init__.py"] +
                   sorted((root / "simcore" / "vendor").glob("*.py")))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def cmd_collect(a):
    from . import gen
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg = gen.default_config(a.design)
    gen.write_json(out / "CONFIG.json", {"design": a.design, "config": cfg, "counts": _counts(a.counts),
                                         "seed_offset": a.seed_offset, "argv": sys.argv,
                                         "source_sha256": source_hashes()})
    man = gen.collect(out / "data", cfg, counts=_counts(a.counts), log=_logger(out / "progress.jsonl"),
                      seed_offset=a.seed_offset)
    gen.write_json(out / "COMPLETE.json", {"files": man["files"], "rejections": len(man["rejections"]),
                                           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({k: v["episodes"] for k, v in man["files"].items()}))


def compare_npz(pa, pb):
    from simcore.gen import array_sha256
    za, zb = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
    rows = {}
    for k in sorted(set(za.files) | set(zb.files)):
        if k not in za.files or k not in zb.files:
            rows[k] = {"present": [k in za.files, k in zb.files], "equal": False}
            continue
        x, y = za[k], zb[k]
        r = {"sha_a": array_sha256(x), "sha_b": array_sha256(y), "shape": list(x.shape), "dtype": str(x.dtype)}
        r["equal"] = bool(r["sha_a"] == r["sha_b"])
        if x.shape == y.shape and np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            dd = np.abs(x.astype(np.float64) - y.astype(np.float64))
            r["max_abs_diff"] = float(dd.max()) if dd.size else 0.0
            r["n_diff"] = int((dd > 0).sum())
        rows[k] = r
    return rows


def cmd_compare(a):
    from .gen import write_json
    rows = compare_npz(a.a, a.b)
    allk = all(r.get("equal", False) for r in rows.values())
    res = {"a": a.a, "b": a.b, "all_equal": allk, "arrays": rows}
    if a.json:
        write_json(a.json, res)
    print(json.dumps({"all_equal": allk, **{k: (v.get("equal"), v.get("max_abs_diff")) for k, v in rows.items()}}))
    return 0 if allk else 3


def cmd_audit(a):
    from .gen import write_json
    from .stats import audit_dir
    res = audit_dir(a.main, a.testb, replay_all=not a.no_replay)
    write_json(a.out, res)
    print(json.dumps({k: res[k] for k in ("hard_checks_ok", "formal_sizes_ok", "redraw_ok", "effect_ok", "passed")}))
    return 0 if res["passed"] else 4


def cmd_hashes(a):
    from simcore.gen import array_sha256
    from .gen import sha256_file, write_json
    d = Path(a.data)
    man = json.loads((d / "PANDA_MANIFEST.json").read_text())
    out = {"data": str(d), "seed_offset": man["seed_offset"], "design": man.get("design"), "splits": {},
           "anchors": {p["parent_id"]: p["anchor"] for p in man["parents"]},
           "accepted_attempts": {p["parent_id"]: [c["attempt"] for c in p["cohorts"]] for p in man["parents"]},
           "rejections": man["rejections"], "replay_audit": man["replay_audit"],
           "anchor_rejections": man.get("anchor_rejections", []),
           "env_model_sha256": man["env"]["model_sha256"], "env_spec_xml_sha256": man["env"]["spec_xml_sha256"]}
    ok = True
    for split, info in sorted(man["files"].items()):
        with np.load(d / info["path"], allow_pickle=False) as z:
            arr = {k: array_sha256(z[k]) for k in z.files}
        match = arr == info["array_sha256"] and sha256_file(d / info["path"]) == info["sha256"]
        ok = ok and match
        out["splits"][split] = {"path": info["path"], "file_sha256": sha256_file(d / info["path"]),
                                "array_sha256": arr, "matches_manifest": bool(match)}
    out["all_match_manifest"] = bool(ok)
    write_json(a.out, out)
    print(json.dumps({"all_match_manifest": ok, "splits": sorted(out["splits"])}))
    return 0 if ok else 6


def cmd_xcheck(a):
    from .gen import write_json
    A = json.loads(Path(a.a).read_text())
    B = json.loads(Path(a.b).read_text())
    res = {"a": a.a, "b": a.b, "arrays": {}, "file_sha256_equal": {}}
    ok = sorted(A["splits"]) == sorted(B["splits"]) and A["all_match_manifest"] and B["all_match_manifest"]
    for split in sorted(set(A["splits"]) | set(B["splits"])):
        xa = A["splits"].get(split, {}).get("array_sha256", {})
        xb = B["splits"].get(split, {}).get("array_sha256", {})
        res["arrays"][split] = {k: xa.get(k) == xb.get(k) for k in sorted(set(xa) | set(xb))}
        ok = ok and all(res["arrays"][split].values()) and bool(xa)
        res["file_sha256_equal"][split] = A["splits"].get(split, {}).get("file_sha256") == B["splits"].get(split, {}).get("file_sha256")
    for k in ("seed_offset", "design", "anchors", "accepted_attempts", "rejections", "replay_audit", "anchor_rejections",
              "env_model_sha256", "env_spec_xml_sha256"):
        same = json.dumps(A.get(k), sort_keys=True) == json.dumps(B.get(k), sort_keys=True)
        res[f"{k}_equal"] = same
        ok = ok and same
    res["identical"] = bool(ok)
    write_json(a.out, res)
    print(json.dumps({"identical": ok, "file_sha256_equal": res["file_sha256_equal"]}))
    return 0 if ok else 7


def cmd_assemble(a):
    import shutil
    from .gen import sha256_file, write_json
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    cfg_main = json.loads((Path(a.main) / "CONFIG.json").read_text())
    cfg_tb = json.loads((Path(a.testb) / "CONFIG.json").read_text())
    if json.dumps(cfg_main["config"], sort_keys=True) != json.dumps(cfg_tb["config"], sort_keys=True):
        raise ValueError("main and TEST_B were generated with different configs")
    if cfg_main["source_sha256"] != cfg_tb["source_sha256"]:
        raise ValueError("main and TEST_B were generated with different generator sources")
    out.mkdir(parents=True)
    main_d, tb_d = Path(a.main) / "data", Path(a.testb) / "data"
    pairs = [(main_d / "panda_train.npz", "panda_train.npz"), (main_d / "panda_val.npz", "panda_val.npz"),
             (main_d / "panda_test.npz", "panda_test.npz"), (tb_d / "panda_test.npz", "panda_testB.npz"),
             (main_d / "PANDA_MANIFEST.json", "PANDA_MANIFEST.json"), (tb_d / "PANDA_MANIFEST.json", "PANDA_MANIFEST_testB.json")]
    lines = []
    for src, name in pairs:
        shutil.copy2(src, out / name)
        lines.append(f"{sha256_file(out / name)}  {name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    write_json(out / "DATASET_CARD.json", {"design": cfg_main["design"], "config": cfg_main["config"],
                                           "main": {"counts": cfg_main["counts"], "seed_offset": cfg_main["seed_offset"]},
                                           "test_b": {"counts": cfg_tb["counts"], "seed_offset": cfg_tb["seed_offset"]},
                                           "source_sha256": cfg_main["source_sha256"],
                                           "sha256": dict(reversed(l.split("  ")) for l in lines)})
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description="panda generator: collect, compare, audit, hashes, xcheck, assemble.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--design", default="panda", choices=["panda", "unittest"])
    c.add_argument("--out", required=True)
    c.add_argument("--counts", default=None)
    c.add_argument("--seed-offset", type=int, default=0)
    m = sub.add_parser("compare")
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    m.add_argument("--json", default=None)
    u = sub.add_parser("audit")
    u.add_argument("--main", required=True)
    u.add_argument("--testb", default=None)
    u.add_argument("--no-replay", action="store_true", help="skip the full replay (tests only; the formal audit replays)")
    u.add_argument("--out", required=True)
    h = sub.add_parser("hashes")
    h.add_argument("--data", required=True)
    h.add_argument("--out", required=True)
    x = sub.add_parser("xcheck")
    x.add_argument("--a", required=True)
    x.add_argument("--b", required=True)
    x.add_argument("--out", required=True)
    s_ = sub.add_parser("assemble")
    s_.add_argument("--main", required=True)
    s_.add_argument("--testb", required=True)
    s_.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rc = {"collect": cmd_collect, "compare": cmd_compare, "audit": cmd_audit,
          "hashes": cmd_hashes, "xcheck": cmd_xcheck, "assemble": cmd_assemble}[a.cmd](a)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
