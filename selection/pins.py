from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

COLLAPSE = "expert collapse: no epoch passed the alive gate"
BASE_KEYS = ("c_e", "epoch", "epochs_run", "stop", "val_score", "stats_sha256", "train_sha256", "val_sha256",
             "stage1a_state_sha256")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def replay_epoch(rows: list, rule: dict) -> dict:
    from afterwm import stage1
    return stage1.replay_selection(rows, int(rule["candidate_min_epoch"]), int(rule["min_alive_streak"]),
                                   int(rule["max_epochs"]), rule.get("patience"))


def collect(bases_dir: str, dataset: str, seeds, abandoned) -> dict:
    rec = {"dataset": dataset, "rule": "update_equivalent", "sha256": {}, "bases": {},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if set(abandoned) - set(seeds):
        raise SystemExit(f"--abandoned {sorted(set(abandoned) - set(seeds))} is not one of the seeds {list(seeds)}")
    if abandoned:
        rec["abandoned"] = {}
    for s in seeds:
        d = os.path.join(bases_dir, str(s))
        if s in abandoned:
            failed = os.path.join(d, "FAILED.json")
            fj = json.load(open(failed))
            if os.path.exists(os.path.join(d, "BASE.json")) or COLLAPSE not in str(fj.get("exception")):
                raise SystemExit(f"seed {s}: not an abandoned base (BASE.json exists or FAILED.json is not the "
                                 "alive-gate failure)")
            rec["abandoned"][str(s)] = {"reason": fj.get("exception"),
                                        "rule": "no epoch passed the alive gate -> the seed is left out of every AFTER-WM "
                                                "row, no replacement"}
            continue
        base = json.load(open(os.path.join(d, "BASE.json")))
        sha = _sha(os.path.join(d, base["file"]))
        rows = [json.loads(line) for line in open(os.path.join(d, "HISTORY.jsonl")) if line.strip()]
        rule = base.get("selection_rule") or {}
        rep = replay_epoch(rows, rule)
        ok = (base.get("dataset") == dataset and not base.get("debug") and rule.get("rule") == "update_equivalent"
              and sha == base["sha256"] and int(base["seed"]) == int(s) and rep["selected_epoch"] == base["epoch"])
        if not ok:
            raise SystemExit(f"seed {s}: base check failed ({d}): replayed epoch {rep['selected_epoch']}, "
                             f"BASE.json epoch {base.get('epoch')}")
        rec["sha256"][str(s)] = sha
        rec["bases"][str(s)] = {k: base.get(k) for k in BASE_KEYS}
        rec["bases"][str(s)]["replayed_epoch"] = rep["selected_epoch"]
    return rec


def main(argv=None) -> int:
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="Collect the stage-1 bases of a dataset, replay the epoch rule on "
                                             "HISTORY.jsonl and write the sha256 pins (PINS.json).")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--abandoned", type=int, action="append", default=[],
                    help="seed whose stage-1 run failed the alive gate (ALFA: 20280916)")
    ap.add_argument("--out", default=None, help="default <runs-root>/bases/<dataset>/PINS.json")
    P.add_cli_options(ap)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(P.BASES_DIR, "PINS.json")
    if a.write and os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    rec = collect(P.BASES_DIR, P.DATASET, tuple(P.SEEDS), a.abandoned)
    print(json.dumps({"sha256": rec["sha256"], "epochs": {k: v["epoch"] for k, v in rec["bases"].items()},
                      "abandoned": sorted(rec.get("abandoned", {}))}), flush=True)
    if a.write:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
        os.replace(tmp, out)
        print(json.dumps({"written": out, "sha256": _sha(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
