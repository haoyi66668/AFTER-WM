from __future__ import annotations

import argparse
import json
import math
import os

GRID_SEEDS = (20260910, 20270913)
STEP_NATIVE, STEP_ALT = 1e-3, 1e-2
STEP_EXT = {STEP_ALT: 1e-1, STEP_NATIVE: 1e-4}
RULE_STAGE1 = ("inner step in {0.001 (native), 0.01} at the fixed learning rate; mean best VAL of seeds "
               "20260910+20270913; eligible iff neither seed diverged; lower wins, ties -> native; one extension: "
               "0.01 -> 0.1, 0.001 -> 0.0001")
RULE_FINAL = ("final step over {0.001, 0.01} + the stage-1 extension; mean best VAL of the two seeds; eligible iff "
              "neither seed diverged; lower wins; ties -> native 0.001, then the value closest to 0.001 in |log10|")


def arm_dir(runs_dir: str, step: float) -> str:
    return os.path.join(runs_dir, "grbal_step", f"inner_lr_{step:g}")


def read_run(run_dir: str) -> dict:
    def rj(name):
        p = os.path.join(run_dir, name)
        return json.load(open(p)) if os.path.exists(p) else None
    comp, div = rj("COMPLETE.json"), rj("DIVERGED.json")
    vals = []
    p = os.path.join(run_dir, "VAL.jsonl")
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                v = r.get("val")
                if isinstance(v, (int, float)) and math.isfinite(v):
                    vals.append((float(v), int(r["step"])))
    if comp is not None and comp.get("status") == "complete":
        best = min(vals) if vals else (None, None)
        return {"status": "complete", "best_val": best[0], "best_step": best[1]}
    if div is not None:
        return {"status": "diverged", "best_val": None, "reason": div.get("reason")}
    return {"status": "unfinished", "best_val": None}


def arm_score(runs: dict) -> dict:
    elig = all(r["status"] == "complete" and r.get("best_val") is not None for r in runs.values())
    score = sum(float(r["best_val"]) for r in runs.values()) / len(runs) if elig else None
    return {"eligible": elig, "score": score, "seeds": runs}


def stage1_choice(arms: dict) -> dict:
    elig = [s for s in (STEP_NATIVE, STEP_ALT) if arms[s]["eligible"]]
    choice = min(elig, key=lambda s: (arms[s]["score"], 0 if s == STEP_NATIVE else 1)) if elig else None
    ext = STEP_EXT.get(choice)
    return {"stage1_choice": choice, "extension_needed": ext is not None, "extension_inner_lr": ext}


def final_choice(arms: dict) -> dict:
    elig = [s for s, a in arms.items() if a["eligible"]]
    if not elig:
        return {"status": "no_eligible_arm", "chosen_inner_lr": None}
    best = min(elig, key=lambda s: (arms[s]["score"], 0 if s == STEP_NATIVE else 1, abs(math.log10(s / STEP_NATIVE))))
    return {"status": "chosen", "chosen_inner_lr": best, "extension_chosen": best in STEP_EXT.values()}


def main(argv=None) -> int:
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="Select GrBAL's inner adaptation step on VAL from the runs trained with each candidate step "
                                             "(<runs-root>/runs/<dataset>/grbal_step/inner_lr_<step>/<seed>).")
    ap.add_argument("stage", choices=["stage1", "final"])
    ap.add_argument("--out", default=None, help="default <runs-root>/runs/<dataset>/grbal_step/<STAGE1|CHOICE>.json")
    P.add_cli_options(ap)
    a = ap.parse_args(argv)
    base = os.path.join(P.RUNS_DIR, "grbal_step")
    steps = [STEP_NATIVE, STEP_ALT]
    if a.stage == "final":
        st = json.load(open(os.path.join(base, "STAGE1.json")))
        if st.get("extension_needed"):
            steps.append(float(st["extension_inner_lr"]))
    arms = {}
    for s in steps:
        runs = {str(seed): read_run(os.path.join(arm_dir(P.RUNS_DIR, s), str(seed))) for seed in GRID_SEEDS}
        un = [k for k, r in runs.items() if r["status"] == "unfinished"]
        if un:
            raise SystemExit(f"step {s:g}: runs of seeds {un} are not finished")
        arms[s] = arm_score(runs)
    rec = {"dataset": P.DATASET, "seeds": list(GRID_SEEDS), "arms": {f"{s:g}": v for s, v in arms.items()}}
    if a.stage == "stage1":
        rec.update(rule=RULE_STAGE1, **stage1_choice(arms))
    else:
        rec.update(rule=RULE_FINAL, **final_choice(arms))
    out = a.out or os.path.join(base, "STAGE1.json" if a.stage == "stage1" else "CHOICE.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "arms"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
