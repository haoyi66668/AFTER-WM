from __future__ import annotations

import argparse
import importlib
import json
import sys

from . import registry as P

BUDGETS = {
    "baseline": P.BUDGET_BASELINE,
    "afterwm_eq": P.BUDGET_AFTERWM_EQ,
    "fast_native": P.BUDGET_FAST_NATIVE,
}
REPORT_KEYS = ("total", "trainable", "frozen", "stored_nongrad", "buffers", "groups")


def count_module(model) -> dict:
    trainable = frozen = 0
    groups = {}
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        top = name.split(".", 1)[0]
        g = groups.setdefault(top, {"trainable": 0, "frozen": 0, "buffers": 0})
        if p.requires_grad:
            trainable += p.numel()
            g["trainable"] += p.numel()
        else:
            frozen += p.numel()
            g["frozen"] += p.numel()
    buffers = persistent = 0
    persistent_names = set(model.state_dict().keys())
    for name, b in model.named_buffers():
        buffers += b.numel()
        if name in persistent_names:
            persistent += b.numel()
        top = name.split(".", 1)[0]
        groups.setdefault(top, {"trainable": 0, "frozen": 0, "buffers": 0})["buffers"] += b.numel()
    return {"trainable": int(trainable), "frozen": int(frozen), "buffers": int(buffers),
            "buffers_persistent": int(persistent), "parameter_tensors": len(seen), "groups": groups}


def make_report(model, stored_nongrad: int = 0, groups: dict | None = None,
                stored_nongrad_groups: dict | None = None, extra: dict | None = None) -> dict:
    c = count_module(model)
    rep = {
        "total": int(c["trainable"] + c["frozen"] + int(stored_nongrad)),
        "trainable": c["trainable"],
        "frozen": c["frozen"],
        "stored_nongrad": int(stored_nongrad),
        "buffers": c["buffers"],
        "groups": groups if groups is not None else {k: v["trainable"] + v["frozen"] for k, v in c["groups"].items()},
    }
    if stored_nongrad_groups:
        rep["stored_nongrad_groups"] = stored_nongrad_groups
    if extra:
        rep.update(extra)
    return rep


def budget_check(total: int, budget_class: str) -> dict:
    if budget_class not in BUDGETS:
        raise ValueError(f"unknown budget class {budget_class!r}; expected one of {sorted(BUDGETS)}")
    budget = BUDGETS[budget_class]
    rel = (int(total) - budget) / budget
    return {"budget_class": budget_class, "budget": budget, "total": int(total),
            "rel_dev": rel, "tol": P.BUDGET_TOL, "within_budget": abs(rel) <= P.BUDGET_TOL + 1e-12,
            "allowed_range": [int(round(budget * (1 - P.BUDGET_TOL))), int(round(budget * (1 + P.BUDGET_TOL)))]}


def check_report(report: dict, model, budget_class: str) -> dict:
    indep = count_module(model)
    problems = []
    missing = [k for k in REPORT_KEYS if k not in report]
    if missing:
        problems.append(f"parameter_report missing keys {missing}")
    tr = int(report.get("trainable", -1))
    fr = int(report.get("frozen", -1))
    st = int(report.get("stored_nongrad", 0))
    tot = int(report.get("total", -1))
    if tot != tr + fr + st:
        problems.append(f"total {tot} != trainable {tr} + frozen {fr} + stored_nongrad {st}")
    if tr != indep["trainable"]:
        problems.append(f"reported trainable {tr} != independent requires_grad count {indep['trainable']}")
    if fr + st > indep["frozen"] + indep["buffers"]:
        problems.append(f"reported frozen+stored {fr + st} exceeds non-trainable params+buffers "
                        f"{indep['frozen'] + indep['buffers']}")
    if fr + st < indep["frozen"]:
        problems.append(f"reported frozen+stored {fr + st} < independent requires_grad=False parameter count "
                        f"{indep['frozen']} (frozen parameters must count toward the budget)")
    budget = budget_check(tot, budget_class) if budget_class in BUDGETS else {"within_budget": False,
                                                                             "error": f"unknown class {budget_class}"}
    if not budget.get("within_budget", False):
        problems.append(f"total {tot} outside budget {budget}")
    return {"reported": report, "independent": indep, "budget": budget, "problems": problems,
            "passed": not problems}


def optimizer_coverage(model, optimizer) -> dict:
    in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
    names = dict((id(p), n) for n, p in model.named_parameters())
    trainable_missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in in_opt]
    frozen_in_opt = [n for n, p in model.named_parameters() if (not p.requires_grad) and id(p) in in_opt]
    foreign = [i for i in in_opt if i not in names]
    return {"trainable_not_in_optimizer": trainable_missing, "frozen_in_optimizer": frozen_in_opt,
            "optimizer_params_not_in_model": len(foreign),
            "passed": not trainable_missing and not frozen_in_opt}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Report the parameter count of a model configuration.', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True)
    ap.add_argument("--budget-class", required=True, choices=sorted(BUDGETS))
    ap.add_argument("--variant", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=P.SEEDS[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--with-stats", action="store_true", help="pass real TRAIN stats to build()")
    P.add_cli_options(ap)
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from .run_io import parse_variant
    from .trainer import resolve_module
    mod = resolve_module(args.module)
    cfg = mod.default_config(args.budget_class, **parse_variant(args.variant))
    stats = None
    if args.with_stats:
        from .evaluate import get_stats
        stats = get_stats()
    model = mod.build(cfg, args.seed, args.device, stats)
    rep = mod.forecaster(model, cfg).parameter_report()
    res = check_report(rep, model, args.budget_class)
    print(json.dumps({"module": args.module, "cfg": cfg, **res}, indent=1, default=str))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
