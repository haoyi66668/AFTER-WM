from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import core as C
from . import legacy as LEGACY

H_BASE = 5
BATCH = 128
LR = 3e-4
WD = 1e-4
CLIP = 10.0
CANDIDATE_MIN_EPOCH = 50
MIN_ALIVE_STREAK = 20
NATIVE_EPOCHS = 250
NATIVE_BATCHES_PER_EPOCH = 47
UPDEQ_MAX_EPOCHS = 30
UPDEQ_PATIENCE = 10
RULES = ("update_equivalent", "native")
SCHEMA = "afterwm_stage1_base_v1"
MODEL_CONFIG = dict(observation_dim=C.OBS, action_dim=C.ACT, num_experts=3, router_hidden_dims=(128, 128),
                    expert_hidden_dims=(256, 256, 256), alpha_q=0.5, route_probability_floor=1e-3,
                    epsilon=1e-8)


def _train_config():
    from afterwm.lib.experts.train.train_1a import Stage1ATrainConfig
    return Stage1ATrainConfig(sigma=0.5, rollout_weight=0.5, mode_weight=0.0, balance_weight=0.01,
                              responsibility_alive_weight=0.01, minimum_responsibility_usage=0.1,
                              gradient_clip_norm=CLIP, supervise_router_mode=False, mode_to_expert=(),
                              target_usage=(1 / 3, 1 / 3, 1 / 3))


def _val_config():
    from afterwm.lib.experts.eval.validate_1a import Stage1AValidationConfig
    return Stage1AValidationConfig(sigma=0.5, rollout_weight=0.5, minimum_effective_experts=2.5,
                                   minimum_effective_responsibility_experts=2.5, minimum_prior_usage=0.05,
                                   minimum_responsibility_usage=0.05, minimum_pairwise_output_distance=1e-4,
                                   minimum_merge_contribution=1e-5)


def _jsonable(x):
    if dataclasses.is_dataclass(x):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


DISCLOSURES = (
    "Data order: torch.randperm per epoch from a seeded torch.Generator.",
    "Validation batches carry one source id ('val'); labels are never read.",
    "The run fails if no epoch passes the alive gate (no starvation, output, duplicate or merge collapse); route "
    "imbalance alone with all experts alive is reported, not failed.",
    "Per-step train metrics are accumulated on the device and synchronised once per epoch.",
    "Selection rule update_equivalent: the alive-gate thresholds of the original stage-1 recipe (minimum epoch 50 and a "
    "20-epoch alive streak at 47 updates per epoch, i.e. 2,350 and 940 updates) are converted to epochs of this "
    "dataset; lowest VAL among eligible epochs; at most 30 epochs; early stop after 10 epochs without improvement.",
)


def disclosures() -> list:
    return list(DISCLOSURES)


def update_equivalent_rule(steps_per_epoch: int, native_batches: int = NATIVE_BATCHES_PER_EPOCH,
                           native_min_epoch: int = CANDIDATE_MIN_EPOCH,
                           native_streak: int = MIN_ALIVE_STREAK) -> dict:
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be >= 1")
    min_updates = native_min_epoch * native_batches
    streak_updates = native_streak * native_batches
    return {"candidate_min_epoch": int(math.ceil(min_updates / steps_per_epoch)),
            "min_alive_streak": int(math.ceil(streak_updates / steps_per_epoch)),
            "derivation": {"native_batches_per_epoch": native_batches,
                           "native_batches_source": "original stage-1 recipe: 6,000 windows / batch 128 = 47 batches per epoch",
                           "native_candidate_min_epoch": native_min_epoch,
                           "native_min_alive_streak": native_streak,
                           "native_min_updates": min_updates, "native_streak_updates": streak_updates,
                           "dataset_batches_per_epoch": int(steps_per_epoch),
                           "formula": "ceil(native_epochs * 47 / dataset_batches_per_epoch)"}}


def resolve_rule(rule: str, steps_per_epoch: int, epochs=None, patience=None, min_epoch=None, min_streak=None) -> dict:
    if rule == "update_equivalent":
        eq = update_equivalent_rule(steps_per_epoch)
        out = {"rule": rule, "candidate_min_epoch": eq["candidate_min_epoch"], "min_alive_streak": eq["min_alive_streak"],
               "max_epochs": UPDEQ_MAX_EPOCHS, "patience": UPDEQ_PATIENCE, "derivation": eq["derivation"]}
    elif rule == "native":
        out = {"rule": rule, "candidate_min_epoch": CANDIDATE_MIN_EPOCH, "min_alive_streak": MIN_ALIVE_STREAK,
               "max_epochs": NATIVE_EPOCHS, "patience": None, "derivation": None}
    else:
        raise ValueError(f"rule must be one of {RULES}")
    for k, v in (("candidate_min_epoch", min_epoch), ("min_alive_streak", min_streak), ("max_epochs", epochs),
                 ("patience", patience)):
        if v is not None:
            out[k] = int(v) if v != 0 or k != "patience" else None
    if out["max_epochs"] < 1:
        raise ValueError("max epochs must be >= 1")
    return out


def stop_due(best_epoch, last_epoch: int, patience) -> bool:
    return bool(patience) and best_epoch is not None and last_epoch - best_epoch >= patience


def select_update(state: dict, epoch: int, score: float, alive: bool, min_epoch: int, min_streak: int) -> dict:
    state["alive_streak"] = state["alive_streak"] + 1 if alive else 0
    eligible = epoch >= min_epoch and state["alive_streak"] >= min_streak
    improved = eligible and score < state["best"]["score"]
    if improved:
        state["best"] = {"epoch": epoch, "score": score}
    return {"eligible": eligible, "improved": improved, "alive_streak": state["alive_streak"]}


def replay_selection(rows, min_epoch: int, min_streak: int, max_epochs: int, patience=None) -> dict:

    st = {"alive_streak": 0, "best": {"epoch": None, "score": math.inf}}
    last = 0
    for r in rows:
        e = int(r["epoch"])
        if e != last + 1:
            raise ValueError(f"HISTORY epochs not consecutive at {e}")
        if e > max_epochs or stop_due(st["best"]["epoch"], last, patience):
            return {"selected_epoch": st["best"]["epoch"], "selected_score": st["best"]["score"],
                    "stopped_after_epoch": last, "stop_reason": "max_epochs" if e > max_epochs else "patience"}
        select_update(st, e, float(r["val_score"]), bool(r["expert_alive"]), min_epoch, min_streak)
        last = e
    reason = None
    if last >= max_epochs:
        reason = "max_epochs"
    elif stop_due(st["best"]["epoch"], last, patience):
        reason = "patience"
    return {"selected_epoch": st["best"]["epoch"], "selected_score": st["best"]["score"],
            "stopped_after_epoch": last if reason else None, "stop_reason": reason}


def window_index(n_episodes: int, horizon: int = H_BASE):
    t = np.arange(0, 300 - horizon + 1)
    e = np.repeat(np.arange(n_episodes), len(t))
    tt = np.tile(t, n_episodes)
    return e.astype(np.int64), tt.astype(np.int64)


class Windows:

    def __init__(self, z, a, device, horizon=H_BASE):
        import torch
        self.z = torch.as_tensor(z, device=device)
        self.a = torch.as_tensor(a, device=device)
        e, t = window_index(self.z.shape[0], horizon)
        self.e = torch.as_tensor(e, device=device)
        self.t = torch.as_tensor(t, device=device)
        self.h = horizon
        self.n = len(e)
        self.ar_o = torch.arange(horizon + 1, device=device)
        self.ar_a = torch.arange(horizon, device=device)

    def batch(self, idx, *, validation=False):
        import torch
        e = self.e[idx]
        t = self.t[idx]
        obs = self.z[e[:, None], t[:, None] + self.ar_o[None, :]]
        act = self.a[e[:, None], t[:, None] + self.ar_a[None, :]]
        b = {"observations": obs, "actions": act,
             "mode_id": torch.full((len(idx),), -1, dtype=torch.long, device=obs.device),
             "has_mode_label": torch.zeros(len(idx), dtype=torch.bool, device=obs.device)}
        if validation:
            b["source_id"] = ["val"] * len(idx)
        return b

    def val_batches(self, batch=BATCH):
        import torch
        for s in range(0, self.n, batch):
            yield self.batch(torch.arange(s, min(s + batch, self.n), device=self.e.device), validation=True)


def make_step(model, opt, tcfg):
    import torch
    from afterwm.lib.experts.train.train_1a import compute_stage1a_loss
    params = [p for p in model.parameters() if p.requires_grad]

    def step(batch):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = compute_stage1a_loss(model, batch, tcfg)
        if losses.labelled_transition_count != 0:
            raise RuntimeError("labels must never be used")
        losses.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, CLIP, error_if_nonfinite=True)
        opt.step()
        return torch.stack([losses.total.detach(), losses.one_step_nll.detach(), losses.rollout_nll.detach(),
                            losses.balance_loss.detach(), losses.responsibility_alive_loss.detach(),
                            norm.detach().to(losses.total.dtype)])
    return step


STEP_KEYS = ("total_loss", "one_step_nll", "rollout_nll", "balance_loss", "responsibility_alive_loss", "gradient_norm")


def calibrate_c_e(model, z, a, device) -> dict:
    import torch
    norms = []
    with torch.no_grad():
        for i in range(z.shape[0]):
            zi = torch.as_tensor(z[i], device=device)
            ai = torch.as_tensor(a[i], device=device)
            pred = model.predict_base_experts(zi[:-1], ai)
            err = pred.z_hat - zi[1:, None, :]
            norms.append(err.double().norm(dim=-1).amin(-1).cpu().numpy())
    n = np.concatenate(norms)
    return {"c_e": max(float(np.quantile(n, 0.99)), 1e-8), "transitions": int(n.size),
            "quantile": 0.99, "median_min_norm": float(np.median(n)),
            "rule": "99th pct of min_k ||G0_k(z_t,a_t)-z_{t+1}||_2 over all TRAIN transitions, TRAIN frame (label-free)"}


def normalizer_block(stats, train_sha, n_obs, n_act):
    return {"dataset_manifest_sha256": train_sha, "observation_dim": C.OBS, "action_dim": C.ACT,
            "observation_count": int(n_obs), "action_count": int(n_act), "epsilon": 1e-6,
            "observation_mean": [float(x) for x in stats.obs_mean],
            "observation_std": [float(x) for x in stats.obs_std],
            "action_mean": [float(x) for x in stats.act_mean],
            "action_std": [float(x) for x in stats.act_std]}


def disclosure_text(rule: dict, epoch, score, stop: dict) -> str:
    from pipeline import registry as P
    ds = P.DATASET.upper()
    return (f"{ds} base (Router+G0, {C.BASE_PARAMS:,} params, frozen afterwards) trained per seed on {ds} TRAIN with "
            f"the stage-1 recipe (AdamW 3e-4, wd 1e-4, clip 10, batch 128, H5 windows), endpoint selected on "
            f"{ds} VAL by the 'alive' gate with its epoch thresholds converted to optimizer updates (min epoch "
            f"{rule['candidate_min_epoch']}, streak {rule['min_alive_streak']}, at most {rule['max_epochs']} epochs, "
            f"early stop after {rule['patience']} epochs without improvement), i.e. the update_equivalent rule. "
            f"Selected epoch {epoch} (VAL {score:.5f}); stopped after epoch "
            f"{stop.get('stopped_after_epoch')} ({stop.get('stop_reason')}). TEST/TEST_B were never used.")


def main(argv=None):
    ap = argparse.ArgumentParser(description='Stage 1: train the frozen multi-expert base of one seed on the TRAIN split.', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None, help="run dir (default <runs-root>/bases/<dataset>/<seed>)")
    ap.add_argument("--rule", choices=RULES, default="update_equivalent")
    ap.add_argument("--epochs", type=int, default=None, help="max epochs (default: rule; native 250, update_equivalent 30)")
    ap.add_argument("--patience", type=int, default=None,
                    help="early stop after N epochs without improvement (default: rule; 0 disables)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--smoke-steps", type=int, default=0, help="timing smoke: N train steps + 1 VAL pass, no formal output")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--candidate-min-epoch", type=int, default=None,
                    help="override the rule's alive-gate minimum epoch (recorded as a deviation)")
    ap.add_argument("--min-alive-streak", type=int, default=None,
                    help="override the rule's consecutive alive VAL epochs (recorded as a deviation)")
    ap.add_argument("--debug-steps-per-epoch", type=int, default=0, help="functional test only (non-formal output dir)")
    ap.add_argument("--debug-min-epoch", type=int, default=None)
    ap.add_argument("--debug-min-streak", type=int, default=None)
    from pipeline import registry as _R
    _R.add_cli_options(ap)
    args = ap.parse_args(argv)
    debug = bool(args.debug_steps_per_epoch or args.debug_min_epoch is not None or args.debug_min_streak is not None)

    C.configure_runtime(args.threads, deterministic=True)
    import random
    import torch
    from pipeline import data as D
    from pipeline import registry as P
    from afterwm.lib.experts.models import Stage1ABaseWorldModel, Stage1AModelConfig
    from afterwm.lib.experts.eval.validate_1a import validate_stage1a
    from afterwm.lib.experts.checkpoints.stage1a import load_frozen_stage1a_base

    seed = int(args.seed)
    formal_seeds = tuple(P.SEEDS)
    if seed not in formal_seeds and not args.smoke_steps:
        raise ValueError(f"formal base seeds are {formal_seeds}")
    data_dir = args.data_dir or P.DATA_DIR
    out = Path(args.out or os.path.join(P.BASES_DIR, str(seed)))
    if debug and str(out.resolve()).startswith(str(Path(P.BASES_DIR).resolve())):
        raise ValueError("debug settings may not write into the formal bases directory")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise FileExistsError(f"{out}/COMPLETE.json exists; refusing to retrain")
    lock = open(out / ".lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"another process holds {out}/.lock")

    random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)

    obs_tr, act_tr = D.load_visible("train", data_dir)
    obs_va, act_va = D.load_visible("val", data_dir)
    stats = D.TrainStats.fit(obs_tr, act_tr)
    z_tr, a_tr = stats.norm_obs(obs_tr), stats.norm_act(act_tr)
    z_va, a_va = stats.norm_obs(obs_va), stats.norm_act(act_va)
    train_sha = D._sha256(D.split_path("train", data_dir))
    val_sha = D._sha256(D.split_path("val", data_dir))
    W = Windows(z_tr, a_tr, device)
    V = Windows(z_va, a_va, device)
    steps_per_epoch = math.ceil(W.n / BATCH)
    if args.debug_steps_per_epoch:
        steps_per_epoch = int(args.debug_steps_per_epoch)
    rule = resolve_rule(args.rule, math.ceil(W.n / BATCH), epochs=args.epochs, patience=args.patience,
                        min_epoch=args.candidate_min_epoch, min_streak=args.min_alive_streak)
    min_epoch = rule["candidate_min_epoch"] if args.debug_min_epoch is None else args.debug_min_epoch
    min_streak = rule["min_alive_streak"] if args.debug_min_streak is None else args.debug_min_streak
    max_epochs, patience = rule["max_epochs"], rule["patience"]
    rule_overridden = any(v is not None for v in (args.candidate_min_epoch, args.min_alive_streak)) or debug

    torch.manual_seed(seed)
    cfg = Stage1AModelConfig(**MODEL_CONFIG)
    model = Stage1ABaseWorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == C.BASE_PARAMS, n_params
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    tcfg, vcfg = _train_config(), _val_config()
    gen = torch.Generator().manual_seed(seed)
    step_fn = make_step(model, opt, tcfg)

    selection = {"gate": "alive", "candidate_min_epoch": min_epoch, "minimum_consecutive_alive_epochs": min_streak}
    if args.rule != "native" or patience is not None:
        selection.update(rule=args.rule, patience=patience, derivation=rule["derivation"],
                         overridden=rule_overridden)
    identity = {"schema": SCHEMA, "seed": seed, "model_config": _jsonable(MODEL_CONFIG),
                "train_config": _jsonable(tcfg), "validation_config": _jsonable(vcfg),
                "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WD, "betas": [0.9, 0.999], "eps": 1e-8},
                "batch_size": BATCH, "horizon": H_BASE, "stride": 1, "train_windows": W.n,
                "val_windows": V.n, "steps_per_epoch": steps_per_epoch, "epochs": max_epochs,
                "selection": selection,
                "debug": debug,
                "train_sha256": train_sha, "val_sha256": val_sha, "stats_sha256": stats.sha256(),
                "code": C.code_manifest(), "torch": torch.__version__, "python": platform.python_version(),
                "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "threads": args.threads, "smoke_steps": args.smoke_steps}
    ident_hash = C.sha256_json({k: v for k, v in identity.items() if k not in ("code", "smoke_steps")})

    if args.smoke_steps:
        model.train()
        times = []
        perm = torch.randperm(W.n, generator=gen).to(device)
        for i in range(args.smoke_steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            idx = perm[(i % steps_per_epoch) * BATCH:(i % steps_per_epoch + 1) * BATCH]
            r = step_fn(W.batch(idx))
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        vres = validate_stage1a(model, V.val_batches(), vcfg)
        val_s = time.perf_counter() - t0
        ms = 1000 * float(np.median(times[10:])) if len(times) > 20 else 1000 * float(np.mean(times))
        est_epoch_s = ms / 1000 * steps_per_epoch + val_s
        res = {"smoke_steps": args.smoke_steps, "ms_per_step_median": ms, "ms_per_step_mean": 1000 * float(np.mean(times)),
               "val_pass_seconds": val_s, "steps_per_epoch": steps_per_epoch, "est_epoch_seconds": est_epoch_s,
               "est_hours_250_epochs": est_epoch_s * 250 / 3600, "last_loss": float(r[0]),
               "val_score": vres.score, "val_alive": vres.expert_alive,
               "peak_mem_MB": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None}
        C.atomic_json(out / "SMOKE.json", {**res, "identity": identity})
        print(json.dumps(res), flush=True)
        return res

    last_path, best_path = out / "last_state.pt", out / "base_stage1a.pt"
    hist_path = out / "HISTORY.jsonl"
    start_epoch, alive_streak = 1, 0
    best = {"epoch": None, "score": math.inf}
    started = time.time()
    if last_path.exists():
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        if st["identity_hash"] != ident_hash:
            raise RuntimeError("last_state.pt identity differs; refusing to resume")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        gen.set_state(st["generator"]); torch.set_rng_state(st["torch_rng"])
        start_epoch, alive_streak, best = st["epoch"] + 1, st["alive_streak"], st["best"]
        started -= st.get("elapsed", 0.0)
        rows = [json.loads(l) for l in hist_path.read_text().splitlines()] if hist_path.exists() else []
        rows = [r for r in rows if r["epoch"] <= st["epoch"]]
        hist_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"resumed after epoch {st['epoch']}", flush=True)
    C.atomic_json(out / "IDENTITY.json", {**identity, "identity_hash": ident_hash})

    norm_block = normalizer_block(stats, train_sha, obs_tr.shape[0] * obs_tr.shape[1],
                                  act_tr.shape[0] * act_tr.shape[1])
    run_config = {"source": LEGACY.STAGE1_SOURCE, "seed": seed, "epochs": max_epochs, "batch_size": BATCH,
                  "horizon": H_BASE, "stride": 1, "learning_rate": LR, "weight_decay": WD,
                  "gradient_clip_norm": CLIP, "deterministic": True, "device": str(device),
                  "selection_rule": args.rule, "patience": patience}
    sel_state = {"alive_streak": alive_streak, "best": best}
    last_epoch = start_epoch - 1
    stop_reason = None
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            if stop_due(sel_state["best"]["epoch"], epoch - 1, patience):
                stop_reason = "patience"
                break
            t_ep = time.time()
            model.train()
            perm = torch.randperm(W.n, generator=gen).to(device)
            acc = None
            gmax = None
            for s in range(steps_per_epoch):
                idx = perm[s * BATCH:(s + 1) * BATCH]
                r = step_fn(W.batch(idx))
                acc = r if acc is None else acc + r
                gmax = r[-1] if gmax is None else torch.maximum(gmax, r[-1])
            vals = (acc / steps_per_epoch).tolist()
            train_m = dict(zip(STEP_KEYS, vals))
            train_m.update(max_gradient_norm=float(gmax), labelled_transition_count=0, batches=steps_per_epoch)
            t_tr = time.time() - t_ep
            vres = validate_stage1a(model, V.val_batches(), vcfg)
            score = float(vres.score)
            if not math.isfinite(score):
                raise FloatingPointError("VAL score nonfinite")
            alive = bool(vres.expert_alive)
            sel = select_update(sel_state, epoch, score, alive, min_epoch, min_streak)
            alive_streak, eligible, improved, best = sel["alive_streak"], sel["eligible"], sel["improved"], sel_state["best"]
            vjs = _jsonable(vres)
            if improved:
                ckpt = {"format_name": LEGACY.STAGE1_CHECKPOINT_FORMAT, "format_version": 2,
                        "epoch": epoch, "validation_score": score, "best_validation_score": score,
                        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "optimizer_state_dict": opt.state_dict(),
                        "model_config": _jsonable(MODEL_CONFIG), "train_config": _jsonable(tcfg),
                        "validation_config": _jsonable(vcfg), "run_config": run_config,
                        "selection_policy": {"format_version": 1, "selection_gate": "alive",
                                             "candidate_min_epoch": min_epoch,
                                             "minimum_consecutive_alive_epochs": min_streak,
                                             "rule": args.rule, "patience": patience},
                        "label_policy": {"labels_used_during_warmup": False, "expert_mode_warmup_epochs": 0,
                                         "expert_mode_warmup_initial_weight": 0.0, "mode_to_expert": [],
                                         "router_mode_supervision": False,
                                         "saved_after_mode_weight_reached_zero": True},
                        "dataset": {"name": f"{P.DATASET.upper()} TRAIN", "train_sha256": train_sha, "val_sha256": val_sha,
                                    "episodes": int(obs_tr.shape[0]), "windows": W.n,
                                    "visible_keys_only": ["obs", "act"]},
                        "normalizer": norm_block,
                        "metrics": {"epoch": epoch, "train": train_m, "validation": vjs,
                                    "optimization_policy": {"expert_mode_weight": 0.0,
                                                            "mode_labels_used_for_expert_loss": False,
                                                            "labelled_transition_count": 0,
                                                            "router_mode_supervision": False,
                                                            "checkpoint_eligible": True,
                                                            "alive_candidate_eligible": True,
                                                            "alive_streak": alive_streak}},
                        "checkpoint_role": "selected", "checkpoint_gate": "alive", "selected_gate": "alive",
                        "candidate_stability": {"alive_streak": alive_streak,
                                                "minimum_consecutive_alive_epochs": min_streak}}
                C.atomic_torch_save(best_path, ckpt)
            row = {"epoch": epoch, "train": train_m, "val_score": score, "val_one_step": vres.one_step_nll,
                   "val_rollout": vres.rollout_nll, "prior_usage": list(vres.prior_usage),
                   "responsibility_usage": list(vres.responsibility_usage),
                   "effective_prior_experts": vres.effective_prior_experts,
                   "effective_responsibility_experts": vres.effective_responsibility_experts,
                   "mean_max_prior_probability": vres.mean_max_prior_probability,
                   "min_pairwise_distance": vres.minimum_pairwise_expert_distance,
                   "merge_contribution": list(vres.merge_contribution_by_expert),
                   "expert_alive": alive, "strictly_balanced": bool(vres.strictly_balanced),
                   "legacy_expert_collapse": bool(vres.expert_collapse), "alive_streak": alive_streak,
                   "eligible": eligible, "selected": improved, "best_epoch": best["epoch"],
                   "best_score": best["score"] if best["epoch"] else None,
                   "train_seconds": t_tr, "epoch_seconds": time.time() - t_ep,
                   "elapsed_seconds": time.time() - started}
            with open(hist_path, "a") as f:
                f.write(json.dumps(_jsonable(row)) + "\n")
            C.atomic_json(out / "PROGRESS.json", _jsonable(row))
            C.atomic_torch_save(last_path, {"identity_hash": ident_hash, "epoch": epoch, "model": model.state_dict(),
                                            "optimizer": opt.state_dict(), "generator": gen.get_state(),
                                            "torch_rng": torch.get_rng_state(), "alive_streak": alive_streak,
                                            "best": best, "elapsed": time.time() - started})
            last_epoch = epoch
            print(json.dumps({"epoch": epoch, "loss": round(train_m["total_loss"], 5), "val": round(score, 5),
                              "alive": alive, "streak": alive_streak, "best_epoch": best["epoch"],
                              "eff": round(vres.effective_prior_experts, 3),
                              "s": round(time.time() - t_ep, 1)}), flush=True)
        else:
            stop_reason = "max_epochs"
        best = sel_state["best"]
        if best["epoch"] is None:
            raise RuntimeError("expert collapse: no epoch passed the alive gate "
                               f"(epoch>={min_epoch}, {min_streak} consecutive alive VAL epochs)")
        file_sha = C.sha256_file(best_path)
        frozen = load_frozen_stage1a_base(best_path, expected_sha256=file_sha, device="cpu",
                                          expected_config=cfg)
        hist_rows = [json.loads(l) for l in hist_path.read_text().splitlines()]
        vmin = min(hist_rows, key=lambda r: r["val_score"])
        sel = torch.load(best_path, map_location="cpu", weights_only=True)
        sel_val = sel["metrics"]["validation"]
        if not sel_val["expert_alive"]:
            raise RuntimeError("selected endpoint is not expert_alive (expert collapse)")
        if int(sel["epoch"]) != int(best["epoch"]):
            raise RuntimeError(f"checkpoint epoch {sel['epoch']} != selected epoch {best['epoch']}")
        model.load_state_dict(frozen.model.state_dict())
        model.eval()
        cal = calibrate_c_e(model, z_tr, a_tr, device)
        stop = {"stopped_after_epoch": last_epoch, "stop_reason": stop_reason, "max_epochs": max_epochs,
                "patience": patience, "updates_run": last_epoch * steps_per_epoch,
                "selected_updates": best["epoch"] * steps_per_epoch}
        rule_rec = {"rule": args.rule, "gate": "alive", "candidate_min_epoch": min_epoch,
                    "min_alive_streak": min_streak, "max_epochs": max_epochs, "patience": patience,
                    "derivation": rule["derivation"], "overridden": rule_overridden,
                    "native": args.rule == "native" and not rule_overridden}
        base = {"schema": SCHEMA, "variant": "own", "dataset": P.DATASET, "train_sha256": train_sha,
                "seed": seed, "file": best_path.name, "sha256": file_sha,
                "stage1a_state_sha256": frozen.state_sha256, "epoch": best["epoch"], "val_score": best["score"],
                "epochs_run": last_epoch, "epochs_max": max_epochs, "epochs_native": NATIVE_EPOCHS, "debug": debug,
                "selection_rule": rule_rec, "stop": stop,
                "val_min_any_epoch": {"epoch": vmin["epoch"], "val_score": vmin["val_score"],
                                      "note": "lowest VAL over all run epochs (informational; selectable only if eligible)"},
                "epochs_deviation": (None if max_epochs == NATIVE_EPOCHS and not patience else
                                     f"at most {max_epochs} epochs" + (f", early stop after {patience} epochs without "
                                                                       "VAL improvement" if patience else "")
                                     + f" instead of the native {NATIVE_EPOCHS} epochs"),
                "c_e": cal["c_e"], "c_e_calibration": cal, "gamma": 1.0, "gamma_rule": "no router hook",
                "normalizer": norm_block, "stats_sha256": stats.sha256(), "train_sha256": train_sha,
                "val_sha256": val_sha, "model_config": _jsonable(MODEL_CONFIG), "params": n_params,
                "selected_validation": {k: sel_val[k] for k in (
                    "score", "one_step_nll", "rollout_nll", "prior_usage", "responsibility_usage",
                    "effective_prior_experts", "effective_responsibility_experts", "mean_max_prior_probability",
                    "minimum_pairwise_expert_distance", "merge_contribution_by_expert", "expert_alive",
                    "strictly_balanced", "route_imbalance", "expert_collapse", "output_collapse",
                    "duplicate_expert_collapse", "merge_contribution_collapse")},
                "loader_check": "afterwm.lib.experts.checkpoints.stage1a.load_frozen_stage1a_base(expected_sha256, expected_config) OK",
                "identity_hash": ident_hash, "wall_seconds": time.time() - started,
                "disclosure": disclosure_text(rule_rec, best["epoch"], best["score"], stop),
                "disclosures": disclosures()}
        C.atomic_json(out / "BASE.json", base)
        C.atomic_json(out / "COMPLETE.json", {"status": "complete", "sha256": file_sha, "epoch": best["epoch"],
                                              "val_score": best["score"], "rule": args.rule, "c_e": cal["c_e"],
                                              "stopped_after_epoch": last_epoch, "stop_reason": stop_reason,
                                              "wall_seconds": time.time() - started})
        print(json.dumps({"complete": True, "epoch": best["epoch"], "val": best["score"], "sha256": file_sha,
                          "c_e": cal["c_e"], "stopped_after_epoch": last_epoch, "stop_reason": stop_reason}), flush=True)
        return base
    except Exception as exc:
        import traceback
        C.atomic_json(out / "FAILED.json", {"exception": repr(exc), "traceback": traceback.format_exc(),
                                            "elapsed_seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
