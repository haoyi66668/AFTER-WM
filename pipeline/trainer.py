from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import collections
import copy
import importlib
import json
import math
import random
import signal
import socket
import sys
import time
import traceback

import numpy as np
import torch

from . import data as D
from . import evaluate as E
from . import paramcount as PC
from . import registry as P
from . import run_io as R

if __name__ == "__main__":
    sys.modules.setdefault("pipeline.trainer", sys.modules["__main__"])

SCHEMA = "afterwm_train_v2"
EXIT_ALREADY_COMPLETE, EXIT_LOCK_BUSY, EXIT_PROTOCOL_MISMATCH, EXIT_DIVERGED, EXIT_TERMINATED = 3, 4, 5, 6, 143
TRAINER_LEVEL_KEYS = ("lr_mult",)


class Diverged(FloatingPointError):

    def __init__(self, msg, step=None, loss=None):
        super().__init__(msg)
        self.step, self.loss = step, loss
PROTOCOL_CONSTANTS = ("CONTEXT", "MAX_H", "HORIZONS", "EVAL_STARTS", "TRAIN_T_MIN", "TRAIN_T_MAX", "SEEDS",
                      "BATCH", "MAX_STEPS", "VAL_EVERY", "CKPT_EVERY", "VAL_SELECTION_METRIC", "BUDGET_BASELINE",
                      "BUDGET_AFTERWM_EQ", "BUDGET_TOL", "SPLIT_SHA256", "VISIBLE_KEYS")


PROTOCOLS = ("p1", "p2")
P2_U = 16000
P2_WARMUP_MIN = 320
P2_VAL_EVERY = 1000
P2_CLIP_FALLBACK = 1.0


def p2_lr_factor(s: int, U: int, W: int) -> float:
    s, U, W = int(s), int(U), int(W)
    if not 1 <= W < U:
        raise ValueError(f"P2 schedule needs 1 <= W < U, got W={W}, U={U}")
    if not 1 <= s <= U:
        raise ValueError(f"P2 update index s={s} is outside 1..{U}")
    if s <= W:
        return s / W
    return 0.5 * (1.0 + math.cos(math.pi * (s - W) / (U - W)))


def resolve_p2(module, cfg: dict, U: int, warmup_min: int) -> dict:
    name = getattr(module, "__name__", str(module))
    if cfg.get("lr_sweep", True) is False:
        raise ValueError(f"{name}: lr_sweep is False (GPMM): P2 does not apply")
    if hasattr(module, "make_scheduler"):
        raise ValueError(f"{name} defines its own lr scheduler; P2 owns the lr schedule")
    if hasattr(module, "custom_train") and not getattr(module, "P2_AWARE", False):
        raise ValueError(f"{name}: custom_train does not implement the P2 hooks (module attribute P2_AWARE)")
    overrides = {}
    nw = 0
    if cfg.get("warmup"):
        nw = int(cfg["warmup"])
        overrides["warmup"] = {"native": cfg["warmup"], "p2": 0,
                               "why": "the optimizer's own warmup is replaced by the P2 schedule (W includes it)"}
        cfg["warmup"] = 0
    W = max(nw, int(warmup_min))
    native_clip = cfg.get("grad_clip")
    if native_clip:
        clip, rule = float(native_clip), "native global-norm clip threshold"
    elif cfg.get("agc"):
        clip, rule = None, "native AGC inside the optimizer; no extra global clip"
    else:
        clip, rule = float(P2_CLIP_FALLBACK), "no native clip -> global-norm clip 1.0"
        overrides["grad_clip"] = {"native": native_clip, "p2": clip}
        cfg["grad_clip"] = clip
    if "max_updates" in cfg:
        overrides["max_updates"] = {"native": cfg["max_updates"], "p2": int(U)}
        cfg["max_updates"] = int(U)
    return {"name": "p2", "U": int(U), "W": int(W), "native_warmup": nw,
            "warmup_min": int(warmup_min), "schedule_steps": int(U),
            "lr_schedule": "eta(s) = eta* * s/W for s <= W; eta* * 0.5*(1 + cos(pi*(s-W)/(U-W))) for s > W; "
                           "per param group (native group ratios kept)",
            "grad_clip": clip, "clip_rule": rule, "overrides": overrides,
            "official": bool(int(U) == P2_U and int(warmup_min) == P2_WARMUP_MIN)}


class P2LRController:

    def __init__(self, opt, U: int, W: int):
        self.U, self.W = int(U), int(W)
        self.keys = ["base_lr" if "base_lr" in g else "lr" for g in opt.param_groups]
        self.base = [float(g[k]) for g, k in zip(opt.param_groups, self.keys)]
        for g in opt.param_groups:
            if g.get("warmup"):
                raise ValueError("optimizer group still has its own warmup under P2")

    def apply(self, opt, s: int) -> float:
        f = p2_lr_factor(s, self.U, self.W)
        for g, k, b in zip(opt.param_groups, self.keys, self.base):
            g[k] = b * f
        return f

    def record(self) -> dict:
        return {"keys": list(self.keys), "base_lrs": list(self.base), "U": self.U, "W": self.W}


def resolve_module(name: str):
    from afterwm import legacy as LEGACY
    name = LEGACY.module_name(name)
    if "." in name:
        return importlib.import_module(name)
    for pkg in ("baselines", "afterwm"):
        try:
            return importlib.import_module(f"{pkg}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{pkg}.{name}":
                raise
    raise ModuleNotFoundError(f"model module {name!r} not found in baselines/ or afterwm/")


class TrainData:

    def __init__(self, stats, seed: int, device: str, data_dir: str = P.DATA_DIR, schedule_steps: int | None = None):
        obs, act = D.load_visible("train", data_dir)
        self.obs_n = stats.norm_obs(obs)
        self.act_n = stats.norm_act(act)
        self.n_episodes = int(obs.shape[0])
        if schedule_steps is None:
            self.schedule = D.make_schedule(seed, self.n_episodes)
        else:
            self.schedule = D.make_schedule(seed, self.n_episodes, steps=int(schedule_steps))
        self.device = device
        self.data_sha256 = R.sha256_file(D.split_path("train", data_dir))

    def windows(self, ep, t, horizon: int = P.MAX_H) -> dict:
        return D.windows(self.obs_n, self.act_n, np.asarray(ep), np.asarray(t), horizon)

    def to_torch(self, w: dict) -> dict:
        return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(self.device) for k, v in w.items()}

    def batch(self, step: int) -> dict:
        ep, tt = self.schedule
        return self.to_torch(self.windows(ep[step], tt[step]))

    def __getitem__(self, k):
        return {"obs_n": self.obs_n, "act_n": self.act_n, "schedule": self.schedule}[k]


def schedule_sha256(schedule) -> str:
    ep, tt = schedule
    return R.sha256_bytes((R.array_sha256(ep) + R.array_sha256(tt)).encode())


def check_prepared_schedule(seed: int, schedule) -> str | None:
    path = os.path.join(P.PREP_DIR, f"SCHEDULE_{seed}.npz")
    if not os.path.exists(path):
        return None
    z = R.load_npz(path)
    if not (np.array_equal(z["episode"], schedule[0]) and np.array_equal(z["t"], schedule[1])):
        raise RuntimeError(f"{path} differs from pipeline.data.make_schedule({seed})")
    return R.sha256_file(path)


_CURRENT = None


def current_context():
    if _CURRENT is None:
        raise RuntimeError("no active RunContext (custom_train must be called by pipeline.trainer)")
    return _CURRENT


def _cuda_mem(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                round(torch.cuda.max_memory_reserved() / 2 ** 20, 1))
    return (None, None)


def _metrics_to_float(metrics) -> dict:
    out = {}
    for k, v in (metrics or {}).items():
        try:
            if torch.is_tensor(v):
                v = v.detach().float()
                v = float(v.mean()) if v.numel() != 1 else float(v)
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


class RunContext:
    def __init__(self, out_dir, model_id, seed, steps, val_every, ckpt_every, module, cfg, device, protocol_sha):
        self.out_dir, self.model_id, self.seed = out_dir, model_id, int(seed)
        self.steps, self.val_every, self.ckpt_every = int(steps), int(val_every), int(ckpt_every)
        self.module, self.cfg, self.device, self.protocol_sha = module, cfg, device, protocol_sha
        self.best_val, self.best_step, self.last_val = math.inf, None, None
        self.best_state = None
        self.t0 = time.time()
        self.wall_prev = 0.0
        self.val_wall = 0.0
        self.recent = collections.deque(maxlen=50)
        self.host = socket.gethostname()
        self.p2 = None
        self.p2_controller = None

    def lr_factor(self, s: int) -> float:
        if self.p2 is None:
            raise RuntimeError("lr_factor() is only defined under --protocol p2")
        return p2_lr_factor(s, self.p2["U"], self.p2["W"])

    def p(self, name):
        return os.path.join(self.out_dir, name)

    def log_train(self, step, loss, metrics=None, step_seconds=None, grad_norm=None, lr=None, component=None,
                  lr_factor=None):
        alloc, reserved = _cuda_mem(self.device)
        rec = {"step": int(step), "loss": float(loss), "metrics": _metrics_to_float(metrics),
               "grad_norm": None if grad_norm is None else float(grad_norm), "lr": lr,
               "step_seconds": None if step_seconds is None else float(step_seconds),
               "max_mem_alloc_mib": alloc, "max_mem_reserved_mib": reserved,
               "peak_rss_mib": round(R.peak_rss_mib(), 1)}
        if component:
            rec["component"] = component
        if lr_factor is not None:
            rec["lr_factor"] = float(lr_factor)
        if step_seconds is not None:
            self.recent.append(float(step_seconds))
        R.append_jsonl(self.p("TRAIN.jsonl"), rec)

    def wall(self):
        return self.wall_prev + (time.time() - self.t0)

    def progress(self, step, phase="train", component=None, steps=None):
        steps = int(steps or self.steps)
        sps = float(np.median(self.recent)) if self.recent else None
        alloc, reserved = _cuda_mem(self.device)
        R.atomic_json(self.p("PROGRESS.json"), {
            "model_id": self.model_id, "seed": self.seed, "step": int(step), "steps": steps, "phase": phase,
            "component": component, "s_per_step_median50": sps,
            "eta_s": None if sps is None else sps * max(0, steps - int(step)),
            "wall_s": self.wall(), "val_wall_s": self.val_wall, "best_val": self.best_val, "best_step": self.best_step,
            "last_val": self.last_val, "pid": os.getpid(), "host": self.host, "updated": time.time(),
            "updated_iso": R.now_iso(), "max_mem_alloc_mib": alloc, "max_mem_reserved_mib": reserved})

    def validate(self, step, module, model, cfg, component=None) -> float:
        self.progress(step, phase="val", component=component)
        t = time.time()
        value, det = E.quick_val(module, model, cfg, self.device, return_details=True)
        self.val_wall += time.time() - t
        improved = math.isfinite(value) and value < self.best_val
        rec = {"step": int(step), "val": value, "finite": det["finite"], "per_h": det["per_h"],
               "seconds": det["seconds"], "improved": improved, "mixture": det["mixture"],
               "metric": P.VAL_SELECTION_METRIC}
        if component:
            rec["component"] = component
        R.append_jsonl(self.p("VAL.jsonl"), rec)
        self.last_val = value
        if improved:
            self.save_best(model, step, value)
        print(json.dumps({"stage": "val", "step": int(step), "val": value, "best_val": self.best_val,
                          "best_step": self.best_step, "seconds": round(det["seconds"], 2)}), flush=True)
        return value

    def save_best(self, model, step, value, extra=None):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        payload = {"model": state, "step": int(step), "val": float(value), "cfg": self.cfg,
                   "model_id": self.model_id, "seed": self.seed, "metric": P.VAL_SELECTION_METRIC,
                   "protocol_sha256": self.protocol_sha}
        if extra:
            payload.update(extra)
        R.atomic_torch_save(self.p("best.pt"), payload)
        self.best_val, self.best_step, self.best_state = float(value), int(step), state

    def save_checkpoint(self, payload: dict):
        if "step" not in payload:
            raise ValueError("checkpoint payload needs 'step'")
        payload = dict(payload)
        payload.update({"best_val": self.best_val, "best_step": self.best_step, "best_model": self.best_state,
                        "protocol_sha256": self.protocol_sha, "wall_s": self.wall(), "val_wall_s": self.val_wall,
                        "rng": E.save_rng(), "saved": R.now_iso()})
        R.atomic_torch_save(self.p("ckpt_last.pt"), payload)

    def load_checkpoint(self):
        path = self.p("ckpt_last.pt")
        if not os.path.exists(path):
            return None
        ck = R.torch_load(path, map_location="cpu")
        if ck.get("protocol_sha256") != self.protocol_sha:
            raise RuntimeError("ckpt_last.pt protocol sha mismatch")
        self.best_val = ck.get("best_val", math.inf)
        self.best_step = ck.get("best_step")
        self.best_state = ck.get("best_model")
        self.wall_prev = float(ck.get("wall_s", 0.0))
        self.val_wall = float(ck.get("val_wall_s", 0.0))
        return ck

    def reconcile_after_resume(self, step):
        for name in ("TRAIN.jsonl", "VAL.jsonl"):
            path = self.p(name)
            if not os.path.exists(path):
                continue
            rows = R.read_jsonl(path)
            keep = [r for r in rows if isinstance(r.get("step"), int) and r["step"] <= step]
            if len(keep) != len(rows):
                aside = R.move_aside(path, "pre_resume")
                R.atomic_write_bytes(path, "".join(json.dumps(r) + "\n" for r in keep).encode())
                print(json.dumps({"stage": "resume_log_trim", "file": name, "kept": len(keep),
                                  "dropped": len(rows) - len(keep), "old_copy": aside}), flush=True)
        best = self.p("best.pt")
        if os.path.exists(best):
            b = R.torch_load(best, map_location="cpu")
            if b.get("step") != self.best_step:
                aside = R.move_aside(best, "pre_resume")
                print(json.dumps({"stage": "resume_best_resync", "old_best_step": b.get("step"),
                                  "ckpt_best_step": self.best_step, "old_copy": aside}), flush=True)
                if self.best_state is not None:
                    R.atomic_torch_save(best, {"model": self.best_state, "step": self.best_step, "val": self.best_val,
                                               "cfg": self.cfg, "model_id": self.model_id, "seed": self.seed,
                                               "metric": P.VAL_SELECTION_METRIC, "protocol_sha256": self.protocol_sha})

    def complete(self, extra=None):
        if not os.path.exists(self.p("best.pt")):
            raise RuntimeError("best.pt missing; cannot write COMPLETE.json")
        b = R.torch_load(self.p("best.pt"), map_location="cpu")
        rec = {"status": "complete", "schema": SCHEMA, "model_id": self.model_id, "seed": self.seed,
               "module": getattr(self.module, "__name__", str(self.module)), "steps": self.steps,
               "best_step": b.get("step"), "best_val": b.get("val"), "metric": P.VAL_SELECTION_METRIC,
               "wall_s": self.wall(), "val_wall_s": self.val_wall, "protocol_sha256": self.protocol_sha,
               "best_pt_sha256": R.sha256_file(self.p("best.pt")), "finished": R.now_iso(), "host": self.host}
        if extra:
            rec.update(extra)
        R.atomic_json(self.p("COMPLETE.json"), rec)
        self.progress(self.steps, phase="done")
        return rec


class _Stop:
    flag = False
    signum = None


def _on_signal(signum, frame):
    _Stop.flag = True
    _Stop.signum = signum


def _protocol_core(module, args, cfg, variant, steps, val_every, ckpt_every, data_shas, stats_sha, sched_sha,
                   custom: bool, lr_info: dict, p2rec: dict | None = None) -> dict:
    core = {
        "schema": SCHEMA, "module": module.__name__, "model_id": args.model_id, "seed": int(args.seed),
        "budget_class": args.budget_class, "variant": variant, "cfg": cfg, "steps": int(steps),
        "val_every": int(val_every), "ckpt_every": int(ckpt_every), "batch": P.BATCH,
        "custom_train": bool(custom), "data_sha256": data_shas, "stats_sha256": stats_sha,
        "schedule_sha256": sched_sha, "selection_metric": P.VAL_SELECTION_METRIC,
        "deterministic": bool(getattr(args, "deterministic", False)),
        "lr_mult": lr_info.get("lr_mult"), "native_lr": lr_info.get("native_lr"),
        "protocol_constants": R.to_jsonable({k: getattr(P, k) for k in PROTOCOL_CONSTANTS}),
        "imported_code_sha256": R.imported_code_files(),
    }
    if p2rec is not None:
        core["train_protocol"] = p2rec
    return core


def load_variant_file(path: str) -> dict:
    obj = R.read_json(path)
    if obj is None:
        raise FileNotFoundError(f"--variant-file {path} is missing or unreadable")
    if not isinstance(obj, dict):
        raise ValueError(f"--variant-file {path} must contain a JSON object")
    var = obj.get("variant", {})
    if not isinstance(var, dict):
        raise ValueError(f"--variant-file {path}: 'variant' must be an object")
    if "variant" not in obj and "lr_mult" not in obj:
        raise ValueError(f"--variant-file {path} has neither 'variant' nor 'lr_mult'")
    tp = obj.get("train_protocol", "p1")
    if tp not in PROTOCOLS:
        raise ValueError(f"--variant-file {path}: train_protocol {tp!r} is not one of {PROTOCOLS}")
    out = {"variant": dict(var), "path": os.path.abspath(path), "sha256": R.sha256_file(path),
           "train_protocol": tp}
    for k in ("lr_mult", "lr", "native_lr"):
        if obj.get(k) is not None:
            v = obj[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"--variant-file {path}: {k} must be a number, got {v!r}")
            out[k] = v
    return out


def merge_variants(cli: dict, from_file: dict) -> dict:
    out = dict(cli)
    for k, v in from_file.items():
        if k in out and out[k] != v:
            raise ValueError(f"variant key {k!r}: --variant gives {out[k]!r} but --variant-file gives {v!r}")
        out[k] = v
    return out


def apply_lr_rule(module, budget_class: str, variant: dict, lr_mult=None, vf: dict | None = None):

    cfg = module.default_config(budget_class, **variant)
    native = cfg.get("lr")
    sweep = cfg.get("lr_sweep", True)
    info = {"native_lr": native, "lr_mult": None, "lr": native, "lr_sweep": sweep}
    if lr_mult is not None:
        m = float(lr_mult)
        if not (math.isfinite(m) and m > 0):
            raise ValueError(f"lr_mult must be a positive finite number, got {lr_mult!r}")
        if "lr" in variant:
            raise ValueError("lr_mult cannot be combined with an explicit 'lr' variant")
        if sweep is False and m != 1.0:
            raise ValueError(f"{module.__name__}: default_config lr_sweep is False; only lr_mult 1 is allowed")
        if native is None:
            if m != 1.0:
                raise ValueError(f"{module.__name__}: cfg has no 'lr'; cannot apply lr_mult {m}")
        else:
            if isinstance(native, bool) or not isinstance(native, (int, float)):
                raise ValueError(f"{module.__name__}: cfg['lr'] must be a number, got {native!r}")
            cfg["lr"] = float(native) * m
        info.update(lr_mult=m, lr=cfg.get("lr"))
    if vf is not None:
        if "native_lr" in vf and vf["native_lr"] != native:
            raise ValueError(f"variant file native_lr {vf['native_lr']!r} != module native lr {native!r} "
                             "(module default changed since the LR was selected?)")
        if "lr" in vf and vf["lr"] != cfg.get("lr"):
            raise ValueError(f"variant file lr {vf['lr']!r} != resulting cfg lr {cfg.get('lr')!r}")
    return cfg, info


def _diff_keys(a: dict, b: dict, prefix="") -> list:
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out.append(prefix + k)
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out += _diff_keys(a[k], b[k], prefix + k + ".")
        elif a[k] != b[k]:
            out.append(prefix + k)
    return out


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args) -> int:
    global _CURRENT
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    out_dir = os.path.abspath(args.out or os.path.join(P.RUNS_DIR, args.model_id, str(args.seed)))
    os.makedirs(out_dir, exist_ok=True)
    try:
        lock = R.FileLock(os.path.join(out_dir, ".lock")).acquire()
    except R.LockBusy as exc:
        print(json.dumps({"status": "lock_busy", "reason": str(exc)}), flush=True)
        return EXIT_LOCK_BUSY
    orig_labels = D.load_analysis_labels
    try:
        if os.path.exists(os.path.join(out_dir, "COMPLETE.json")):
            print(json.dumps({"status": "refused", "reason": f"{out_dir}/COMPLETE.json exists"}), flush=True)
            return EXIT_ALREADY_COMPLETE
        if os.path.exists(os.path.join(out_dir, "DIVERGED.json")):
            print(json.dumps({"status": "refused", "reason": f"{out_dir}/DIVERGED.json exists (diverged runs are "
                                                             "never resumed or retried)"}), flush=True)
            return EXIT_DIVERGED
        return _train_locked(args, device, out_dir)
    except FloatingPointError as exc:
        _record_failure(out_dir, exc, diverged=True)
        return EXIT_DIVERGED
    except Exception as exc:
        _record_failure(out_dir, exc, diverged=False)
        return 1
    finally:
        D.load_analysis_labels = orig_labels
        _CURRENT = None
        lock.release()


def _record_failure(out_dir, exc, diverged: bool):
    path, n = R.next_free_path(os.path.join(out_dir, "TRAIN_FAILED_{n}.json"))
    R.atomic_json(path, {"status": "diverged" if diverged else "exception", "exception": repr(exc),
                         "traceback": traceback.format_exc(), "time": R.now_iso(), "host": socket.gethostname()})
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    if not diverged:
        return
    ctx = _CURRENT
    rec = {"status": "diverged", "reason": str(exc), "exception": repr(exc), "step": getattr(exc, "step", None),
           "loss": None if getattr(exc, "loss", None) is None else repr(exc.loss), "train_failed_file": path,
           "time": R.now_iso(), "host": socket.gethostname()}
    if ctx is not None:
        rec.update({"model_id": ctx.model_id, "seed": ctx.seed, "best_step": ctx.best_step,
                    "best_val": ctx.best_val if math.isfinite(ctx.best_val) else None, "last_val": ctx.last_val,
                    "protocol_sha256": ctx.protocol_sha, "wall_s": ctx.wall(), "lr": (ctx.cfg or {}).get("lr")})
    R.atomic_json(os.path.join(out_dir, "DIVERGED.json"), rec)
    print(json.dumps({"stage": "diverged", **{k: rec.get(k) for k in ("reason", "step", "best_step", "best_val")}}),
          flush=True)


def _train_locked(args, device, out_dir) -> int:
    global _CURRENT
    module = resolve_module(args.module)
    vf = load_variant_file(args.variant_file) if getattr(args, "variant_file", None) else None
    variant = R.parse_variant(args.variant)
    if vf is not None:
        variant = merge_variants(variant, vf["variant"])
    lr_mult = getattr(args, "lr_mult", None)
    if vf is not None and "lr_mult" in vf:
        if lr_mult is not None and float(lr_mult) != float(vf["lr_mult"]):
            raise ValueError(f"--lr-mult {lr_mult} conflicts with the variant file lr_mult {vf['lr_mult']}")
        lr_mult = float(vf["lr_mult"])
    proto = getattr(args, "protocol", None) or "p1"
    if proto not in PROTOCOLS:
        raise ValueError(f"--protocol must be one of {PROTOCOLS}, got {proto!r}")
    if vf is not None and vf.get("train_protocol", "p1") != proto:
        raise ValueError(f"--variant-file {vf['path']} was written for protocol {vf.get('train_protocol')!r}, "
                         f"but this run is --protocol {proto}")
    if proto == "p2":
        U = int(getattr(args, "p2_horizon", None) or P2_U)
        warm_min = int(getattr(args, "p2_warmup_min", None) or P2_WARMUP_MIN)
        steps = U if args.steps is None else int(args.steps)
        if not 1 <= steps <= U:
            raise ValueError(f"--steps must be in [1, {U}] under --protocol p2")
        val_every = int(args.val_every or P2_VAL_EVERY)
        sched_steps = U
    else:
        if getattr(args, "p2_horizon", None) or getattr(args, "p2_warmup_min", None):
            raise ValueError("--p2-horizon / --p2-warmup-min need --protocol p2")
        steps = P.MAX_STEPS if args.steps is None else int(args.steps)
        if not 1 <= steps <= P.MAX_STEPS:
            raise ValueError(f"--steps must be in [1, {P.MAX_STEPS}]")
        val_every = int(args.val_every or P.VAL_EVERY)
        sched_steps = None
    ckpt_every = int(args.ckpt_every or P.CKPT_EVERY)
    custom = hasattr(module, "custom_train")
    stats = E.get_stats()
    data = TrainData(stats, args.seed, device, schedule_steps=sched_steps)
    sched_sha = schedule_sha256(data.schedule)
    prep_sched = check_prepared_schedule(args.seed, data.schedule) if proto == "p1" else None
    val_sha = R.sha256_file(D.split_path("val"))
    D.load_analysis_labels = _blocked_labels
    cfg, lr_info = apply_lr_rule(module, args.budget_class, variant, lr_mult, vf)
    p2rec = resolve_p2(module, cfg, U, warm_min) if proto == "p2" else None
    if p2rec is not None:
        p2rec["val_every"] = int(val_every)
        p2rec["steps"] = int(steps)
    bad = R.json_exactness_problems(cfg)
    if bad:
        raise ValueError(f"default_config must return JSON-exact values (use None/strings instead): {bad}")
    cfg = json.loads(R.canonical_json(cfg))
    _set_seeds(args.seed)
    model = module.build(cfg, args.seed, device, stats)
    opt = sched = None
    p2c = None
    if not custom:
        opt = module.make_optimizer(model, cfg)
        sched = module.make_scheduler(opt, cfg) if hasattr(module, "make_scheduler") else None
        if p2rec is not None:
            p2c = P2LRController(opt, p2rec["U"], p2rec["W"])
            p2rec["lr_groups"] = p2c.record()
    core = _protocol_core(module, args, cfg, variant, steps, val_every, ckpt_every,
                          {"train": data.data_sha256, "val": val_sha}, stats.sha256(), sched_sha, custom, lr_info,
                          p2rec)
    psha = R.sha256_json(core)
    ppath = os.path.join(out_dir, "PROTOCOL.json")
    existing = R.read_json(ppath)
    if existing is not None:
        if existing.get("protocol_sha256") != psha:
            diff = _diff_keys(existing.get("core", {}), json.loads(R.canonical_json(core)))
            print(json.dumps({"status": "protocol_mismatch", "existing": existing.get("protocol_sha256"),
                              "new": psha, "differing_keys": diff[:50]}), flush=True)
            return EXIT_PROTOCOL_MISMATCH
        R.append_jsonl(os.path.join(out_dir, "RESUMES.jsonl"), {"time": R.now_iso(), "env": R.env_info(device),
                                                                "argv": sys.argv})
    else:
        R.atomic_json(ppath, {**core, "core": core, "protocol_sha256": psha, "code_files": R.code_manifest(),
                              "prepared_schedule_npz_sha256": prep_sched, "out_dir": out_dir,
                              "argv": sys.argv, "created": R.now_iso(), "env": R.env_info(device),
                              "gpu_driver": R.gpu_driver(), "label_guard": True,
                              "determinism": getattr(args, "determinism_info", None),
                              "lr_rule": {**lr_info, "rule": "cfg['lr'] = default_config lr x lr_mult (trainer level)"},
                              "variant_file": None if vf is None else dict(vf)})
    fm = module.forecaster(model, cfg)
    ipath = os.path.join(out_dir, "INITIALIZATION.json")
    if not os.path.exists(ipath):
        rep = fm.parameter_report()
        R.atomic_json(ipath, {"parameter_report": rep, "check": PC.check_report(rep, model, args.budget_class),
                              "optimizer_coverage": PC.optimizer_coverage(model, opt) if opt is not None else None,
                              "state_hash_init": E.state_hash(model), "model_repr": repr(model)[:20000],
                              "forecaster": {"name": getattr(fm, "name", None), "family": getattr(fm, "family", None),
                                             "budget_class": getattr(fm, "budget_class", None),
                                             "needs_grad_at_eval": getattr(fm, "needs_grad_at_eval", None),
                                             "stochastic": getattr(fm, "stochastic", None)}})
    ctx = RunContext(out_dir, args.model_id, args.seed, steps, val_every, ckpt_every, module, cfg, device, psha)
    ctx.p2, ctx.p2_controller = p2rec, p2c
    _CURRENT = ctx
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    if custom:
        print(json.dumps({"stage": "custom_train_start", "module": module.__name__}), flush=True)
        module.custom_train(cfg, args.seed, out_dir, device, stats, data.schedule, data)
        if not os.path.exists(ctx.p("COMPLETE.json")):
            ctx.complete({"custom_train": True})
        return 0
    return _standard_loop(ctx, module, model, opt, sched, cfg, data, steps)


def _blocked_labels(*a, **k):
    raise RuntimeError("hidden labels are not available to models/trainers (trainer label guard)")


def _standard_loop(ctx, module, model, opt, sched, cfg, data, steps) -> int:
    start = 0
    ck = ctx.load_checkpoint()
    if ck is not None:
        if hasattr(module, "load_checkpoint"):
            module.load_checkpoint(model, ck, cfg)
        else:
            model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck.get("scheduler") is not None:
            sched.load_state_dict(ck["scheduler"])
        E.restore_rng(ck["rng"])
        start = int(ck["step"])
        print(json.dumps({"stage": "resumed", "step": start, "best_step": ctx.best_step,
                          "best_val": ctx.best_val}), flush=True)
    ctx.reconcile_after_resume(start)
    clip = cfg.get("grad_clip")
    trainable = [p for p in model.parameters() if p.requires_grad]
    has_after = hasattr(module, "after_step")
    p2c = ctx.p2_controller
    model.train()
    ctx.progress(start)

    def _ckpt(step_done):
        ctx.save_checkpoint({"model": model.state_dict(), "optimizer": opt.state_dict(),
                             "scheduler": sched.state_dict() if sched is not None else None, "step": step_done})

    for i in range(start, steps):
        t0 = time.time()
        batch = data.batch(i)
        loss, metrics = module.train_step(model, batch, i, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            gn = torch.nn.utils.clip_grad_norm_(trainable, float(clip))
        else:
            gn = torch.nn.utils.clip_grad_norm_(trainable, float("inf"))
        lr_f = p2c.apply(opt, i + 1) if p2c is not None else None
        opt.step()
        if sched is not None:
            sched.step()
        if has_after:
            module.after_step(model, i, cfg)
        loss_f = float(loss.detach())
        if not math.isfinite(loss_f):
            raise Diverged(f"nonfinite loss at update {i + 1}: {loss_f}", step=i + 1, loss=loss_f)
        gn_f = float(gn)
        metrics = _metrics_to_float(metrics)
        if str(ctx.device).startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
        s = i + 1
        ctx.log_train(s, loss_f, metrics, dt, gn_f, opt.param_groups[0].get("lr"), lr_factor=lr_f)
        is_val = (s % ctx.val_every == 0) or s == steps
        if is_val:
            ctx.validate(s, module, model, cfg)
        if is_val or s % ctx.ckpt_every == 0 or s == steps or _Stop.flag:
            _ckpt(s)
        if s % 10 == 0 or is_val:
            ctx.progress(s)
        if _Stop.flag and s < steps:
            print(json.dumps({"stage": "terminated", "signal": _Stop.signum, "checkpoint_step": s}), flush=True)
            return EXIT_TERMINATED
    if ctx.best_step is None:
        raise Diverged(f"no finite VAL value in {steps} updates (best.pt never written)", step=steps)
    ctx.complete()
    print(json.dumps({"stage": "complete", "best_step": ctx.best_step, "best_val": ctx.best_val,
                      "wall_s": round(ctx.wall(), 1)}), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Train one model on one dataset (stage 2 for AFTER-WM).', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget-class", required=True, choices=sorted(PC.BUDGETS))
    ap.add_argument("--variant", nargs="*", default=[], help="key=value pairs passed to default_config")
    ap.add_argument("--variant-file", default=None,
                    help="JSON file: 'variant' dict merged into --variant, 'lr_mult' used as --lr-mult, "
                         "'lr'/'native_lr' cross-checked (e.g. configs/models/<dataset>/<model>.json)")
    ap.add_argument("--lr-mult", type=float, default=None, help="cfg['lr'] = native default_config lr x M")
    ap.add_argument("--deterministic", action="store_true",
                    help="CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms (default: off)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=None,
                    help=f"updates to run (default: {P.MAX_STEPS} under p1, U = {P2_U} under p2)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-every", type=int, default=None, help="override (tests/smoke only)")
    ap.add_argument("--ckpt-every", type=int, default=None, help="override (tests/smoke only)")
    ap.add_argument("--protocol", choices=PROTOCOLS, default="p1",
                    help="p1 = fixed-step protocol (GPMM); p2 = 16,000 updates, linear warm-up and cosine decay (all SGD models)")
    ap.add_argument("--p2-horizon", type=int, default=None, help=f"P2 U override (tests only; official {P2_U})")
    ap.add_argument("--p2-warmup-min", type=int, default=None,
                    help=f"P2 minimum warmup override (tests only; official {P2_WARMUP_MIN})")
    ap.add_argument("--rssm-eval-mode", default=None, choices=["mode", "sample16"],
                    help="evaluation mode of the RSSM used for VAL selection during training")
    P.add_cli_options(ap)
    args = ap.parse_args(argv)
    if args.rssm_eval_mode is not None:
        from baselines import keyed_sampling
        keyed_sampling.set_eval_mode(args.rssm_eval_mode)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.determinism_info = E.configure_determinism(args.deterministic)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
