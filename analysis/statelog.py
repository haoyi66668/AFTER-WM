from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

STATE = ("q", "F", "c", "P")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def check_model(model) -> None:
    if str(model.family) != "afterwm":
        raise SystemExit(f"family {model.family} is not afterwm")
    if not bool(getattr(model.fmap, "identity", False)):
        raise RuntimeError("only identity frame maps are supported")


def as_dict(state) -> dict:
    return dict(zip(STATE, state))


def from_dict(d) -> tuple:
    return tuple(d[k] for k in STATE)


@torch.no_grad()
def one_step_parts(model, state, z, a):
    fp = model.fp
    d = as_dict(state)
    joint = torch.cat((z, a), dim=-1)
    _, p = fp.route(joint, d["q"])
    K = p.shape[-1]
    base = torch.stack([z + fp.g0[k](joint) for k in range(K)], dim=-2)
    F = d["F"]
    d_fast = torch.stack([fp.gf[k](torch.cat((z, a, F[:, k]), -1))
                          - fp.gf[k](torch.cat((z, a, torch.zeros_like(F[:, k])), -1)) for k in range(K)], dim=-2)
    m, c = model.core, d["c"]
    d_slow = torch.stack([m.readout[k](torch.cat((z, a, F[:, k], c[:, k]), -1))
                          - m.readout[k](torch.cat((z, a, F[:, k], torch.zeros_like(c[:, k])), -1))
                          for k in range(K)], dim=-2)
    return {"p": p, "base": base, "d_fast": d_fast, "d_slow": d_slow}


@torch.no_grad()
def replay(model, zc, ac):
    state = model.init_state(zc.shape[0], zc)
    for j in range(ac.shape[1]):
        state = model.observe_fast(state, zc[:, j], ac[:, j], zc[:, j + 1])
    return state


def forecast_arrays_from(model, pred, logp):
    logw = torch.log_softmax(logp.double(), -1)
    mean_native = (logw.exp()[:, None, :, None] * pred.double()).sum(2)
    comp = model.fmap.obs_to_train(pred)
    mean = model.fmap.obs_to_train(mean_native).float()
    return comp.float().cpu().numpy(), logw.float().cpu().numpy(), mean.cpu().numpy()


def variant_states(state) -> dict:
    d = as_dict(state)
    out = {"intact": state}
    z = dict(d)
    z["c"] = torch.zeros_like(d["c"])
    out["slow0"] = from_dict(z)
    z = dict(d)
    z["F"] = torch.zeros_like(d["F"])
    out["fast0"] = from_dict(z)
    z["c"] = torch.zeros_like(d["c"])
    out["base"] = from_dict(z)
    return out


def step_errors(comp, logw, y, own, common):
    mu = torch.from_numpy(np.asarray(comp)).double()
    lp = torch.log_softmax(torch.from_numpy(np.asarray(logw)).double(), -1)
    yhat = (lp.exp()[:, None, :, None] * mu).sum(2).numpy()
    e = (yhat - np.asarray(y, np.float64)) * own / common
    return (e * e).mean(-1)


class Row:
    def __init__(self, run, eval_dir, device):
        from pipeline import evaluate as E
        self.run = run
        self.eval_dir = eval_dir or os.path.join(run, "eval")
        eprot = json.load(open(os.path.join(self.eval_dir, "EVALUATION_PROTOCOL.json")))
        det = eprot.get("determinism")
        self.deterministic = bool(det.get("deterministic")) if isinstance(det, dict) else False
        self.runtime = E.setup_runtime(device, self.deterministic)
        torch.set_grad_enabled(False)
        comp = json.load(open(os.path.join(run, "COMPLETE.json")))
        self.best_sha = sha256_file(os.path.join(run, "best.pt"))
        if comp.get("best_pt_sha256") not in (None, self.best_sha):
            raise SystemExit("best.pt sha256 differs from COMPLETE.json")
        self.mod, self.model, self.cfg, self.prot, self.stats, _ = E.load_run(run, device=device)
        check_model(self.model)

    def chunk(self, split):
        r = json.load(open(os.path.join(self.eval_dir, split, "RESULT.json")))
        return int(r.get("chunk_used") or 720)

    def stored(self, split, keys=("comp_mean", "comp_logw", "mean", "query_id", "e", "t")):
        with np.load(os.path.join(self.eval_dir, split, "PREDICTIONS.npz"), allow_pickle=False) as z:
            return {k: np.asarray(z[k]) for k in keys}
