from __future__ import annotations

import torch
import torch.nn.functional as F

from baselines import tdmpc2 as T
from pipeline import registry as P

NAME = "tdmpc2_h40"
FAMILY = "baseline"
DEC_WEIGHTINGS = ("uniform", "rho")

param_count = T.param_count
build = T.build
make_optimizer = T.make_optimizer
fixed_chunk_forecast = T.fixed_chunk_forecast
TDMPC2WorldModel = T.TDMPC2WorldModel


def default_config(budget_class: str = "baseline", **variant) -> dict:
    variant = dict(variant)
    horizon = int(variant.pop("horizon", 40))
    dec_weighting = str(variant.pop("dec_weighting", "uniform"))
    if dec_weighting not in DEC_WEIGHTINGS:
        raise ValueError(f"{NAME}: dec_weighting must be one of {DEC_WEIGHTINGS}, got {dec_weighting!r}")
    if not 1 <= horizon <= P.CONTEXT + P.MAX_H - 1:
        raise ValueError(f"{NAME}: horizon must be in [1, {P.CONTEXT + P.MAX_H - 1}], got {horizon}")
    cfg = T.default_config(budget_class, **variant)
    cfg.update(name=NAME, horizon=horizon, dec_weighting=dec_weighting, base_module="baselines.tdmpc2")
    cfg["param_count_formula"] = param_count(cfg)
    return cfg


def train_step(model, batch: dict, step: int, cfg: dict):
    model.train()
    obs = batch["win_obs"]
    act = batch["win_act"]
    Hz = int(cfg["horizon"])
    rho = float(cfg["rho"])
    uniform = cfg.get("dec_weighting", "uniform") == "uniform"
    B, Tw, _ = obs.shape
    n_sub = Tw - Hz
    Z = model.encode(obs)
    Zt = Z.detach()
    z = Z[:, :n_sub]
    zs = [z]
    cons = 0.0
    for t in range(Hz):
        z = model.next(z, act[:, t:t + n_sub])
        cons = cons + F.mse_loss(z, Zt[:, t + 1:t + 1 + n_sub]) * rho ** t
        zs.append(z)
    cons = cons / Hz
    pred = model.decode(torch.stack(zs, 0))
    dec = 0.0
    for t in range(Hz + 1):
        w_t = 1.0 if uniform else rho ** t
        dec = dec + F.mse_loss(pred[t], obs[:, t:t + n_sub]) * w_t
    dec = dec / (Hz + 1)
    loss = cfg["consistency_coef"] * cons + cfg["dec_coef"] * dec
    return loss, T._floats(loss=loss, consistency=cons, decoder=dec)


class _Forecaster(T._Forecaster):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.name = NAME

    def parameter_report(self) -> dict:
        return parameter_report(self.model, self.cfg)


def forecaster(model, cfg):
    return _Forecaster(model, cfg)


def parameter_report(model, cfg) -> dict:
    rep = T.parameter_report(model, cfg)
    rep["note"] = rep.get("note", "") + "; tdmpc2_h40 = tdmpc2 architecture, horizon-40 objective"
    return rep
