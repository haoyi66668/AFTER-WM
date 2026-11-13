from __future__ import annotations

import torch

from .. import core as C
from .. import model as T

NAME = "afterwm_noerr"
FAMILY = "afterwm"
WIDTHS = T.WIDTHS


def default_config(budget_class: str = "afterwm_eq", **variant) -> dict:
    cfg = T.default_config(budget_class, **variant)
    cfg["name"] = NAME
    cfg["model_id"] = cfg["model_id"].replace("afterwm", "afterwm_no_error_input", 1)
    cfg["online_error_input"] = "zeros"
    return cfg


class _ZeroErrorNextState:
    __slots__ = ("z_hat",)

    def __init__(self):
        self.z_hat = None

    def _get(self):
        if self.z_hat is None:
            raise RuntimeError("z_hat was not captured before the error was formed")
        return self.z_hat

    def __getitem__(self, idx):
        return self._get()

    def unsqueeze(self, dim):
        return self._get()


class AfterWMNoErrModel(T.AfterWMModel):
    family = "afterwm"

    def observe_fast(self, state, z, a, zn):
        fp = self.fp
        proxy = _ZeroErrorNextState()
        had = "experts_zhat" in vars(fp)
        prev = vars(fp).get("experts_zhat")
        orig = fp.experts_zhat

        def capture(*args, **kw):
            out = orig(*args, **kw)
            proxy.z_hat = out
            return out
        fp.experts_zhat = capture
        try:
            return T.AfterWMModel.observe_fast(self, state, z, a, proxy)
        finally:
            if had:
                fp.experts_zhat = prev
            else:
                del fp.experts_zhat


def build(cfg: dict, seed: int, device, stats) -> AfterWMNoErrModel:
    model = T.build(cfg, seed, device, stats)
    model.__class__ = AfterWMNoErrModel
    return model


make_optimizer = T.make_optimizer
train_step = T.train_step


def forecaster(model, cfg) -> C.MemoryForecaster:
    return C.MemoryForecaster(model, cfg.get("model_id", NAME), cfg["budget_class"])


C.register_family(NAME, __import__(__name__, fromlist=["*"]))
