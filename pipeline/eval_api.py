from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class QueryBatch:
    ctx_obs: torch.Tensor
    ctx_act: torch.Tensor
    fut_act: torch.Tensor
    key: torch.Tensor


@dataclass
class Forecast:
    mean: torch.Tensor
    comp_mean: Optional[torch.Tensor] = None
    comp_logw: Optional[torch.Tensor] = None
    comp_var: Optional[torch.Tensor] = None


@runtime_checkable
class ForecastModel(Protocol):
    name: str
    family: str
    budget_class: str
    needs_grad_at_eval: bool
    stochastic: bool

    def forecast(self, q: QueryBatch) -> Forecast:
        ...

    def parameter_report(self) -> dict:
        ...


def mixture_mean(comp_mean: torch.Tensor, comp_logw: torch.Tensor) -> torch.Tensor:
    w = torch.softmax(comp_logw, dim=-1)
    return (comp_mean * w[:, None, :, None]).sum(dim=2)
