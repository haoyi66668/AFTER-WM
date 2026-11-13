from __future__ import annotations

import torch

from pipeline.eval_api import QueryBatch
from afterwm.core import strict_loss


def loss_from_pred(model, pred, fut_obs):
    logw = pred.new_zeros(pred.shape[0], 1)
    return strict_loss(pred[:, :, None, :], logw, fut_obs, model.loss_own_std, model.loss_common_std,
                       model.raw_scale)


def fixed_chunk_forecast(fn, q: QueryBatch, chunk: int, dev) -> torch.Tensor:
    tens = [q.ctx_obs.to(dev, torch.float32), q.ctx_act.to(dev, torch.float32), q.fut_act.to(dev, torch.float32)]
    n_all = tens[0].shape[0]
    outs = []
    for s in range(0, n_all, chunk):
        part = [x[s:s + chunk] for x in tens]
        n = part[0].shape[0]
        if n < chunk:
            part = [torch.cat([x, x[-1:].expand(chunk - n, *x.shape[1:])], 0) for x in part]
        outs.append(fn(*part)[:n])
    return torch.cat(outs, 0)
