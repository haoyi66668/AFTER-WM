from __future__ import annotations

import torch

EVAL_MODES = ("mode", "sample16")
N_SAMPLES = 16
KEYED_CHUNK = 48
PHASE_OBSERVE, PHASE_IMAGINE = 1, 2
_MOD = 2147483647


_EVAL_MODE = ["mode"]


def set_eval_mode(m: str) -> None:
    if m not in EVAL_MODES:
        raise ValueError(f"RSSM evaluation mode {m!r} is not one of {EVAL_MODES}")
    _EVAL_MODE[0] = m


def eval_mode() -> str:
    return _EVAL_MODE[0]


def _mix(x: torch.Tensor) -> torch.Tensor:
    for a in (48271, 69621, 16807):
        x = (x * a) % _MOD
        x = (x ^ (x >> 11)) % _MOD
    return x


def query_sample_seeds(keys: torch.Tensor, n_samples: int, phase: int) -> torch.Tensor:
    dev = keys.device
    x = _mix((keys.to(torch.long).remainder(_MOD) + 1) % _MOD)
    x = _mix((x + 7919 * int(phase)) % _MOD)
    s = torch.arange(n_samples, device=dev, dtype=torch.long)
    x = _mix((x[:, None] + 104729 * (s[None, :] + 1)) % _MOD)
    return x.reshape(-1)


def gumbel_step(seeds: torch.Tensor, t: int, S: int, K: int, dtype=torch.float32) -> torch.Tensor:
    dev = seeds.device
    x = _mix((seeds + 1299709 * (int(t) + 1)) % _MOD)
    i = torch.arange(S, device=dev, dtype=torch.long)
    k = torch.arange(K, device=dev, dtype=torch.long)
    x = _mix((x[:, None] + 15485863 % _MOD * (i[None, :] + 1)) % _MOD)
    x = _mix((x[:, :, None] + 32452843 % _MOD * (k[None, None, :] + 1)) % _MOD)
    u = (x.double() + 0.5) / float(_MOD)
    return (-torch.log(-torch.log(u))).to(dtype)


def chunked_keyed(fn, q, chunk: int, dev) -> torch.Tensor:
    tens = [q.ctx_obs.to(dev, torch.float32), q.ctx_act.to(dev, torch.float32), q.fut_act.to(dev, torch.float32),
            q.key.to(dev, torch.long)]
    n_all = tens[0].shape[0]
    outs = []
    for s in range(0, n_all, chunk):
        part = [x[s:s + chunk] for x in tens]
        n = part[0].shape[0]
        if n < chunk:
            part = [torch.cat([x, x[-1:].expand(chunk - n, *x.shape[1:])], 0) for x in part]
        outs.append(fn(*part)[:n])
    return torch.cat(outs, 0)
