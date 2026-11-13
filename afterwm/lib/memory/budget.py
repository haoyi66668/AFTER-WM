from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping

SCHEMA = "strict_mtx_fairness_v1"
ORIGINAL_TARGET = 415243
TARGET = 390581
EXPECTED_BASE_TENSORS = 34
SHARED_GROUPS = ("Uq", "UF", "GF")
NORMALIZER_FIELDS = (
    "observation_mean", "observation_std", "action_mean", "action_std",
    "mean", "std", "obs_mean", "obs_std", "act_mean", "act_std",
)


def _torch():
    import torch
    return torch


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def seed_for(seed: int, group: str, local_name: str = "") -> int:
    if type(seed) is not int or not isinstance(group, str):
        raise TypeError("integer seed and string group required")
    raw = _canonical([SCHEMA, seed, group, local_name])
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


@contextmanager
def seeded_rng(seed: int):
    torch = _torch()
    import numpy as np
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(int(seed)); np.random.seed(int(seed) % (2**32))
            torch.manual_seed(int(seed))
            if devices:
                torch.cuda.manual_seed_all(int(seed))
            yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state)


def _tensor_description(tensor) -> dict[str, Any]:
    torch = _torch()
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    shape, dtype = list(value.shape), str(value.dtype)
    digest = hashlib.sha256(_canonical([shape, dtype]) + raw).hexdigest()
    result = {"shape": shape, "dtype": dtype, "numel": value.numel(), "sha256": digest}
    if value.is_floating_point() or value.is_complex():
        finite = bool(torch.isfinite(value).all())
        result["finite"] = finite
        if not finite:
            raise FloatingPointError("nonfinite tensor in fairness snapshot")
        if not value.is_complex() and value.numel():
            x = value.double()
            result.update(mean=float(x.mean()), std=float(x.std(unbiased=False)),
                          min=float(x.min()), max=float(x.max()),
                          nonzero_scalars=int(torch.count_nonzero(value)))
    return result


def _mapping_fingerprint(descriptions: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(descriptions)).hexdigest()


def _normalizer_value(value):
    torch = _torch()
    import numpy as np
    if torch.is_tensor(value):
        return {"tensor": _tensor_description(value)}
    if isinstance(value, np.ndarray):
        x = np.ascontiguousarray(value)
        if x.dtype.kind not in "biufc":
            raise TypeError("normalizer arrays must be numeric")
        if not np.isfinite(x).all():
            raise FloatingPointError("nonfinite normalizer")
        return {"shape": list(x.shape), "dtype": str(x.dtype),
                "sha256": hashlib.sha256(x.tobytes()).hexdigest()}
    if isinstance(value, Mapping):
        return {str(k): _normalizer_value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [_normalizer_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError("nonfinite normalizer scalar")
        return value
    fields = {name: getattr(value, name) for name in NORMALIZER_FIELDS if hasattr(value, name)}
    if not fields and hasattr(value, "state_dict"):
        fields = dict(value.state_dict())
    if not fields:
        raise TypeError("unrecognized normalizer; pass an explicit numeric dict")
    return _normalizer_value(fields)


def parameter_manifest(model, *, target: int = TARGET, require_training: bool = True):
    declared = list(model.declared_trainable_named())
    ids = [id(p) for _, p in declared]
    if len(ids) != len(set(ids)):
        raise AssertionError("declared trainable parameter aliases/duplicates")
    groups = model.learning_modules()
    owner: dict[int, str] = {}
    for group, module in groups.items():
        for p in module.parameters():
            if id(p) in owner and owner[id(p)] != group:
                raise AssertionError(f"trainable parameter shared by {owner[id(p)]}/{group}")
            owner[id(p)] = group
    if set(ids) != set(owner):
        raise AssertionError("declared scope differs from learning_modules")
    all_named = dict(model.named_parameters())
    rows = []
    for name, p in declared:
        if name not in all_named or all_named[name] is not p:
            raise AssertionError("declared parameter is not a module parameter: " + name)
        if require_training and not p.requires_grad:
            raise AssertionError("declared parameter not trainable: " + name)
        rows.append({"name": name, "group": owner[id(p)], **_tensor_description(p)})
    foreign = [n for n, p in all_named.items() if p.requires_grad and id(p) not in owner]
    if foreign:
        raise AssertionError("undeclared trainable parameters: " + repr(foreign))
    count = sum(row["numel"] for row in rows)
    if count != target:
        raise AssertionError(f"trainable count {count} != {target}")
    by_group = {g: sum(r["numel"] for r in rows if r["group"] == g) for g in groups}
    return {"total_scalars": count, "parameter_tensors": len(rows), "groups": by_group,
            "parameters": rows, "scope_only_not_activity_proof": True}


def capture_invariants(model, normalizers: Mapping[str, Any], *, expected_base_tensors=EXPECTED_BASE_TENSORS):
    torch = _torch()
    allowed = {n for n, _ in model.declared_trainable_named()}
    parameters = dict(model.named_parameters())
    base, frozen = {}, {}
    legacy = hashlib.sha256()
    canonical_base = {}
    for name, value in model.state_dict().items():
        if not torch.is_tensor(value):
            raise TypeError("non-tensor model state: " + name)
        if name not in allowed:
            frozen[name] = _tensor_description(value)
        marker = ".base_model."
        if marker in name:
            suffix = name.split(marker, 1)[1]
        elif name.startswith("base_model."):
            suffix = name[len("base_model."):]
        else:
            continue
        if suffix.startswith(("router.", "g0_experts.")):
            if suffix in canonical_base:
                raise AssertionError("duplicate canonical frozen base tensor: " + suffix)
            canonical_base[suffix] = value
            if name in allowed or (name in parameters and parameters[name].requires_grad):
                raise AssertionError("Router/G0 is trainable: " + name)
    for suffix in sorted(canonical_base):
        value = canonical_base[suffix]
        base[suffix] = _tensor_description(value)
        legacy.update(suffix.encode()); legacy.update(str(tuple(value.shape)).encode())
        legacy.update(value.detach().cpu().contiguous().numpy().tobytes())
    if len(base) != expected_base_tensors:
        raise AssertionError(f"Router/G0 tensor count {len(base)} != {expected_base_tensors}")
    if not normalizers:
        raise AssertionError("explicit normalizer mapping is required")
    norm = _normalizer_value(normalizers)
    return {"base_tensors": len(base), "base_sha256": _mapping_fingerprint(base),
            "legacy_base_sha256": legacy.hexdigest(), "base": base,
            "all_nontrainable_state_sha256": _mapping_fingerprint(frozen),
            "all_nontrainable_state": frozen,
            "normalizer_sha256": _mapping_fingerprint(norm), "normalizers": norm}


def assert_invariants_unchanged(before, after):
    changes = [key for key in ("base_sha256", "legacy_base_sha256",
                               "all_nontrainable_state_sha256", "normalizer_sha256")
               if before[key] != after[key]]
    if changes:
        changed_tensors = sorted(n for n in set(before["all_nontrainable_state"]) | set(after["all_nontrainable_state"])
                                 if before["all_nontrainable_state"].get(n) != after["all_nontrainable_state"].get(n))
        raise AssertionError(f"frozen/normalizer invariants changed: {changes}; tensors={changed_tensors}")
    return {"unchanged": True, "checked": ["34 Router/G0", "all nontrainable model state", "all supplied normalizers"]}


def reset_common_fast(model, seed: int, *, joint_gf_init_std: float = .001):

    torch = _torch()
    groups = model.learning_modules()
    supported = (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding)
    evidence = {}
    for group in SHARED_GROUPS:
        if group not in groups:
            raise AssertionError("missing common fast group: " + group)
        module = groups[group]
        rows = []
        for name, child in module.named_modules():
            own = list(child.named_parameters(recurse=False))
            if not own:
                continue
            if not isinstance(child, supported) or type(child).__module__.split(".")[0] != "torch":
                raise TypeError(f"unknown common-fast direct-parameter module {group}/{name}: {type(child)}")
            if any(True for _ in child.children()):
                raise TypeError(f"unsupported nested parameter-owner {group}/{name}")
            derived = seed_for(seed, group, name)
            with seeded_rng(derived), torch.no_grad():
                child.reset_parameters()
            rows.append({"module": name, "type": type(child).__name__, "seed": derived})
        if not rows:
            raise AssertionError("common group has no parameter owners: " + group)
        evidence[group] = rows
    for group in ("Uq", "UF"):
        wrapper = groups[group].network
        network = wrapper.network if hasattr(wrapper, "network") else wrapper
        last = network[-1]
        if type(last) is not torch.nn.Linear:
            raise AssertionError("native common writer must end in Linear: " + group)
        with torch.no_grad():
            torch.nn.init.zeros_(last.weight)
            if last.bias is not None:
                torch.nn.init.zeros_(last.bias)
        evidence[group].append({"design_restore": "zero final output weight/bias", "source": type(groups[group]).__name__})
    if not math.isfinite(joint_gf_init_std) or joint_gf_init_std <= 0:
        raise ValueError("positive finite joint GF output std required to avoid UF/GF zero deadlock")
    for expert_index, expert in enumerate(groups["GF"]):
        last = expert.network[-1]
        if type(last) is not torch.nn.Linear or last.bias is not None:
            raise AssertionError("native GF centered readout must end in bias-free Linear")
        derived = seed_for(seed, "GF", f"{expert_index}.joint-output")
        with seeded_rng(derived), torch.no_grad():
            torch.nn.init.normal_(last.weight, std=joint_gf_init_std)
        evidence["GF"].append({"expert": expert_index, "joint_output_init_std": joint_gf_init_std,
                               "seed": derived, "bias_free": True,
                               "reason": "joint UF/GF zero-output initialization otherwise blocks both F learning paths"})
    model.zero_grad(set_to_none=True)
    return {"policy": "named primitive reset; Uq/UF zero output; GF seeded small nonzero output; no slow reset",
            "groups": evidence}


_PACKED_BIAS_CLASS = None
_PACKED_INPUT_CLASS = None


def _packed_bias_class():
    global _PACKED_BIAS_CLASS
    if _PACKED_BIAS_CLASS is None:
        torch = _torch()

        class PackedBiasLinear(torch.nn.Module):

            def __init__(self, original, kept_indices):
                super().__init__()
                self.in_features = original.in_features
                self.out_features = original.out_features
                self.weight = original.weight
                indices = torch.as_tensor(kept_indices, dtype=torch.long,
                                          device=original.weight.device)
                self.register_buffer("bias_kept_indices", indices)
                self.bias = torch.nn.Parameter(original.bias.detach().index_select(0, indices).clone(),
                                               requires_grad=original.bias.requires_grad)

            def expanded_bias(self):
                full = self.bias.new_zeros(self.out_features)
                return full.index_copy(0, self.bias_kept_indices, self.bias)

            def forward(self, value):
                return torch.nn.functional.linear(value, self.weight, self.expanded_bias())

            def extra_repr(self):
                return (f"in_features={self.in_features}, out_features={self.out_features}, "
                        f"trainable_bias={self.bias.numel()}, fixed_zero_bias={self.out_features-self.bias.numel()}")

        _PACKED_BIAS_CLASS = PackedBiasLinear
    return _PACKED_BIAS_CLASS


def pack_linear_bias(linear, *, fixed_last: int = 17):
    torch = _torch()
    if type(linear) is not torch.nn.Linear or linear.bias is None:
        raise TypeError("expected an ordinary biased torch.nn.Linear")
    if type(fixed_last) is not int or not 0 < fixed_last < linear.out_features:
        raise ValueError("fixed_last must remove a proper bias subset")
    kept = list(range(linear.out_features - fixed_last))
    return _packed_bias_class()(linear, kept)


def _packed_input_class():
    global _PACKED_INPUT_CLASS
    if _PACKED_INPUT_CLASS is None:
        torch = _torch()

        class PackedInputLinear(torch.nn.Module):
            def __init__(self, original, kept_indices, fixed_zero_indices):
                super().__init__()
                self.in_features = original.in_features
                self.out_features = original.out_features
                kept = torch.as_tensor(kept_indices, dtype=torch.long, device=original.weight.device)
                fixed = torch.as_tensor(fixed_zero_indices, dtype=torch.long, device=original.weight.device)
                self.register_buffer("input_kept_indices", kept)
                self.register_buffer("input_fixed_zero_indices", fixed)
                self.weight = torch.nn.Parameter(original.weight.detach().index_select(1, kept).clone(),
                                                 requires_grad=original.weight.requires_grad)
                self.bias = original.bias

            def forward(self, value):
                if value.shape[-1] != self.in_features:
                    raise ValueError("original Uq input interface width changed")
                excluded = value.index_select(-1, self.input_fixed_zero_indices)
                if bool(torch.count_nonzero(excluded.detach())):
                    raise AssertionError("removed Uq S/H inputs are no longer exactly zero")
                kept = value.index_select(-1, self.input_kept_indices)
                return torch.nn.functional.linear(kept, self.weight, self.bias)

            def extra_repr(self):
                return (f"original_in_features={self.in_features}, active_in_features={self.weight.shape[1]}, "
                        f"out_features={self.out_features}, bias={self.bias is not None}")

        _PACKED_INPUT_CLASS = PackedInputLinear
    return _PACKED_INPUT_CLASS


def pack_linear_input(linear, *, fixed_zero_indices):
    torch = _torch()
    if type(linear) is not torch.nn.Linear:
        raise TypeError("expected original ordinary Linear before input packing")
    excluded = sorted(set(int(i) for i in fixed_zero_indices))
    if not excluded or excluded[0] < 0 or excluded[-1] >= linear.in_features:
        raise ValueError("invalid fixed zero input coordinates")
    kept = [i for i in range(linear.in_features) if i not in set(excluded)]
    if not kept:
        raise ValueError("all inputs removed")
    return _packed_input_class()(linear, kept, excluded)


def _writer_network(writer):
    wrapper = writer.network
    return wrapper.network if hasattr(wrapper, "network") else wrapper


def remove_common_nulls(model):
    torch = _torch()
    groups = model.learning_modules()
    names = {id(module): name for name, module in model.named_modules()}
    uq, uf = groups["Uq"], groups["UF"]
    config = uq.config
    network = _writer_network(uq)
    first = network[0]
    z, a, k, f, s = (config.observation_dim, config.action_dim,
                    config.num_experts, config.fast_dim, config.slow_dim)
    boundary = z + a + k + k * f
    width = k * s + k * f
    expected_input = z + a + k + k * f + k * s + k * f + k * z
    if type(first) is not torch.nn.Linear or first.in_features != expected_input:
        raise AssertionError("Uq input geometry/order differs from reviewed z,a,p,F,S,H,e")
    if (z, a, k, f, s, first.out_features, width) != (17, 6, 3, 32, 32, 128, 192):
        raise AssertionError("strict native Uq geometry changed")
    excluded = list(range(boundary, boundary + width))
    packed = pack_linear_input(first, fixed_zero_indices=excluded)
    network[0] = packed
    rows = [{"group": "Uq", "layer": names[id(first)], "method": "remove weights on permanent zero S/H inputs",
             "original_input_width": first.in_features, "active_input_width": packed.weight.shape[1],
             "feature_order": ["z", "a", "p", "F", "S", "H", "e"],
             "zero_input_coordinates": excluded,
             "removed_trainable_scalars": first.out_features * width,
             "removal_forward_invariant": True, "every_forward_zero_premise_asserted": True}]
    for group, writer in (("Uq", uq), ("UF", uf)):
        last = _writer_network(writer)[-1]
        expected = 3 if group == "Uq" else 32
        if type(last) is not torch.nn.Linear or last.bias is None or last.bias.numel() != expected:
            raise AssertionError("unexpected centered common-writer final bias: " + group)
        last.register_parameter("bias", None)
        rows.append({"group": group, "layer": names[id(last)], "method": "delete centered writer final bias",
                     "removed_trainable_scalars": expected, "removal_forward_invariant": True,
                     "reason": "raw_with_error - raw_without_error cancels final bias"})
    if sum(row["removed_trainable_scalars"] for row in rows) != 24611:
        raise AssertionError("common exact null removal count changed")
    return rows


def apply_active_budget(model, model_name: str = ""):

    torch = _torch()
    if getattr(model, "_strict_active_budget_receipt", None) is not None:
        raise AssertionError("active budget already applied")
    original = sum(p.numel() for _, p in model.declared_trainable_named())
    if original != ORIGINAL_TARGET:
        raise AssertionError(f"unexpected pre-restriction count {original} != {ORIGINAL_TARGET}")
    groups = model.learning_modules()
    names = {id(module): name for name, module in model.named_modules()}
    rows = []
    if all(hasattr(model, n) for n in ("measurement", "filter", "readout")):
        architecture = "neural_LF_AFTERWM"
        if len(model.readout) != 3:
            raise AssertionError("AFTER-WM must have three experts")
        for k, net in enumerate(model.readout):
            last = net[-1]
            if type(last) is not torch.nn.Linear or last.bias is None or last.bias.numel() != 17:
                raise AssertionError("expected 17 centered-readout final biases per AFTER-WM expert")
            name = names[id(last)]
            last.register_parameter("bias", None)
            rows.append({"expert": k, "layer": name, "method": "delete final centered readout bias",
                         "removed_trainable_scalars": 17, "fixed_zero_coordinates": list(range(17)),
                         "weight_unchanged": True, "removal_forward_invariant": True})
    elif "GS" in groups or "GS_basis" in groups:
        group = "GS" if "GS" in groups else "GS_basis"
        architecture = "slow_wide_readout" if group == "GS" else "slow_basis_readout"
        if len(groups[group]) != 3:
            raise AssertionError("slow readouts must have three experts")
        for k, readout in enumerate(groups[group]):
            first = readout.network[0]
            name = names[id(first)]
            count_before = sum(p.numel() for p in first.parameters())
            packed = pack_linear_bias(first, fixed_last=17)
            readout.network[0] = packed
            count_after = sum(p.numel() for p in packed.parameters())
            if count_before - count_after != 17:
                raise AssertionError("packed bias did not remove exactly 17 genuine scalars")
            rows.append({"expert": k, "layer": name, "method": "packed first-layer slow readout bias",
                         "original_bias_width": first.out_features, "trainable_bias_width": packed.bias.numel(),
                         "removed_trainable_scalars": 17,
                         "fixed_zero_coordinates": list(range(first.out_features - 17, first.out_features)),
                         "weight_unchanged": packed.weight is first.weight,
                         "removal_forward_invariant": False,
                         "reason": "common active-budget restriction; no new mechanism"})
    else:
        raise TypeError("unrecognized strict dual architecture")
    rows.extend(remove_common_nulls(model))
    actual = sum(p.numel() for _, p in model.declared_trainable_named())
    removed = sum(r["removed_trainable_scalars"] for r in rows)
    if actual != TARGET or removed != ORIGINAL_TARGET - TARGET:
        raise AssertionError(f"restricted active budget {actual} != {TARGET}")
    receipt = {"schema": SCHEMA + "_active_budget", "model": model_name, "architecture": architecture,
               "original_declared_scalars": original, "removed_scalars": removed, "target_scalars": TARGET,
               "actual_declared_scalars": actual, "restrictions": rows,
               "new_readout_mechanism_added": False,
               "activity_still_requires_actual_loss_gradient_evidence": True}
    model._strict_active_budget_receipt = receipt
    return receipt


def design_initialization_report(model):
    torch = _torch()
    report = {"ContextBasis": [], "slow_US": [], "slow_GS": [], "AFTERWM_readout": [], "common_fast": {}}
    for name, module in model.named_modules():
        if type(module).__name__ == "ContextBasis":
            offset = module.offset.detach().double()
            rows, cols = offset.shape
            gram = offset @ offset.T if rows <= cols else offset.T @ offset
            error = float((gram - torch.eye(min(rows, cols), dtype=gram.dtype, device=gram.device)).abs().max())
            last = module.network[-1]
            std = float(last.weight.detach().double().std(unbiased=False))
            if error > 2e-6 or not .00085 <= std <= .00115:
                raise AssertionError(f"ContextBasis design init lost: {name}; orthogonal error={error}, std={std}")
            if last.bias is None or torch.count_nonzero(last.bias.detach()):
                raise AssertionError("ContextBasis last bias must be zero: " + name)
            report["ContextBasis"].append({"name": name, "offset_orthogonality_max_abs_error": error,
                                           "last_weight_population_std": std, "offset": _tensor_description(module.offset)})
    groups = model.learning_modules()
    for group in ("Uq", "UF"):
        last = _writer_network(groups[group])[-1]
        if torch.count_nonzero(last.weight.detach()) or last.bias is not None:
            raise AssertionError("strict common writer output must be zero and centered bias removed: " + group)
        report["common_fast"][group] = {"zero_output_weight": True, "centered_final_bias_removed": True}
    for k, expert in enumerate(groups["GF"]):
        last = expert.network[-1]
        if last.bias is not None or not torch.count_nonzero(last.weight.detach()):
            raise AssertionError("GF must be bias-free with nonzero joint output initialization")
        std = float(last.weight.detach().double().std(unbiased=False))
        report["common_fast"][f"GF_{k}"] = {"bias_free": True, "output_weight_population_std": std,
                                                "zero_F_forward_preserved_by_centering": True}
    if "US" in groups:
        for k, module in enumerate(groups["US"]):
            last = module.network[-1]
            if last.bias is not None or last.out_features % 2:
                raise AssertionError("slow U_S paired output must be even and bias-free")
            even = int(torch.count_nonzero(last.weight[0::2]))
            odd = int(torch.count_nonzero(last.weight[1::2]))
            if not even or not odd:
                raise AssertionError("slow classification/evidence paired output initialized to zero")
            report["slow_US"].append({"expert": k, "even_nonzero": even, "odd_nonzero": odd,
                                     "constructor_pairing_preserved_by_no_slow_reset": True})
    if "GS" in groups:
        for k, module in enumerate(groups["GS"]):
            last = module.network[-1]
            if last.bias is not None or torch.count_nonzero(last.weight.detach()):
                raise AssertionError("slow G_S final output must be bias-free and zero at construction")
            report["slow_GS"].append({"expert": k, "zero_output_weight": True, "bias_free": True})
    if hasattr(model, "measurement") and hasattr(model, "filter") and hasattr(model, "readout"):
        for k, module in enumerate(model.readout):
            last = module[-1]
            if torch.count_nonzero(last.weight.detach()) or (last.bias is not None and torch.count_nonzero(last.bias.detach())):
                raise AssertionError("AFTER-WM centered readout output must start at zero")
            report["AFTERWM_readout"].append({"expert": k, "zero_output_weight": True,
                                          "has_final_bias": last.bias is not None})
    return report


def structural_null_parameters(model):
    nulls = []
    groups = model.learning_modules()
    names = {id(p): n for n, p in model.named_parameters()}
    for group in ("Uq", "UF"):
        if group not in groups:
            continue
        network = _writer_network(groups[group])
        if network[-1].bias is not None:
            p = network[-1].bias
            nulls.append({"name": names[id(p)], "numel": p.numel(), "group": group,
                          "reason": "final writer bias cancels in raw_with_error-raw_without_error"})
    if "Uq" in groups and type(_writer_network(groups["Uq"])[0]).__name__ == "Linear":
        first = _writer_network(groups["Uq"])[0]
        c = groups["Uq"].config
        start = c.observation_dim + c.action_dim + c.num_experts + c.num_experts * c.fast_dim
        width = c.num_experts * (c.slow_dim + c.fast_dim)
        nulls.append({"name": names[id(first.weight)], "numel": width * first.out_features,
                      "group": "Uq", "zero_input_columns": list(range(start, start + width)),
                      "reason": "accepted fast _q_only_view passes S and H as permanent zeros"})
    if hasattr(model, "measurement") and hasattr(model, "filter") and hasattr(model, "readout"):
        try:
            correction_source = inspect.getsource(type(model).correction)
        except (OSError, TypeError):
            correction_source = ""
        centered = "self.readout[k](x) - self.readout[k](x0)" in correction_source
        if centered:
            names = {id(p): n for n, p in model.named_parameters()}
            for k, net in enumerate(model.readout):
                if net[-1].bias is not None:
                    p = net[-1].bias
                    nulls.append({"name": names[id(p)], "numel": p.numel(), "expert": k,
                                  "reason": "final readout bias cancels identically in readout(x)-readout(x0)"})
    return nulls


def construct_independent(factory: Callable[[int], Any], seed: int,
                          normalizers: Mapping[str, Any], model_name: str,
                          *, apply_budget: bool = True, joint_gf_init_std: float = .001):
    with seeded_rng(seed_for(seed, "architecture-constructor", model_name)):
        model = factory(seed)
    if int(getattr(model, "seed", seed)) != seed:
        raise AssertionError("factory did not pass init_seed to actual Candidate/AFTER-WM")
    model.set_training(True)
    before = capture_invariants(model, normalizers)
    slow_before = {g: _mapping_fingerprint({n: _tensor_description(v) for n, v in m.state_dict().items()})
                   for g, m in model.learning_modules().items() if g not in SHARED_GROUPS}
    reset = reset_common_fast(model, seed, joint_gf_init_std=joint_gf_init_std)
    after_reset = capture_invariants(model, normalizers)
    assert_invariants_unchanged(before, after_reset)
    slow_after = {g: _mapping_fingerprint({n: _tensor_description(v) for n, v in m.state_dict().items()})
                  for g, m in model.learning_modules().items() if g not in SHARED_GROUPS}
    if slow_before != slow_after:
        raise AssertionError("common reset modified constructor-initialized slow modules")
    if apply_budget:
        budget = apply_active_budget(model, model_name)
    else:
        budget = getattr(model, "_strict_active_budget_receipt", None)
    after = capture_invariants(model, normalizers)
    for key in ("base_sha256", "legacy_base_sha256", "normalizer_sha256"):
        if before[key] != after[key]:
            raise AssertionError("budget restriction altered frozen base/normalizer: " + key)
    manifest = parameter_manifest(model)
    group_tensors = {g: {n: _tensor_description(p) for n, p in m.named_parameters()}
                    for g, m in model.learning_modules().items()}
    nulls = structural_null_parameters(model)
    report = {"schema": SCHEMA, "model": model_name, "seed": seed,
              "invariants": after, "reset": reset, "manifest": manifest,
              "active_budget": budget,
              "group_parameters": group_tensors,
              "group_sha256": {g: _mapping_fingerprint(v) for g, v in group_tensors.items()},
              "slow_constructor_before_after_equal": True,
              "design_initialization": design_initialization_report(model),
              "known_structural_nulls": nulls,
              "all_trainable_activity_verified": False}
    if nulls:
        raise AssertionError("known structurally null trainable parameters: " + repr(nulls))
    return model, report


class GradientAudit:

    def __init__(self, model):
        self.model = model
        self.manifest = parameter_manifest(model)
        self.names = list(model.declared_trainable_named())
        self.seen = {n: {"non_none_batches": 0, "nonzero_batches": 0,
                         "max_l2": 0., "max_abs": 0., "ever_nonzero": None}
                     for n, _ in self.names}
        self.samples = []
        self.structural_nulls = structural_null_parameters(model)

    def observe(self, step: int, loss):
        torch = _torch()
        if type(step) is not int or step < 1 or (self.samples and step <= self.samples[-1]["step"]):
            raise ValueError("gradient audit steps must increase")
        if not torch.is_tensor(loss) or loss.numel() != 1 or not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("actual training loss must be finite scalar")
        group_squared = {g: 0. for g in self.manifest["groups"]}
        group_of = {r["name"]: r["group"] for r in self.manifest["parameters"]}
        missing, zero = [], []
        for name, p in self.names:
            row = self.seen[name]
            grad = p.grad
            if grad is None:
                missing.append(name); continue
            if grad.shape != p.shape or not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("invalid actual gradient: " + name)
            row["non_none_batches"] += 1
            value = grad.detach().double()
            sq, mx = float(value.square().sum()), float(value.abs().max())
            mask = (grad.detach() != 0).cpu()
            row["ever_nonzero"] = mask if row["ever_nonzero"] is None else row["ever_nonzero"] | mask
            row["max_l2"] = max(row["max_l2"], math.sqrt(sq)); row["max_abs"] = max(row["max_abs"], mx)
            group_squared[group_of[name]] += sq
            if bool(mask.any()):
                row["nonzero_batches"] += 1
            else:
                zero.append(name)
        declared = {id(p) for _, p in self.names}
        foreign = [n for n, p in self.model.named_parameters() if id(p) not in declared and p.grad is not None]
        if foreign:
            raise AssertionError("frozen parameters received gradients: " + repr(foreign))
        sample = {"step": step, "loss": float(loss.detach()), "before_clipping": True,
                  "groups_l2": {g: math.sqrt(s) for g, s in group_squared.items()},
                  "missing_grad_tensors": missing, "zero_grad_tensors": zero}
        self.samples.append(sample)
        return sample

    def report(self):
        torch = _torch()
        manifest_by_name = {r["name"]: r for r in self.manifest["parameters"]}
        rows, total_nonzero = [], 0
        for name, _ in self.names:
            row = dict(self.seen[name]); mask = row.pop("ever_nonzero")
            numel = manifest_by_name[name]["numel"]
            nz = int(torch.count_nonzero(mask)) if mask is not None else 0
            total_nonzero += nz
            rows.append({"name": name, "group": manifest_by_name[name]["group"], "numel": numel,
                         **row, "ever_nonzero_scalars": nz, "never_nonzero_scalars": numel - nz})
        no_graph = [r["name"] for r in rows if r["non_none_batches"] == 0]
        no_signal = [r["name"] for r in rows if r["nonzero_batches"] == 0]
        return {"schema": SCHEMA + "_gradients", "actual_loss_gradient_samples": len(self.samples),
                "samples": self.samples, "parameters": rows,
                "declared_trainable_scalars": self.manifest["total_scalars"],
                "ever_nonzero_scalars": total_nonzero,
                "never_nonzero_scalars": self.manifest["total_scalars"] - total_nonzero,
                "no_graph_tensors": no_graph, "never_nonzero_tensors": no_signal,
                "known_structural_nulls": self.structural_nulls,
                "all_parameter_tensors_have_observed_nonzero_gradient": bool(self.samples) and not no_signal,
                "not_a_proof_every_scalar_is_identifiable": True}

    def finalize(self, *, require_each_tensor_nonzero=True):
        report = self.report()
        if not self.samples:
            raise AssertionError("no actual loss gradients audited")
        if report["known_structural_nulls"] or report["no_graph_tensors"]:
            raise AssertionError("structurally null/disconnected tensors: " + repr(report))
        if require_each_tensor_nonzero and report["never_nonzero_tensors"]:
            raise AssertionError("no observed gradient signal for tensors: " + repr(report["never_nonzero_tensors"]))
        return report


def compare_initializations(reports, *, expected_models=None, expected_seeds=None):
    if not reports:
        raise AssertionError("no initialization reports")
    keys = [(r["model"], r["seed"]) for r in reports]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate model/seed initialization report")
    models = sorted({r["model"] for r in reports})
    seeds = sorted({r["seed"] for r in reports})
    if expected_models is not None and set(models) != set(expected_models):
        raise AssertionError("model set differs from the expected set")
    if expected_seeds is not None and set(seeds) != set(expected_seeds):
        raise AssertionError("seed set differs from the expected set")
    if set(keys) != {(m, s) for m in models for s in seeds}:
        raise AssertionError("initialization grid incomplete")
    by_seed = {s: [r for r in reports if r["seed"] == s] for s in seeds}
    for seed, group in by_seed.items():
        for common in SHARED_GROUPS:
            if len({r["group_sha256"][common] for r in group}) != 1:
                raise AssertionError(f"same-seed common group not identical: {seed}/{common}")
    invariants = [r["invariants"] for r in reports]
    for key in ("base_sha256", "legacy_base_sha256", "normalizer_sha256"):
        if len({r[key] for r in invariants}) != 1:
            raise AssertionError("cross-model/seed frozen base or normalizer differs: " + key)
    independent, designed_constants = [], []
    for model in models:
        group = [r for r in reports if r["model"] == model]
        first = group[0]["group_parameters"]
        for g, tensors in first.items():
            for name, row in tensors.items():
                peer = [r["group_parameters"][g][name] for r in group]
                hashes = {v["sha256"] for v in peer}
                detail = {"model": model, "group": g, "name": name, "seeds": seeds}
                constant_design = all(v.get("min") == v.get("max") for v in peer)
                if len(seeds) > 1 and len(hashes) != len(seeds) and not constant_design:
                    raise AssertionError("nonconstant trainable tensor not independent across seeds: " + repr(detail))
                (designed_constants if constant_design else independent).append(detail)
    return {"schema": SCHEMA + "_grid", "models": models, "seeds": seeds,
            "grid_complete": True, "common_fast_same_init_per_seed": True,
            "all_stochastic_tensor_hashes_distinct_across_seeds": True,
            "frozen_base_and_supplied_normalizers_same": True,
            "stochastic_tensors": independent, "documented_designed_constant_tensors": designed_constants,
            "gradients_still_need_actual_loss_audit": True}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description='Seed and initialisation checks of the memory networks.')
    parser.add_argument("--initialization-reports", nargs="+", help="JSON reports from construct_independent")
    parser.add_argument("--output", help="save JSON-only grid audit")
    parser.add_argument("--self-check", action="store_true", help="stdlib-only stable seed/JSON smoke check")
    args = parser.parse_args()
    if args.self_check:
        assert seed_for(20260910, "Uq", "0") == seed_for(20260910, "Uq", "0")
        assert seed_for(20260910, "Uq", "0") != seed_for(20270913, "Uq", "0")
        assert seed_for(20260910, "Uq", "0") != seed_for(20260910, "UF", "0")
        print(json.dumps({"schema": SCHEMA, "stdlib_self_check": "passed", "tensor_validation": "requires training runtime torch"}))
        return
    if not args.initialization_reports:
        parser.error("provide --initialization-reports or --self-check")
    reports = [json.loads(Path(path).read_text()) for path in args.initialization_reports]
    report = compare_initializations(reports)
    if args.output:
        save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
