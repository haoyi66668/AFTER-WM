from __future__ import annotations

import copy

NAMESPACE = "Code5.JSHC-v1"
SCHEMA = "jshc_v1"
ACTUATED_JOINTS = ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")
CANDIDATE_JOINTS = (1, 2, 3, 4, 5)
T = 300
SWITCH_PERIOD = 100
PHASE_RANGE = (0, 100)
SEQ_PER_COHORT = 6
SEQUENCE_IDS = tuple(f"d{i}" for i in range(SEQ_PER_COHORT))

_SHARED_KEYS = ("env_id", "max_episode_steps", "T", "anchor_step_min", "anchor_step_max", "max_parent_attempts",
                    "action_limit", "action_kinds", "low_amplitude_windows", "low_amplitude_multiplier",
                    "contact_quota", "parents")


def base_protocol():
    from simcore.gen import BASE_PROTOCOL
    return copy.deepcopy(BASE_PROTOCOL)


def make_protocol(namespace: str = NAMESPACE) -> dict:
    m = base_protocol()
    p = {k: copy.deepcopy(m[k]) for k in _SHARED_KEYS}
    p.update({
        "schema": SCHEMA, "namespace": namespace,
        "sources": {
            "GrBAL": "Nagabandi et al., ICLR 2019, arXiv 1803.11347; github.com/iclavera/learning_to_adapt "
                     "learning_to_adapt/envs/half_cheetah_env.py (reset_task: randint(1, action_dim); step: cripple_mask * action)",
            "MOLe": "Nagabandi, Finn, Levine, ICLR 2019, arXiv 1812.07671 (malfunction changes every 500 timesteps)",
            "MBCD": "Alegre, Bazzan, da Silva, AAMAS 2021, arXiv 2105.09452, Fig. 4 (random joints disabled after every 100 time steps)"},
        "joint_switch": {"candidate_joints": list(CANDIDATE_JOINTS),
                         "candidate_names": [ACTUATED_JOINTS[j] for j in CANDIDATE_JOINTS],
                         "switch_period": SWITCH_PERIOD, "phase_range": list(PHASE_RANGE),
                         "draw_times": "0 and phi + 100 k for 0 < phi + 100 k < T; phi = first draw of the sequence's rng "
                                       "(integers(0, 100)); then one integers(1, 6) per draw time",
                         "draw": "independent uniform over candidate_joints at every draw time (repeats allowed, as GrBAL)",
                         "mechanism": "action mask: applied[j] = 0 before the simulator step (GrBAL cripple_mask); "
                                      "recorded act = commanded action",
                         "sequences_per_cohort": SEQ_PER_COHORT, "sequence_ids": list(SEQUENCE_IDS)},
        "no_impulses": True, "friction_scale": 1.0, "gain_scale": 1.0,
    })
    return p


def shared_keys_ok(protocol: dict) -> bool:
    m = base_protocol()
    return all(protocol[k] == m[k] for k in _SHARED_KEYS)
