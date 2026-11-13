from __future__ import annotations

import argparse
import json
import os

NAME = {"rssm": "RSSM", "tdmpc2": "TD-MPC2", "tdmpc2_h40": "TD-MPC2-40", "mlp": "MLP", "grbal": "GrBAL",
        "rebal": "ReBAL", "mole": "MOLe", "mbcd": "MBCD", "gpmm": "GPMM", "mts3": "MTS3"}


def f1(x):
    return f"{x:.1f}"


def rng(a, b, fmt=f1):
    return f"{fmt(a)}--{fmt(b)}"


def load(results):
    out = {}
    for k in ("table1", "per_step", "phases", "windows", "mechanism", "sensitivity", "probes", "control",
              "examples", "per_joint"):
        with open(os.path.join(results, f"{k}.json")) as f:
            out[k] = json.load(f)
    return out


def numbers(R):
    T, P, PH, W, M = R["table1"], R["per_step"]["summary"], R["phases"]["summary"], R["windows"], R["mechanism"]
    ms = T["main_summary"]
    lo, hi = ms["vs_strongest_range_pct"]
    n = {}
    n["main.strongest_models"] = sorted({NAME[b] for b in ms["strongest"].values()})
    n["main.all_seeds_lower"] = f"{ms['all_seeds_lower']} of {ms['n_comparisons']}"
    n["main.ci_below_zero"] = f"{ms['ci_below_zero']} of {ms['n_comparisons']}"
    n["main.vs_strongest_range"] = rng(lo, hi)
    n["main.vs_strongest_range_rounded"] = f"{lo:.0f}--{hi:.0f}"
    n["main.afterwm_best_both_metrics"] = ms["afterwm_lowest_multi_step_and_step40_everywhere"]
    n["per_step.lowest_from_step"] = P["afterwm_lowest_from_step"]
    n["per_step.wins"] = f"{P['afterwm_lower']:,} of the {P['comparisons']:,}"
    n["per_step.ci_below_zero"] = f"{P['ci_below_zero']:,}"
    n["per_step.vs_strongest_from_step"] = P["vs_strongest_ci_below_zero_from_step"]
    n["per_step.step40_range"] = rng(*P["vs_strongest_step40_range_pct"])
    n["per_step.clipped"] = {ds: {NAME[m]: k for m, k in v.items()} for ds, v in P["clipped"].items()}
    n["phases.cells"] = PH["cells"]
    n["phases.afterwm_lower"] = PH["afterwm_lower"]
    n["phases.ci_below_zero"] = PH["ci_below_zero"]
    n["phases.phase1_min"] = f1(PH["phase1_min_reduction_pct"])
    n["phases.escphc_all_positive"] = PH["escphc_all_phases_positive"]
    n["windows.no_change_min"] = f1(W["no_change_summary"]["smallest_margin_pct"])
    n["windows.transient_min"] = f1(W["transient_summary"]["smallest_margin_pct"])
    ex = R["examples"]["facts"]
    n["examples.candidates"] = sorted({v["n_candidates"] for v in ex.values()})
    n["examples.median_rank"] = {ds: f"{v['rank']} of {v['n_candidates']}" for ds, v in ex.items()}
    n["examples.steps_since_change"] = sorted({v["steps_since_change"] for v in ex.values()})
    n["examples.panda_payload_after"] = f1(ex["panda"]["change"]["payload"]["after"])
    n["examples.step40"] = {ds: [v["step40_label"]["afterwm"], v["step40_label"]["compared"]] for ds, v in ex.items()}
    n["examples.less_than_half"] = all(v["afterwm_less_than_half"] for v in ex.values())
    n["per_joint"] = {ds: f"{v['afterwm_lower']} of {v['joints']}" for ds, v in R["per_joint"].items()}
    iv = M["interventions"]["range_pct"]
    for k in iv:
        n[f"interventions.{k}"] = rng(*iv[k])
    n["interventions.other_sequence_largest"] = M["interventions"]["other_sequence_largest_everywhere"]
    pr = R["probes"]["summary"]
    n["probes.fast_r2"] = rng(*pr["fast_r2_range"], fmt=lambda x: f"{x:.2f}")
    n["probes.slow_r2"] = rng(*pr["slow_r2_range"], fmt=lambda x: f"{x:.2f}")
    n["probes.shuffled_below_zero"] = pr["shuffled_r2_max"] < 0
    ba = pr["jshc_balanced_accuracy"]
    n["probes.jshc_ba"] = [f"{ba['F']:.2f}", f"{ba['c']:.2f}", f"{ba['c_shuf']:.2f}"]
    rs = M["removal"]["summary"]
    n["removal.no_fast_step1"] = rng(*rs["no_fast/step1"])
    n["removal.no_fast_step40_max"] = f1(rs["no_fast/step40_max_abs"])
    n["removal.no_slow_step1_max"] = f1(rs["no_slow/step1_max_abs"])
    n["removal.no_slow_step40"] = rng(*rs["no_slow/step40"])
    n["removal.other_step_max"] = f1(max(rs["no_fast/step40_max_abs"], rs["no_slow/step1_max_abs"]))
    bp = rs["by_phase"]
    n["removal.no_fast_early"] = [f1(bp["panda"]["no_fast/early"]), f1(bp["ur5e"]["no_fast/early"])]
    n["removal.no_fast_late"] = [f1(bp["panda"]["no_fast/late"]), f1(bp["ur5e"]["no_fast/late"])]
    n["removal.no_slow_early_max"] = f1(max(bp["panda"]["no_slow/early"], bp["ur5e"]["no_slow/early"]))
    n["removal.no_slow_late"] = [f1(bp["panda"]["no_slow/late"]), f1(bp["ur5e"]["no_slow/late"])]
    n["removal.seeds"] = {ds: len(c["seeds"]) for ds, c in M["removal"]["cells"].items()}
    n["crossing.step1"] = rng(*P["crossing_step1_range_pct"])
    cx = {ds: -v["step40"] for ds, v in P["crossing"].items()}
    n["crossing.step40_without_ur5e"] = rng(min(v for ds, v in cx.items() if ds != "ur5e"),
                                            max(v for ds, v in cx.items() if ds != "ur5e"))
    n["crossing.step40_ur5e"] = f1(cx["ur5e"])
    sc = T["single_correction"]
    n["single.fast_step1_within"] = f1(sc["fast_only_step1_max_abs_pct"])
    n["single.fast_step40"] = rng(*sc["fast_only_step40_range_pct"])
    n["single.slow_step40"] = rng(*sc["slow_only_step40_range_pct"])
    rc = M["routing"]["cells"]
    n["routing.without_escphc"] = rng(min(c["value"] for ds, c in rc.items() if ds != "escphc"),
                                      max(c["value"] for ds, c in rc.items() if ds != "escphc"))
    n["routing.escphc"] = f1(rc["escphc"]["value"])
    n["routing.seeds"] = {ds: len(c["seeds"]) for ds, c in M["routing"]["cells"].items()}
    n["ablations.range"] = rng(*T["ablations"]["range_pct"])
    n["ablations.all_positive"] = T["ablations"]["all_positive"]
    br = M["beyond_range"]
    n["beyond.min"] = f1(br["smallest_margin_pct"])
    n["beyond.beaten"] = f"{br['beaten']} of {br['of']}"
    n["beyond.closest"] = NAME[br["closest"]]
    al = T["alfa"]
    n["alfa.seeds"] = len(al["afterwm_seeds"])
    n["alfa.min"] = f1(al["smallest_margin_pct"])
    n["alfa.closest"] = NAME[al["closest"]]
    n["alfa.all_seeds"] = al["all_beaten_in_all_seeds"]
    ct = R["control"]
    n["control.afterwm"] = f1(ct["summary"]["afterwm_mean_return"])
    n["control.gap"] = f1(ct["summary"]["gap_to_second"])
    n["control.second"] = NAME[ct["summary"]["second"]]
    n["control.highest"] = ct["summary"]["afterwm_highest"]
    n["control.spearman"] = f"{ct['prediction_link']['spearman']:.2f}"
    n["control.n_models"] = len(ct["rows"])
    n["control.episodes"] = ct["n_common_episodes"]
    n["control.seeds"] = ct["n_seeds"]
    se = R["sensitivity"]
    n["sensitivity.n"] = se["n_other"]
    n["sensitivity.all_seeds"] = se["all_lower_all_seeds"]
    n["sensitivity.range"] = rng(*se["reduction_range_pct"])
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Numbers of the paper text that follow from the stored outputs, formatted as printed.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.out, "w") as f:
        json.dump(numbers(load(a.results)), f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
