import argparse
import os

from figures.style import results

DS = ["jshc", "escphc", "panda", "ur5e", "hopper"]
DN = {"jshc": "JSHC", "escphc": "ESCP-HC", "panda": "Panda", "ur5e": "UR5e", "hopper": "Hopper"}
GROUPS = [["rssm", "tdmpc2", "tdmpc2_h40", "mlp"], ["grbal", "rebal", "mole", "mbcd", "gpmm"], ["mts3"], ["afterwm"]]
NAME = {"rssm": "DreamerV3-style RSSM", "tdmpc2": "TD-MPC2", "tdmpc2_h40": "TD-MPC2-40", "mlp": "MLP", "grbal": "GrBAL",
        "rebal": "ReBAL", "mole": "MOLe", "mbcd": "MBCD", "gpmm": "GPMM", "mts3": "MTS3", "afterwm": r"\mbox{AFTER-WM} (ours)"}
MET = ["multi_step", "step40"]


def fm(m, s):
    d = 4 if abs(m) < 0.1 else 3
    return f"{m:.{d}f}$\\pm${s:.{d}f}"


def rows(R):
    main = R["main"]
    models = [m for g in GROUPS for m in g]
    best = {}
    for ds in DS:
        for met in MET:
            order = sorted(models, key=lambda m: main[ds]["rows"][m][met]["mean"])
            best[(ds, met)] = (order[0], order[1])
    out = []
    for gi, g in enumerate(GROUPS):
        if gi:
            out.append(r"\midrule")
        for m in g:
            cells = []
            for ds in DS:
                for met in MET:
                    r = main[ds]["rows"][m][met]
                    s = fm(r["mean"], r["sd"])
                    if m == best[(ds, met)][0]:
                        s = r"\textbf{" + s + "}"
                    elif m == best[(ds, met)][1]:
                        s = r"\underline{" + s + "}"
                    cells.append(s)
            out.append(NAME[m] + " & " + " & ".join(cells) + r" \\")
    last = []
    for ds in DS:
        b = main[ds]["strongest"]
        c = main[ds]["comparisons"][b]
        last.append(r"\multicolumn{2}{c}{" + f"$-${-c['pct']:.1f}\\% vs.\\ {NAME[b]} ({c['seeds_a_lower']}/{c['n_seeds']})" + "}")
    out.append(r"\mbox{AFTER-WM} vs.\ strongest & " + " & ".join(last) + r" \\")
    return out


def table(R):
    head = [r"\begin{tabular}{lcccccccccc}", r"\toprule",
            "Horizon: 40 steps & " + " & ".join(r"\multicolumn{2}{c}{" + DN[d] + "}" for d in DS) + r" \\",
            " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DS))),
            "Model & " + " & ".join(["multi-step mean & step 40"] * len(DS)) + r" \\", r"\midrule"]
    return head + rows(R) + [r"\bottomrule", r"\end{tabular}"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Table 1 (LaTeX tabular).")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables/table1.tex")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(table(results(a.results, "table1"))) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
