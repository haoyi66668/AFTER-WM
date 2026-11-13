import argparse
import os

from figures.style import results

NAME = {"rssm": "RSSM", "tdmpc2": "TD-MPC2", "tdmpc2_h40": "TD-MPC2-40", "mlp": "MLP", "grbal": "GrBAL",
        "rebal": "ReBAL", "mole": "MOLe", "mbcd": "MBCD", "gpmm": "GPMM", "mts3": "MTS3"}


def rows(R):
    br = R["mechanism"]["beyond_range"]
    nc = R["windows"]["no_change_summary"]
    al = R["table1"]["alfa"]
    ct = R["control"]["summary"]
    items = [("Beyond training (ESCP-HC)", br["beaten"], br["of"], f"{br['smallest_margin_pct']:.1f}\\%", br["closest"]),
             ("No change (5 datasets)", nc["beaten"], nc["of"], f"{nc['smallest_margin_pct']:.1f}\\%", nc["closest"]),
             ("Real flights (ALFA)", al["beaten"], al["of"], f"{al['smallest_margin_pct']:.1f}\\%", al["closest"]),
             ("Control (JSHC)", ct["beaten"], ct["of"], f"+{ct['smallest_gain']:.1f}", ct["closest"])]
    return [f"{a} & {b} of {c} & {d} ({NAME[e]})" + r" \\" for a, b, c, d, e in items]


def table(R):
    return [r"\begin{tabular}{lcc}", r"\toprule",
            r"Setting & \makecell{Compared\\models beaten} & \makecell{Smallest margin\\(closest model)} \\",
            r"\midrule"] + rows(R) + [r"\bottomrule", r"\end{tabular}"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Table 2 (LaTeX tabular).")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables/table2.tex")
    a = ap.parse_args(argv)
    R = {k: results(a.results, k) for k in ("mechanism", "windows", "table1", "control")}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(table(R)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
