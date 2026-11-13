import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300,
})
AFTERWM_BLUE = "#1f5fbf"
NAMES = {"rssm": "RSSM", "tdmpc2": "TD-MPC2", "tdmpc2_h40": "TD-MPC2-40", "mlp": "MLP", "grbal": "GrBAL",
         "rebal": "ReBAL", "mole": "MOLe", "mbcd": "MBCD", "gpmm": "GPMM", "mts3": "MTS3"}
PDF_META = {"Creator": None, "Producer": None, "CreationDate": None}


def results(path, name):
    with open(os.path.join(path, f"{name}.json")) as f:
        return json.load(f)


def save(fig, out, plot_data):
    base = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(base + ".pdf", metadata=PDF_META)
    fig.savefig(base + ".png", dpi=250)
    with open(base + "_data.json", "w") as f:
        json.dump(plot_data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
