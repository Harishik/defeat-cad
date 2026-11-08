"""Render README benchmark charts from bench/results/*.json (light and dark variants).

    python docs/make_charts.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "bench", "results")
OUT = os.path.join(ROOT, "docs", "images")

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#212529", "muted": "#6c757d", "grid": "#e9ecef",
              "c": ["#c5cad1", "#3a3fd4", "#fb5607"]},
    "dark": {"bg": "#0d1117", "fg": "#e9ecef", "muted": "#9aa1a9", "grid": "#262b33",
             "c": ["#3a4048", "#6c72ff", "#fb5607"]},
}


def load(name):
    return json.load(open(os.path.join(RES, name), encoding="utf-8"))


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=11)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def grouped(ax, t, groups, series, values, ylim=(0, 1.05), fmt="{:.2f}", cis=None):
    n = len(series)
    w = 0.8 / n
    for i, s in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        bars = ax.bar(xs, values[i], w * 0.92, color=t["c"][i % len(t["c"])], label=s, zorder=3)
        tops = values[i]
        if cis and all(cis[i]):
            lo = [v - c[0] for v, c in zip(values[i], cis[i])]
            hi = [c[1] - v for v, c in zip(values[i], cis[i])]
            ax.errorbar(xs, values[i], yerr=[lo, hi], fmt="none", ecolor=t["fg"], elinewidth=1, capsize=3, zorder=4)
            tops = [c[1] for c in cis[i]]
        for b, v, y in zip(bars, values[i], tops):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.015, fmt.format(v), ha="center", va="bottom",
                    fontsize=10, color=t["fg"], fontweight="bold" if i == n - 1 else "normal")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, color=t["fg"], fontsize=12)
    ax.set_ylim(*ylim)
    leg = ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=n, bbox_to_anchor=(0, 1.13))
    for txt in leg.get_texts():
        txt.set_color(t["fg"])


def fig(t, w=9, h=4.2):
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    f, ax = plt.subplots(figsize=(w, h), dpi=200)
    f.patch.set_facecolor(t["bg"])
    style(ax, t)
    return f, ax


def title(f, t, a, b):
    f.text(0.012, 0.975, a, fontsize=15, fontweight="bold", color=t["fg"], va="top")
    f.text(0.012, 0.905, b, fontsize=10.5, color=t["muted"], va="top")


def main():
    os.makedirs(OUT, exist_ok=True)
    fu, mf = load("fusion_test.json"), load("mfcad_test.json")
    rob = load("robustness.json")
    for name, t in THEMES.items():
        # Fusion 360: fillets / chamfers, F1
        f, ax = fig(t)
        series = ["Rules only", "Network only", "DeFeat (rules + network)"]
        keys = ["rule", "net", "fused"]
        vals = [[fu[k]["fillet"]["f1"], fu[k]["chamfer"]["f1"]] for k in keys]
        cis = [[fu[k]["fillet"].get("f1_ci95"), fu[k]["chamfer"].get("f1_ci95")] for k in keys]
        grouped(ax, t, ["Fillets", "Chamfers"], series, vals, cis=cis)
        title(f, t, "Blend recognition on real designer parts (F1, 95% bootstrap interval)",
              f"Fusion 360 Gallery, official test split — {fu['parts_ok']:,} parts, {fu['faces']:,} faces")
        f.subplots_adjust(top=0.72, bottom=0.1, left=0.05, right=0.98)
        f.savefig(os.path.join(OUT, f"chart-fusion-{name}.png"), facecolor=t["bg"])
        plt.close(f)

        # MFCAD
        f, ax = fig(t)
        series = ["Rules only", "Network only", "DeFeat (rules + network)"]

        def row(p):
            return [p["feature_vs_stock"]["f1"], p["per_class"]["per_class"]["cut"]["f1"], p["per_class"]["per_class"]["step"]["f1"]]
        vals = [row(mf["rule"]), row(mf["net"]), row(mf["fused"])]
        grouped(ax, t, ["Feature vs. body", "Cut-outs", "Steps"], series, vals)
        title(f, t, "Machining-feature recognition (F1)",
              f"MFCAD held-out split — {mf['parts_ok']:,} parts, {mf['rule']['per_class']['faces']:,} faces")
        f.subplots_adjust(top=0.72, bottom=0.1, left=0.05, right=0.98)
        f.savefig(os.path.join(OUT, f"chart-mfcad-{name}.png"), facecolor=t["bg"])
        plt.close(f)

        # Robustness
        f, ax = fig(t, h=3.6)
        labels = ["Parts completed", "Valid solid out", "Targeted features removed"]
        v = [rob["completion_rate"], rob["output_valid_rate"], rob["feature_removal_success_rate"]]
        bars = ax.barh(labels[::-1], v[::-1], color=t["c"][2], height=0.55, zorder=3)
        for b, x in zip(bars, v[::-1]):
            ax.text(x - 0.01, b.get_y() + b.get_height() / 2, f"{100 * x:.1f}%", va="center", ha="right",
                    color="white", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.xaxis.grid(True, color=t["grid"])
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", labelcolor=t["fg"], labelsize=12)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        title(f, t, "Full pipeline on random real-world parts",
              f"{rob['parts']} Fusion 360 test models · median {rob['seconds_per_part']['median']:.2f} s per part on a laptop CPU")
        f.subplots_adjust(top=0.72, bottom=0.12, left=0.27, right=0.98)
        f.savefig(os.path.join(OUT, f"chart-robust-{name}.png"), facecolor=t["bg"])
        plt.close(f)
    print("charts written to", OUT)


if __name__ == "__main__":
    main()
