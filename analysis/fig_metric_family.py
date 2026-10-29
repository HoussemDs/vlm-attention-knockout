#!/usr/bin/env python3
"""Figure: pre-registered metric family, Holm significance and effect direction.

Reads figures_compare_fixed/stats_family.json and draws a horizontal bar chart of
-log10(p_Holm) per metric, sorted, with a dashed significance threshold at p=0.05,
bars coloured by direction (MedGemma higher vs lower than Gemma). This is the visual
summary of "7 of 11 metrics survive Holm correction".
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "figures_compare_fixed", "stats_family.json")
OUTS = [os.path.join(ROOT, "figures_compare_fixed", "fig_metric_family"),
        os.path.join(ROOT, "paper", "figures", "fig_metric_family")]

PRETTY = {
    "answer_perplexity": "Answer perplexity",
    "answer_confidence": "Mean answer log-prob",
    "first_token_entropy": "First-token entropy",
    "image_entry_layer": "Image entry layer",
    "image_entry_depth": "Image entry depth",
    "if_auc": "IF AUC (image→last)",
    "closed_margin": "Closed yes/no margin",
    "image_peak_layer": "Image peak layer",
    "qdr": "Question dominance ratio",
    "closed_accuracy": "Closed yes/no accuracy",
    "if_gini": "IF Gini (image→last)",
}


def main():
    blob = json.load(open(SRC))
    data = blob["metrics"]
    # plot only the DISTINCT quantities (drop deterministic-transform aliases)
    rows = []
    for key, d in data.items():
        if "redundant_with" in d:
            continue
        p = max(d["p_holm"], 1e-12)
        higher = d["new_median"] >= d["base_median"]  # MedGemma vs Gemma
        rows.append((PRETTY.get(key, key), -math.log10(p), higher, d["p_holm"]))
    rows.sort(key=lambda r: r[1])  # least to most significant (bottom-up)

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#c44e52" if r[2] else "#4c72b0" for r in rows]  # red=MedGemma higher, blue=lower

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = range(len(rows))
    ax.barh(list(y), vals, color=colors, edgecolor="black", linewidth=0.4)
    thr = -math.log10(0.05)
    ax.axvline(thr, color="black", linestyle="--", linewidth=1)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$-\log_{10}(p_{\mathrm{Holm}})$", fontsize=10)
    ax.set_title("Pre-specified metric family (discovery, $n{=}100$)\n"
                 "5 of 9 distinct metrics survive Holm correction", fontsize=10)

    # legend for direction
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#c44e52", edgecolor="black", label="MedGemma higher"),
                       Patch(facecolor="#4c72b0", edgecolor="black", label="MedGemma lower"),
                       Line2D([0], [0], color="black", linestyle="--", label="p = 0.05 (Holm)")],
              fontsize=8, loc="lower right", frameon=True)
    ax.margins(y=0.01)
    fig.tight_layout()
    for base in OUTS:
        os.makedirs(os.path.dirname(base), exist_ok=True)
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
        print("wrote", base + ".pdf")


if __name__ == "__main__":
    main()
