#!/usr/bin/env python3
"""Figure: confirmation-split replication of the pre-registered primary test and
discovery candidates.

Reads figures_compare_fixed/stats_family_confirm.json (indices 0100-0249, n=150) and
draws a horizontal bar chart of -log10(p_raw) for the pre-registered primary metric
(QDR) and the five distinct discovery candidates, with a dashed p=0.05 replication
threshold. Bars are coloured by verdict: confirmed (survives Holm across the five
candidates), raw-only (p_raw<0.05 but fails candidate-set Holm), and not replicated.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "figures_compare_fixed", "stats_family_confirm.json")
OUTS = [os.path.join(ROOT, "figures_compare_fixed", "fig_confirmation"),
        os.path.join(ROOT, "paper", "figures", "fig_confirmation")]

# metric key -> (pretty label, verdict). Verdict from the pre-registered rule:
#   raw p<0.05 same direction = confirmed; Holm across the 5 distinct candidates
#   decides "confirmed" vs "raw only"; QDR is the primary test (no correction).
PLAN = [
    ("qdr",                 "Question dominance ratio (primary)", "confirmed"),
    ("answer_confidence",   "Mean answer log-prob",               "confirmed"),
    ("first_token_entropy", "First-token entropy",                "confirmed"),
    ("closed_margin",       "Closed yes/no logit margin",         "confirmed"),
    ("if_auc",              "IF AUC (image->last)",               "raw"),
    ("image_entry_layer",   "Image entry layer",                  "null"),
]
COLOR = {"confirmed": "#55a868", "raw": "#dd8452", "null": "#8c8c8c"}


def main():
    data = json.load(open(SRC))["metrics"]
    rows = []
    for key, label, verdict in PLAN:
        p = max(data[key]["p_raw"], 1e-16)
        rows.append((label, -math.log10(p), verdict))
    rows.sort(key=lambda r: r[1])  # least to most significant, bottom-up

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [COLOR[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y = range(len(rows))
    ax.barh(list(y), vals, color=colors, edgecolor="black", linewidth=0.4)
    thr = -math.log10(0.05)
    ax.axvline(thr, color="black", linestyle="--", linewidth=1)
    ax.text(thr, -0.45, " p=0.05", fontsize=8, va="center")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$-\log_{10}(p_{\mathrm{raw}})$", fontsize=10)
    ax.set_title("Confirmation split (indices 0100-0249, $n{=}150$)\n"
                 "primary test and 4 of 5 candidates replicate", fontsize=10)

    ax.legend(handles=[
        Patch(facecolor=COLOR["confirmed"], edgecolor="black", label="confirmed"),
        Patch(facecolor=COLOR["raw"], edgecolor="black", label="raw only"),
        Patch(facecolor=COLOR["null"], edgecolor="black", label="not replicated"),
    ], fontsize=8, loc="lower right", frameon=True)
    ax.margins(y=0.08)
    fig.tight_layout()
    for base in OUTS:
        os.makedirs(os.path.dirname(base), exist_ok=True)
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
        print("wrote", base + ".pdf")


if __name__ == "__main__":
    main()
