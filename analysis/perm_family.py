"""Permutation p-values for the per-sample metrics, computed from the exported CSVs.

Reads results/per_sample/{gemma,medgemma}_image_to_last.csv and writes one JSON per split.
Standard library only. For each metric, the two-sided test shuffles the base/medical labels
n_perm times and counts how often the absolute difference in medians is at least the observed
one. p = (count + 1) / (n_perm + 1), the same estimator as stats.permutation_test.

Not covered here: question dominance ratio and image entry layer, which need the per-layer
information-flow data. Those p-values come from analysis/stats.py on the raw run JSON.

Usage:
    python analysis/perm_family.py
"""

import csv
import json
import os
import random
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PER_SAMPLE = os.path.join(ROOT, "results", "per_sample")
OUT_DIR = os.path.join(ROOT, "figures_compare_fixed")

N_PERM = 10000
SEED = 0
SPLITS = {"discovery": (0, 99), "confirmation": (100, 249)}
METRICS = ["if_auc", "first_token_entropy", "answer_confidence", "if_gini"]


def load(model, lo, hi):
    path = os.path.join(PER_SAMPLE, f"{model}_image_to_last.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if lo <= int(r["id"][-4:]) <= hi]


def permutation_p(a, b, n_perm=N_PERM, seed=SEED):
    rng = random.Random(seed)
    observed = abs(st.median(a) - st.median(b))
    pooled = a + b
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(st.median(pooled[:na]) - st.median(pooled[na:])) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    for split, (lo, hi) in SPLITS.items():
        gemma = load("gemma", lo, hi)
        medgemma = load("medgemma", lo, hi)
        out = {"split": split, "n_perm": N_PERM, "seed": SEED, "metrics": {}}
        for col in METRICS:
            a = [float(r[col]) for r in gemma]
            b = [float(r[col]) for r in medgemma]
            out["metrics"][col] = {"n": len(a), "p_perm": permutation_p(a, b)}
        a = [float(r["closed_margin"]) for r in gemma if r["closed_ended"] == "True"]
        b = [float(r["closed_margin"]) for r in medgemma if r["closed_ended"] == "True"]
        out["metrics"]["closed_margin"] = {"n": len(a), "p_perm": permutation_p(a, b)}
        path = os.path.join(OUT_DIR, f"perm_family_{split}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(path)


if __name__ == "__main__":
    main()
