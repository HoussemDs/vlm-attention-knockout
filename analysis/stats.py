#!/usr/bin/env python3
"""Per-sample statistics for the probe (QDR + significance), numpy-only.

Computes the quantities the paper needs but the pipeline never produced:
per-sample Question Dominance Ratio (QDR), bootstrap confidence intervals, a
permutation test and Mann-Whitney U (with effect sizes) on the base-vs-medical
QDR shift, and per-sample image-entry-layer with a CI. No scipy dependency so it
runs unchanged on Kaggle.

Usage:
    python analysis/stats.py \
        --base_dir outputs_gemma_fixed --new_dir outputs_medgemma_fixed \
        --base_label "Gemma-3-4B (base)" --new_label "MedGemma-4B (medical)" \
        --out figures_compare_fixed/stats_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-8
Q_FILE = "results_vqa_rad_question_to_last_k1.json"
I_FILE = "results_vqa_rad_image_to_last_k1.json"


# --------------------------------------------------------------------------- #
# Loading / per-sample metrics
# --------------------------------------------------------------------------- #
def _info_flow_by_id(path: str) -> Dict[str, np.ndarray]:
    """Map sample id -> per-layer info_flow curve (np array, len = num_layers)."""
    with open(path) as fh:
        data = json.load(fh)
    out: Dict[str, np.ndarray] = {}
    for r in data["results"]:
        curve = np.array([w["info_flow"] for w in r["windows"]], dtype=np.float64)
        out[r["id"]] = curve
    return out


def per_sample_qdr(model_dir: str) -> Tuple[List[str], np.ndarray]:
    """QDR_s = mean_l(max(q_if,0)) / (mean_l(max(i_if,0)) + eps) for each sample.

    Matches the aggregate QDR definition in metrics.question_dominance_ratio but
    computes it *per sample* so it has a distribution we can test and error-bar.
    """
    q = _info_flow_by_id(os.path.join(model_dir, Q_FILE))
    i = _info_flow_by_id(os.path.join(model_dir, I_FILE))
    ids = sorted(set(q) & set(i))
    vals = []
    for sid in ids:
        qpos = np.clip(q[sid], 0.0, None).mean()
        ipos = np.clip(i[sid], 0.0, None).mean()
        vals.append(qpos / (ipos + EPS))
    return ids, np.array(vals, dtype=np.float64)


def per_sample_image_entry(model_dir: str) -> np.ndarray:
    """First layer index reaching >=50% of that sample's peak image IF (clipped).

    Samples whose peak image IF <= 0 are skipped (no detectable image pathway).
    """
    i = _info_flow_by_id(os.path.join(model_dir, I_FILE))
    entries = []
    for curve in i.values():
        pos = np.clip(curve, 0.0, None)
        peak = pos.max()
        if peak <= 0:
            continue
        entries.append(int(np.argmax(pos >= 0.5 * peak)))
    return np.array(entries, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Statistics (numpy-only)
# --------------------------------------------------------------------------- #
def bootstrap_ci(x: np.ndarray, stat=np.median, n_boot: int = 10000,
                 seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([stat(x[rng.integers(0, n, n)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(stat(x)), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(n)}


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Two-sided Mann-Whitney U with tie-corrected normal approximation, plus
    rank-biserial effect size. (scipy-free.)"""
    na, nb = len(a), len(b)
    combined = np.concatenate([a, b])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # average ties
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    ra = ranks[:na].sum()
    u_a = ra - na * (na + 1) / 2.0
    u_b = na * nb - u_a
    u = min(u_a, u_b)
    mu = na * nb / 2.0
    n = na + nb
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_term = (tie_counts ** 3 - tie_counts).sum()
    sigma = np.sqrt(na * nb / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u - mu) / sigma if sigma > 0 else 0.0
    p = math.erfc(abs(z) / np.sqrt(2)) if sigma > 0 else 1.0
    rank_biserial = 1.0 - 2.0 * u_a / (na * nb)  # effect size in [-1,1]
    return {"U": float(u), "z": float(z), "p_value": float(min(p, 1.0)),
            "rank_biserial": float(rank_biserial)}


def permutation_test(a: np.ndarray, b: np.ndarray, stat="median",
                     n_perm: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Two-sided permutation test on the difference of (median|mean)."""
    rng = np.random.default_rng(seed)
    fn = np.median if stat == "median" else np.mean
    obs = fn(a) - fn(b)
    pooled = np.concatenate([a, b]); na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(fn(pooled[:na]) - fn(pooled[na:])) >= abs(obs):
            count += 1
    return {"observed_diff": float(obs), "p_value": float((count + 1) / (n_perm + 1)),
            "stat": stat}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


# --------------------------------------------------------------------------- #
def analyse(base_dir: str, new_dir: str, base_label: str, new_label: str,
            seed: int = 0) -> dict:
    _, qdr_base = per_sample_qdr(base_dir)
    _, qdr_new = per_sample_qdr(new_dir)
    ent_base = per_sample_image_entry(base_dir)
    ent_new = per_sample_image_entry(new_dir)

    return {
        "qdr": {
            base_label: {
                "median_ci": bootstrap_ci(qdr_base, np.median, seed=seed),
                "mean_ci": bootstrap_ci(qdr_base, np.mean, seed=seed),
            },
            new_label: {
                "median_ci": bootstrap_ci(qdr_new, np.median, seed=seed),
                "mean_ci": bootstrap_ci(qdr_new, np.mean, seed=seed),
            },
            "comparison": {
                "mann_whitney": mann_whitney_u(qdr_base, qdr_new),
                "permutation_median": permutation_test(qdr_base, qdr_new, "median", seed=seed),
                "permutation_mean": permutation_test(qdr_base, qdr_new, "mean", seed=seed),
                "cliffs_delta": cliffs_delta(qdr_base, qdr_new),
                "median_drop_pct": float(100 * (1 - np.median(qdr_new) / np.median(qdr_base))),
            },
        },
        "image_entry_layer": {
            base_label: bootstrap_ci(ent_base, np.median, seed=seed),
            new_label: bootstrap_ci(ent_new, np.median, seed=seed),
            "comparison": {
                "mann_whitney": mann_whitney_u(ent_base, ent_new),
            },
        },
        "meta": {"base_dir": base_dir, "new_dir": new_dir, "seed": seed,
                 "qdr_definition": "mean_l(max(q_if,0)) / (mean_l(max(i_if,0)) + 1e-8), per sample"},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="outputs_gemma_fixed")
    ap.add_argument("--new_dir", default="outputs_medgemma_fixed")
    ap.add_argument("--base_label", default="Gemma-3-4B (base)")
    ap.add_argument("--new_label", default="MedGemma-4B (medical)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures_compare_fixed/stats_summary.json")
    args = ap.parse_args()

    res = analyse(args.base_dir, args.new_dir, args.base_label, args.new_label, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    q = res["qdr"]
    print("=== Per-sample QDR ===")
    for lab in (args.base_label, args.new_label):
        m = q[lab]["median_ci"]
        print(f"  {lab}: median {m['point']:.3f}  95% CI [{m['ci_low']:.3f}, {m['ci_high']:.3f}]  (n={m['n']})")
    c = q["comparison"]
    print(f"  median drop: {c['median_drop_pct']:.1f}%")
    print(f"  Mann-Whitney p={c['mann_whitney']['p_value']:.4g}  rank-biserial={c['mann_whitney']['rank_biserial']:.3f}")
    print(f"  permutation(median) p={c['permutation_median']['p_value']:.4g}")
    print(f"  Cliff's delta={c['cliffs_delta']:.3f}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
