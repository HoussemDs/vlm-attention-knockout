#!/usr/bin/env python3
"""Full pre-registered metric family with per-sample tests + Holm correction.

Extends analysis/stats.py (QDR + entry layer only) to the whole family declared
in PREREGISTRATION.md section 3. Every metric that both models produce *per
sample* is compared base-vs-medical with a two-sided Mann-Whitney U; closed
accuracy is compared with Fisher's exact test. Holm-Bonferroni is applied across
the family (raw and adjusted p reported).

Encoder-representation metrics (linear CKA, encoder divergence, JS/KL, the
encoder-control regression intercept) are NOT two-sample base-vs-medical tests,
so they are reported in a separate block, not folded into the Holm family.

numpy-only (scipy-free) so it matches stats.py and runs on Kaggle.

Usage:
    python analysis/stats_family.py \
        --base_dir outputs_gemma_fixed --new_dir outputs_medgemma_fixed \
        --out figures_compare_fixed/stats_family.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import (  # noqa: E402
    _info_flow_by_id, per_sample_qdr, per_sample_image_entry,
    mann_whitney_u, cliffs_delta, I_FILE,
)

EPS = 1e-8


# --------------------------------------------------------------------------- #
# per-sample extractors (id-aligned across the two models)
# --------------------------------------------------------------------------- #
def _field_by_id(path, field):
    with open(path) as fh:
        data = json.load(fh)
    out = {}
    for r in data["results"]:
        v = r.get(field)
        if v is not None:
            out[r["id"]] = float(v)
    return out


def per_sample_peak_layer(model_dir):
    """Layer index of max clipped image info-flow (the image 'peak layer')."""
    i = _info_flow_by_id(os.path.join(model_dir, I_FILE))
    out = {}
    for sid, curve in i.items():
        pos = np.clip(curve, 0.0, None)
        if pos.max() <= 0:
            continue
        out[sid] = float(np.argmax(pos))
    return out


def per_sample_entry_depth(model_dir):
    """Image entry layer normalised by num_layers (depth in [0,1])."""
    i = _info_flow_by_id(os.path.join(model_dir, I_FILE))
    out = {}
    for sid, curve in i.items():
        pos = np.clip(curve, 0.0, None)
        peak = pos.max()
        if peak <= 0:
            continue
        entry = int(np.argmax(pos >= 0.5 * peak))
        out[sid] = entry / len(curve)
    return out


def _entry_by_id(model_dir):
    i = _info_flow_by_id(os.path.join(model_dir, I_FILE))
    out = {}
    for sid, curve in i.items():
        pos = np.clip(curve, 0.0, None)
        if pos.max() <= 0:
            continue
        out[sid] = float(np.argmax(pos >= 0.5 * pos.max()))
    return out


def _qdr_by_id(model_dir):
    ids, vals = per_sample_qdr(model_dir)
    return dict(zip(ids, vals))


def aligned(base_map, new_map):
    """Return (base_arr, new_arr) over the shared ids, id-sorted."""
    ids = sorted(set(base_map) & set(new_map))
    a = np.array([base_map[i] for i in ids], dtype=np.float64)
    b = np.array([new_map[i] for i in ids], dtype=np.float64)
    return a, b


# --------------------------------------------------------------------------- #
def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]] (small counts, exact)."""
    def logfact(n):
        return math.lgamma(n + 1)
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d

    def logp(x):
        y = c1 - x
        z = r1 - x
        w = r2 - c1 + x
        if y < 0 or z < 0 or w < 0:
            return None
        return (logfact(r1) + logfact(r2) + logfact(c1) + logfact(c2)
                - logfact(n) - logfact(x) - logfact(y) - logfact(z) - logfact(w))
    p_obs = logp(a)
    total = 0.0
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp is not None and lp <= p_obs + 1e-9:
            total += math.exp(lp)
    return float(min(total, 1.0))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone-enforced)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


# --------------------------------------------------------------------------- #
# metric family
# --------------------------------------------------------------------------- #
# (key, human label, direction note, extractor)  -- all per-model per-sample
FIELD_METRICS = [
    ("if_auc",            "IF AUC (image->last)",        "if_auc"),
    ("if_gini",           "IF Gini (image->last)",       "if_gini"),
    ("first_token_entropy", "First-token entropy",       "first_token_entropy"),
    ("answer_confidence", "Mean answer log-prob",        "answer_confidence"),
    ("answer_perplexity", "Answer perplexity",           "answer_perplexity"),
    ("closed_margin",     "Closed yes/no logit margin",  "closed_margin"),
]


def analyse(base_dir, new_dir, base_label, new_label):
    metrics = {}

    # derived-curve metrics
    derived = {
        "qdr":               (_qdr_by_id(base_dir),          _qdr_by_id(new_dir),          "QDR (image->last)"),
        "image_entry_layer": (_entry_by_id(base_dir),        _entry_by_id(new_dir),        "Image entry layer"),
        "image_entry_depth": (per_sample_entry_depth(base_dir), per_sample_entry_depth(new_dir), "Image entry depth"),
        "image_peak_layer":  (per_sample_peak_layer(base_dir), per_sample_peak_layer(new_dir), "Image peak layer"),
    }
    for key, (bm, nm, label) in derived.items():
        a, b = aligned(bm, nm)
        mw = mann_whitney_u(a, b)
        metrics[key] = {
            "label": label, "n": int(len(a)),
            "base_median": float(np.median(a)), "new_median": float(np.median(b)),
            "p_raw": mw["p_value"], "rank_biserial": mw["rank_biserial"],
            "cliffs_delta": cliffs_delta(a, b), "test": "mann_whitney",
        }

    # direct per-sample field metrics
    for key, label, field in FIELD_METRICS:
        a, b = aligned(_field_by_id(os.path.join(base_dir, I_FILE), field),
                       _field_by_id(os.path.join(new_dir, I_FILE), field))
        if len(a) == 0:
            continue
        mw = mann_whitney_u(a, b)
        metrics[key] = {
            "label": label, "n": int(len(a)),
            "base_median": float(np.median(a)), "new_median": float(np.median(b)),
            "p_raw": mw["p_value"], "rank_biserial": mw["rank_biserial"],
            "cliffs_delta": cliffs_delta(a, b), "test": "mann_whitney",
        }

    # closed accuracy -> Fisher exact on the 2x2
    def closed_counts(model_dir):
        with open(os.path.join(model_dir, I_FILE)) as fh:
            data = json.load(fh)
        cor = inc = 0
        for r in data["results"]:
            if r.get("closed_ended") and r.get("closed_pred") is not None:
                if bool(r.get("correct")):
                    cor += 1
                else:
                    inc += 1
        return cor, inc
    bc, bi = closed_counts(base_dir)
    nc, ni = closed_counts(new_dir)
    metrics["closed_accuracy"] = {
        "label": "Closed yes/no accuracy", "n": int(bc + bi),
        "base_median": bc / (bc + bi), "new_median": nc / (nc + ni),
        "base_counts": [bc, bi], "new_counts": [nc, ni],
        "p_raw": fisher_exact_2x2(bc, bi, nc, ni),
        "rank_biserial": None, "cliffs_delta": None, "test": "fisher_exact",
    }

    # Two pairs are deterministic transforms of each other and carry identical test
    # statistics, so they must not be double-counted in the correction:
    #   answer_perplexity = exp(-answer_confidence)        (monotonic)
    #   image_entry_depth = image_entry_layer / num_layers (linear, num_layers const)
    # Holm is applied across the DISTINCT quantities only; each alias inherits its
    # canonical partner's adjusted p.
    REDUNDANT = {"answer_perplexity": "answer_confidence",
                 "image_entry_depth": "image_entry_layer"}
    distinct = [k for k in metrics if k not in REDUNDANT]
    adj = holm([metrics[k]["p_raw"] for k in distinct])
    for k, p in zip(distinct, adj):
        metrics[k]["p_holm"] = float(p)
    for alias, canon in REDUNDANT.items():
        metrics[alias]["p_holm"] = metrics[canon]["p_holm"]
        metrics[alias]["redundant_with"] = canon

    n_survive = sum(1 for k in distinct if metrics[k]["p_holm"] < 0.05)
    return {"metrics": metrics,
            "family_size_declared": len(metrics),
            "family_size_distinct": len(distinct),
            "n_distinct_surviving": n_survive,
            "redundant_pairs": REDUNDANT,
            "meta": {"base_dir": base_dir, "new_dir": new_dir,
                     "base_label": base_label, "new_label": new_label}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="outputs_gemma_fixed")
    ap.add_argument("--new_dir", default="outputs_medgemma_fixed")
    ap.add_argument("--base_label", default="Gemma-3-4B (base)")
    ap.add_argument("--new_label", default="MedGemma-4B (medical)")
    ap.add_argument("--out", default="figures_compare_fixed/stats_family.json")
    args = ap.parse_args()

    res = analyse(args.base_dir, args.new_dir, args.base_label, args.new_label)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    m = res["metrics"]
    order = sorted(m, key=lambda k: m[k]["p_raw"])
    print(f"=== Pre-registered family: {res['family_size_declared']} metrics, "
          f"{res['family_size_distinct']} distinct (Holm over distinct); "
          f"{res['n_distinct_surviving']} independent survive ===")
    hdr = f"{'metric':28s} {'base':>9s} {'medgemma':>9s} {'p_raw':>9s} {'p_holm':>9s}  sig"
    print(hdr); print("-" * len(hdr))
    for k in order:
        d = m[k]
        star = "***" if d["p_holm"] < 0.001 else "**" if d["p_holm"] < 0.01 else "*" if d["p_holm"] < 0.05 else ""
        print(f"{d['label']:28s} {d['base_median']:9.4f} {d['new_median']:9.4f} "
              f"{d['p_raw']:9.4g} {d['p_holm']:9.4g}  {star}")
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
