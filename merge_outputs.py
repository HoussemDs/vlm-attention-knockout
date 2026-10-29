#!/usr/bin/env python3
"""Merge the probe output dirs produced by a GPU-split run into one dir.

When a model's run is split across GPUs (disjoint --start_index ranges), each
process writes its own output dir. Sample ids are globally unique
(vqarad_<split>_<absidx>), so merging is just a dedup-by-id concatenation plus a
recompute of the aggregate `curve` and `run_summary`.

Usage:
    python merge_outputs.py --inputs outputs_gemma_g0 outputs_gemma_g1 \
                            --output outputs_gemma
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean, pstdev
from typing import List

FILES = [
    "results_vqa_rad_image_to_last_k1.json",
    "results_vqa_rad_question_to_last_k1.json",
]


def _recompute_curve(results: List[dict], num_layers: int) -> dict:
    layer_if = [[] for _ in range(num_layers)]
    layer_lp = [[] for _ in range(num_layers)]
    for r in results:
        for w in r["windows"]:
            layer_if[w["start"]].append(w["info_flow"])
            layer_lp[w["start"]].append(w.get("info_flow_logprob", 0.0))
    return {
        "layer": list(range(num_layers)),
        "info_flow": [mean(v) if v else 0.0 for v in layer_if],
        "info_flow_logprob": [mean(v) if v else 0.0 for v in layer_lp],
        "std": [pstdev(v) if len(v) > 1 else 0.0 for v in layer_if],
    }


def _recompute_summary(results: List[dict]) -> dict:
    def _acc(rs):
        rs = [r for r in rs if r.get("correct") is not None]
        return (sum(1 for r in rs if r["correct"]) / len(rs)) if rs else None

    closed = [r for r in results if r.get("closed_ended")]
    openq = [r for r in results if r.get("closed_ended") is False]
    confs = [r["answer_confidence"] for r in results if r.get("answer_confidence") is not None]
    cc = [r["answer_confidence"] for r in results if r.get("correct") and r.get("answer_confidence") is not None]
    ci = [r["answer_confidence"] for r in results if r.get("correct") is False and r.get("answer_confidence") is not None]
    acc = _acc(closed)  # headline = closed yes/no only; open-ended not evaluated
    return {
        "accuracy": acc,
        "exact_match": acc,
        "accuracy_closed": acc,
        "accuracy_open": None,
        "n_closed": len(closed),
        "n_open": len(openq),
        "mean_answer_confidence": mean(confs) if confs else None,
        "mean_conf_correct": mean(cc) if cc else None,
        "mean_conf_incorrect": mean(ci) if ci else None,
    }


def merge_file(input_dirs: List[str], output_dir: str, fname: str) -> None:
    merged = {}
    meta = None
    for d in input_dirs:
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        data = json.load(open(path))
        meta = meta or dict(data["meta"])
        for r in data["results"]:
            merged[r["id"]] = r  # dedup by id; later wins
    if not merged:
        print(f"  nothing to merge for {fname}")
        return
    results = [merged[i] for i in sorted(merged)]
    num_layers = results[0]["num_layers"]
    meta["num_samples"] = len(results)
    out = {
        "meta": meta,
        "run_summary": _recompute_summary(results),
        "curve": _recompute_curve(results, num_layers),
        "results": results,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, fname), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  {fname}: merged {len(results)} samples from {len(input_dirs)} dirs -> {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    for fname in FILES:
        merge_file(args.inputs, args.output, fname)


if __name__ == "__main__":
    main()
