#!/usr/bin/env python3
"""Export the numeric results behind the paper: per-layer information-flow curves
and per-sample numeric metrics. Text fields (questions, reference answers, generated
answers) are dropped, so the export contains no VQA-RAD content.

Usage:
    python reproduce/export_results.py --runs outputs_gemma_full outputs_medgemma_full \
        --out results
"""
import argparse
import csv
import json
from pathlib import Path

TEXT_FIELDS = {
    "question", "reference", "generated", "extracted_answer", "words", "word_ids",
    "generated_token_ids", "model", "windows", "token_delta", "window_token_delta",
    "correct_source", "task",
}


def label_of(model_path: str) -> str:
    return "medgemma" if "medgemma" in model_path.lower() else "gemma"


def export_run(run_dir: Path, out: Path) -> None:
    for json_path in sorted(run_dir.glob("results_vqa_rad_*_k1.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        meta, curve, rows = data["meta"], data["curve"], data["results"]
        model = label_of(meta["model"])
        block = meta["block_type"]
        tag = f"{model}_{block}"

        # 1. mean per-layer curve (the aggregate plotted in the paper)
        with open(out / "curves" / f"{tag}_mean_curve.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "info_flow", "info_flow_logprob", "std"])
            for row in zip(curve["layer"], curve["info_flow"],
                           curve["info_flow_logprob"], curve["std"]):
                w.writerow(row)

        # 2. raw per-sample, per-layer information flow (one row per sample)
        n_layers = meta["num_layers"]
        with open(out / "per_layer_if" / f"{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id"] + [f"layer_{i}" for i in range(n_layers)])
            for r in rows:
                by_layer = {win["start"]: win["info_flow"] for win in r["windows"]}
                w.writerow([r["id"]] + [by_layer.get(i, "") for i in range(n_layers)])

        # 3. per-sample numeric metrics (no text)
        numeric = [k for k in rows[0] if k not in TEXT_FIELDS]
        with open(out / "per_sample" / f"{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=numeric, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in numeric})
        print(f"exported {tag}: {len(rows)} samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="output directories")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("curves", "per_layer_if", "per_sample"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        export_run(Path(run), out)


if __name__ == "__main__":
    main()
