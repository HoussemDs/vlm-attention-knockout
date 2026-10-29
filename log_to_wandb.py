"""Log the probe results to Weights & Biases.

Reads the fixed output JSONs (post-reeval_accuracy.py) and the generated figures,
then logs everything as a single W&B run. No model rerun required.

Usage:
    WANDB_API_KEY=<your_key> python log_to_wandb.py
    python log_to_wandb.py --old outputs_gemma_fixed --new outputs_medgemma_fixed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import wandb


def load_best_payload(answer_path: Path) -> dict | None:
    """Return the image_to_last payload from a fixed output dir."""
    for p in sorted(answer_path.glob("results_*image_to_last*.json")):
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def build_metrics(old_dir: Path, new_dir: Path) -> dict:
    """Assemble the paper metrics table from both models' fixed JSONs."""
    metrics = {}

    for label, d in [("gemma", old_dir), ("medgemma", new_dir)]:
        img_path = next(d.glob("results_*image_to_last*.json"), None)
        q_path = next(d.glob("results_*question_to_last*.json"), None)
        if not img_path:
            continue

        img_data = json.loads(img_path.read_text(encoding="utf-8"))
        rs = img_data.get("run_summary", {})
        curve = img_data.get("curve", {})
        info_flow = curve.get("info_flow", [])

        prefix = label
        metrics[f"{prefix}/accuracy"] = rs.get("accuracy")
        metrics[f"{prefix}/accuracy_closed"] = rs.get("accuracy_closed")
        metrics[f"{prefix}/mean_conf_correct"] = rs.get("mean_conf_correct")
        metrics[f"{prefix}/mean_conf_incorrect"] = rs.get("mean_conf_incorrect")

        # IF curve stats
        if info_flow:
            pos = [max(v, 0.0) for v in info_flow]
            total = sum(pos) or 1e-8
            metrics[f"{prefix}/if_auc"] = round(sum(info_flow) / len(info_flow), 6)
            metrics[f"{prefix}/if_peak_layer"] = int(max(range(len(info_flow)), key=lambda i: info_flow[i]))
            n = len(info_flow)
            # Gini
            sp = sorted(pos)
            gini_num = sum((2 * i - n - 1) * v for i, v in enumerate(sp, 1))
            metrics[f"{prefix}/if_gini"] = round(gini_num / (n * total), 4)

        # Question dominance ratio
        if q_path:
            q_data = json.loads(q_path.read_text(encoding="utf-8"))
            q_curve = q_data.get("curve", {})
            q_flow = q_curve.get("info_flow", [])
            if q_flow and info_flow:
                q_mean = sum(max(v, 0.0) for v in q_flow) / len(q_flow)
                i_mean = sum(max(v, 0.0) for v in info_flow) / len(info_flow)
                metrics[f"{prefix}/question_dominance_ratio"] = round(q_mean / (i_mean + 1e-8), 3)

        # Image entry layer
        if info_flow:
            peak = max(info_flow)
            if peak > 0:
                thr = 0.5 * peak
                for layer, v in enumerate(info_flow):
                    if v >= thr:
                        metrics[f"{prefix}/image_entry_layer"] = layer
                        metrics[f"{prefix}/image_entry_depth"] = round(layer / max(len(info_flow) - 1, 1), 3)
                        break

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="outputs_gemma_fixed")
    parser.add_argument("--new", default="outputs_medgemma_fixed")
    parser.add_argument("--figures_dir", default="figures_compare_fixed")
    parser.add_argument("--project", default="attn-knockout-medical-xai")
    parser.add_argument("--run_name", default="gemma-vs-medgemma-vqarad-v7")
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("ERROR: set WANDB_API_KEY env var before running.", file=sys.stderr)
        sys.exit(1)

    old_dir = Path(args.old)
    new_dir = Path(args.new)

    if not any(old_dir.glob("results_*.json")):
        print(f"ERROR: no results_*.json in {old_dir}. Run reeval_accuracy.py first.", file=sys.stderr)
        sys.exit(1)

    wandb.login(key=api_key)
    run = wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "dataset": "VQA-RAD",
            "n_samples": 50,
            "base_model": "google/gemma-3-4b-it",
            "medical_model": "google/medgemma-4b-it",
            "block_types": ["image_to_last", "question_to_last"],
            "quantization": "int4 (HQQ)",
            "max_new_tokens": 24,
            "kaggle_notebook_version": 7,
        },
    )

    # Log metrics
    metrics = build_metrics(old_dir, new_dir)
    wandb.log(metrics)
    print("Logged metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # Log figures
    fig_dir = Path(args.figures_dir)
    if fig_dir.exists():
        imgs = list(fig_dir.glob("*.png"))
        if imgs:
            wandb.log({p.stem: wandb.Image(str(p)) for p in sorted(imgs)})
            print(f"Logged {len(imgs)} figures from {fig_dir}")
    else:
        print(f"[INFO] No figures dir {fig_dir}. Run compare_models.py first, then re-run this script to add figures.")

    run.finish()
    print(f"\nW&B run: {run.url}")
    print("Add this URL to the paper's reproducibility section.")


if __name__ == "__main__":
    main()
