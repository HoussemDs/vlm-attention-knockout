#!/usr/bin/env python3
"""Dual experiment tracking for the probe: log every paper metric to BOTH
Weights & Biases (cloud) and MLflow (local ./mlruns), plus the raw result JSONs,
figures, and analysis summaries as artifacts. Redundant on purpose: if one
tracker is down or renders something wrong, the other (and the raw files) still
hold the complete record.

Reads the post-reeval dirs by default (outputs_*_fixed/), falling back to the
raw kernel dirs. Safe to run repeatedly; missing inputs are skipped, not fatal.

Usage:
    python log_to_trackers.py                     # auto-discovers the dirs
    python log_to_trackers.py --tag v8-split
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLOCK_TYPES = ("image_to_last", "question_to_last")


def _first(d: Path, pattern: str):
    hits = sorted(d.glob(pattern))
    return hits[0] if hits else None


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def pick_dir(model: str) -> Path | None:
    for cand in (ROOT / f"outputs_{model}_fixed", ROOT / f"outputs_{model}"):
        if cand.is_dir() and any(cand.glob("results_*.json")):
            return cand
    return None


def flat_metrics() -> tuple[dict, dict, list[Path]]:
    """Return (scalar_metrics, params, artifact_paths) across both models."""
    metrics, params, artifacts = {}, {}, []
    for model in ("gemma", "medgemma"):
        d = pick_dir(model)
        if d is None:
            continue
        params[f"{model}_dir"] = str(d.relative_to(ROOT))
        for bt in BLOCK_TYPES:
            f = _first(d, f"results_*{bt}*.json")
            if not f:
                continue
            artifacts.append(f)
            data = _load(f) or {}
            meta = data.get("meta", {})
            rs = data.get("run_summary", {})
            params.setdefault(f"{model}_model", meta.get("model"))
            params.setdefault(f"{model}_num_samples", meta.get("num_samples"))
            params.setdefault(f"{model}_max_new_tokens", meta.get("max_new_tokens"))
            params.setdefault(f"{model}_num_layers", meta.get("num_layers"))
            for k, v in rs.items():
                if isinstance(v, (int, float)) and v is not None:
                    metrics[f"{model}/{bt}/{k}"] = float(v)
            # info-flow curve summary (peak layer = image/question entry layer)
            curve = data.get("curve", {})
            iflow = curve.get("info_flow") or []
            layers = curve.get("layer") or list(range(len(iflow)))
            if iflow:
                peak_i = max(range(len(iflow)), key=lambda i: iflow[i])
                metrics[f"{model}/{bt}/if_peak_value"] = float(iflow[peak_i])
                metrics[f"{model}/{bt}/if_peak_layer"] = float(layers[peak_i])
                metrics[f"{model}/{bt}/if_mean"] = float(sum(iflow) / len(iflow))
    # analysis summaries + deep-XAI outputs
    summ = ["figures_compare_fixed/metrics_summary.json",
            "figures_compare_fixed/stats_summary.json",
            "figures_compare_fixed/encoder_probe.json",
            "figures_compare_fixed/encoder_control.json",
            "figures_compare_fixed/xai/xai_divergence.json",
            "figures_compare_fixed/xai/xai_layerwise_cka.json",
            "figures_compare_fixed/xai/xai_gemma.json",
            "figures_compare_fixed/xai/xai_medgemma.json"]
    for name in summ:
        p = ROOT / name
        if p.exists():
            artifacts.append(p)
            payload = _load(p) or {}
            payload.pop("per_sample", None)   # keep scalars; raw file is the artifact
            prefix = Path(name).stem
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    metrics[f"{prefix}/{k}"] = float(v)
                elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    # log a curve's endpoints + peak so it's visible as scalars too
                    metrics[f"{prefix}/{k}/last"] = float(v[-1])
                    metrics[f"{prefix}/{k}/max"] = float(max(v))
    # figures (recursive: includes figures_compare_fixed/xai/*.png) + provenance
    figdir = ROOT / "figures_compare_fixed"
    if figdir.is_dir():
        artifacts += [Path(p) for p in glob.glob(str(figdir / "**/*.png"), recursive=True)]
        artifacts += [Path(p) for p in glob.glob(str(figdir / "**/*.pdf"), recursive=True)]
    for extra in ("figures_compare_fixed/reproducibility_manifest.json",
                  "results_bundle.zip"):
        p = ROOT / extra
        if p.exists():
            artifacts.append(p)
    return metrics, params, artifacts


def log_wandb(metrics, params, artifacts, tag):
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] unavailable: {e}")
        return
    try:
        run = wandb.init(project="attn-knockout", job_type="eval",
                         name=f"fgtracer-{tag}", config=params, reinit=True)
        run.log({k: v for k, v in metrics.items()})
        art = wandb.Artifact(f"fgtracer-results-{tag}", type="results")
        for p in artifacts:
            if Path(p).exists():
                art.add_file(str(p), name=str(Path(p).relative_to(ROOT)) if ROOT in Path(p).resolve().parents else Path(p).name)
        run.log_artifact(art)
        run.finish()
        print(f"[wandb] logged {len(metrics)} metrics, {len(artifacts)} artifacts")
    except Exception as e:
        print(f"[wandb] FAILED: {e}")


def log_mlflow(metrics, params, artifacts, tag):
    try:
        import mlflow
    except Exception as e:
        print(f"[mlflow] unavailable: {e}")
        return
    try:
        mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
        mlflow.set_experiment("attn-knockout")
        with mlflow.start_run(run_name=f"fgtracer-{tag}"):
            mlflow.log_params({k: v for k, v in params.items() if v is not None})
            mlflow.log_metrics(metrics)
            for p in artifacts:
                if Path(p).exists():
                    mlflow.log_artifact(str(p), artifact_path="results")
        print(f"[mlflow] logged {len(metrics)} metrics, {len(artifacts)} artifacts -> {ROOT/'mlruns'}")
    except Exception as e:
        print(f"[mlflow] FAILED: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    metrics, params, artifacts = flat_metrics()
    if not metrics:
        print("no metrics found yet (no result dirs) -- nothing to log")
        return
    print(f"discovered {len(metrics)} metrics, {len(artifacts)} artifacts")
    log_wandb(metrics, params, artifacts, args.tag)
    log_mlflow(metrics, params, artifacts, args.tag)


if __name__ == "__main__":
    main()
