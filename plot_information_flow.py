"""Plot the information flow produced by the probe.

Reads the `results_*.json` files written by `run_tracing.py` and produces the
figures discussed in the paper:

    fig_general_information_flow.pdf : mean information flow per layer, per block type
    fig_k_ablation.pdf               : effect of the window size K
    fig_correct_vs_wrong.pdf         : correct vs. incorrect predictions

Example
-------
python plot_information_flow.py --answer_path outputs --task coco
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from attn_knockout.tracing import aggregate_curve  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the probe information flow")
    parser.add_argument("--answer_path", default="outputs", help="directory with results_*.json files")
    parser.add_argument("--output_dir", default="figures", help="directory where the PDFs are written")
    parser.add_argument("--task", default=None, help="restrict to a single task")
    parser.add_argument("--primary_block_type", default="image_to_last", help="block type used for the ablation plots")
    return parser.parse_args()


def load_payloads(answer_path: str, task: str | None = None) -> List[Dict]:
    payloads = []
    for path in sorted(Path(answer_path).glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if task and data["meta"]["task"] != task:
            continue
        payloads.append(data)
    return payloads


def _plot_curve(curve: Dict, label: str, color=None, linestyle="-"):
    if not curve["layer"]:
        return
    plt.plot(curve["layer"], curve["info_flow"], label=label, color=color, linestyle=linestyle, marker="o", ms=3)


def plot_general(payloads: List[Dict], out_dir: Path) -> None:
    """One curve per block type (using the smallest available K)."""
    plt.figure(figsize=(8, 4.5))
    plotted = False
    for payload in payloads:
        curve = payload["curve"] if payload.get("curve") else aggregate_curve(payload["results"])
        _plot_curve(curve, f'{payload["meta"]["block_type"]} (k={payload["meta"]["k"]})')
        plotted = plotted or bool(curve["layer"])
    if not plotted:
        plt.close()
        print("[plot] no data for the general information-flow figure")
        return
    plt.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    plt.xlabel("layer")
    plt.ylabel("normalised information flow")
    plt.title("the probe: layer-wise multimodal information flow")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig_general_information_flow.pdf"
    plt.savefig(path)
    plt.close()
    print(f"[plot] wrote {path}")


def plot_k_ablation(payloads: List[Dict], primary: str, out_dir: Path) -> None:
    """Curves for every window size K of the primary block type."""
    plt.figure(figsize=(8, 4.5))
    plotted = False
    for payload in payloads:
        if payload["meta"]["block_type"] != primary:
            continue
        curve = payload["curve"] if payload.get("curve") else aggregate_curve(payload["results"])
        _plot_curve(curve, f'K={payload["meta"]["k"]}')
        plotted = plotted or bool(curve["layer"])
    if not plotted:
        plt.close()
        print("[plot] no data for the K-ablation figure (run with several --k values)")
        return
    plt.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    plt.xlabel("layer")
    plt.ylabel("normalised information flow")
    plt.title(f"Window-size (K) ablation - {primary}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig_k_ablation.pdf"
    plt.savefig(path)
    plt.close()
    print(f"[plot] wrote {path}")


def plot_correct_vs_wrong(payloads: List[Dict], primary: str, out_dir: Path) -> None:
    """Correct vs. incorrect predictions for the primary block type."""
    candidates = [p for p in payloads if p["meta"]["block_type"] == primary]
    if not candidates:
        print("[plot] no data for the correct-vs-wrong figure")
        return
    payload = min(candidates, key=lambda p: p["meta"]["k"])

    plt.figure(figsize=(8, 4.5))
    plotted = False
    for flag, label in ((True, "correct"), (False, "incorrect")):
        subset = [r for r in payload["results"] if r.get("correct") is flag]
        curve = aggregate_curve(subset)
        _plot_curve(curve, f"{label} (n={len(subset)})")
        plotted = plotted or bool(curve["layer"])
    if not plotted:
        plt.close()
        print("[plot] no labelled correct/incorrect samples for the figure")
        return
    plt.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    plt.xlabel("layer")
    plt.ylabel("normalised information flow")
    plt.title(f"Correct vs. incorrect predictions - {primary}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig_correct_vs_wrong.pdf"
    plt.savefig(path)
    plt.close()
    print(f"[plot] wrote {path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_payloads(args.answer_path, args.task)
    if not payloads:
        raise SystemExit(f"no results_*.json found in {args.answer_path!r}")
    print(f"[plot] loaded {len(payloads)} result file(s)")

    plot_general(payloads, out_dir)
    plot_k_ablation(payloads, args.primary_block_type, out_dir)
    plot_correct_vs_wrong(payloads, args.primary_block_type, out_dir)


if __name__ == "__main__":
    main()
