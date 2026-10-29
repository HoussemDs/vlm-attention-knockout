"""Compare the probe layer-wise information flow between two models.

Reads two `outputs*/` directories (one per model) produced by `run_tracing.py`
on the *same* samples, and produces:

    fig_layer_touch_diagram.pdf : which decoder layers the K-window touches, plus the
                                  per-layer information-flow curve of each model
    fig_model_layer_comparison.pdf : the two models overlaid on a normalised depth axis
    fig_blocktype_heatmap.pdf   : information flow per block type x layer, one panel/model

Example
-------
python compare_models.py --old outputs_hf --new outputs_qwen \
    --old_label SmolVLM-256M --new_label "Qwen2.5-VL-3B (int8)" \
    --block_type image_to_last --output_dir figures_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from attn_knockout.metrics import image_entry_layer, question_dominance_ratio  # noqa: E402
from attn_knockout.tracing import aggregate_curve  # noqa: E402


def save_fig(fig, path: Path) -> None:
    """Write both a PDF (paper-ready) and a PNG (quick viewing)."""
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the probe information flow across two models")
    parser.add_argument("--old", required=True, help="answer_path of the base model, e.g. outputs_gemma_full")
    parser.add_argument("--new", required=True, help="answer_path of the medical model, e.g. outputs_medgemma_full")
    parser.add_argument("--old_label", default="Gemma-3-4B (base)")
    parser.add_argument("--new_label", default="MedGemma-4B (medical)")
    parser.add_argument("--block_type", default="image_to_last", help="block type for the overlay figure")
    parser.add_argument("--output_dir", default="figures_compare")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_payloads(answer_path: str) -> Dict[str, Dict]:
    """Return {block_type: payload} keeping the smallest K available per block type."""
    payloads: Dict[str, Dict] = {}
    for path in sorted(Path(answer_path).glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        bt = data["meta"]["block_type"]
        if bt not in payloads or data["meta"]["k"] < payloads[bt]["meta"]["k"]:
            payloads[bt] = data
    return payloads


def curve_of(payload: Dict) -> Dict:
    return payload.get("curve") or aggregate_curve(payload["results"])


def summarise(payload: Dict) -> Dict:
    curve = curve_of(payload)
    if not curve["layer"]:
        return {"mean": 0.0, "peak_layer": -1, "peak": 0.0, "layers": 0}
    info = curve["info_flow"]
    peak_i = int(np.argmax(info))
    return {
        "mean": float(np.mean(info)),
        "peak_layer": int(curve["layer"][peak_i]),
        "peak": float(info[peak_i]),
        "layers": len(curve["layer"]),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_layer_touch_diagram(old: Dict, new: Dict, args, out_dir: Path) -> None:
    """Show the sliding-window coverage and the per-layer flow for both models."""
    bt = args.block_type if args.block_type in old and args.block_type in new else sorted(old)[0]
    old_p, new_p = old[bt], new[bt]
    k_old, k_new = old_p["meta"]["k"], new_p["meta"]["k"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 1]})
    for col, (payload_local, label, color) in enumerate(
        ((old_p, args.old_label, "tab:blue"), (new_p, args.new_label, "tab:red"))
    ):
        layers = payload_local["meta"]["num_layers"]
        kk = payload_local["meta"]["k"]
        touch = np.zeros((layers, layers))
        for start in range(layers):
            for l in range(start, min(start + kk, layers)):
                touch[start, l] = 1.0

        ax = axes[0, col]
        ax.imshow(touch, aspect="auto", cmap="Greys", origin="lower", vmin=0, vmax=1.4)
        ax.set_title(f"{label}\nwindow coverage (K={kk}), {layers} text layers", fontsize=10)
        ax.set_xlabel("layer index (touched)")
        ax.set_ylabel("window start layer")
        ax.annotate(
            "each row = one window [s, s+K)" ,
            xy=(0.5, 1.08), xycoords="axes fraction", ha="center", fontsize=8, color="dimgray",
        )

        summary = summarise(payload_local)
        curve = curve_of(payload_local)
        ax = axes[1, col]
        ax.bar(curve["layer"], curve["info_flow"], color=color, alpha=0.75)
        if summary["peak_layer"] >= 0:
            ax.axvline(summary["peak_layer"], color="black", linestyle="--", linewidth=1)
            ax.annotate(
                f"peak layer {summary['peak_layer']}\n({summary['peak']:+.4f})",
                xy=(summary["peak_layer"], summary["peak"]),
                xytext=(0.55, 0.9), textcoords="axes fraction", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            )
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_title(f"{label} - information flow per layer ({bt})", fontsize=10)
        ax.set_xlabel("layer index")
        ax.set_ylabel("information flow (IF)")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "the probe: which layers are touched, and their contribution\n"
        "A new window start is used for every layer; the window blocks the chosen "
        "pathway (here: image -> answer) only inside those layers.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "fig_layer_touch_diagram.pdf"
    save_fig(fig, path)
    plt.close(fig)
    print(f"[compare] wrote {path}")


def plot_model_overlay(old: Dict, new: Dict, args, out_dir: Path) -> None:
    """Overlay both models on a normalised depth axis (they may differ in #layers)."""
    bt = args.block_type
    if bt not in old or bt not in new:
        print(f"[compare] block type {bt!r} missing in one of the models - skipping overlay")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (payload, label, color) in zip(
        axes, ((old[bt], args.old_label, "tab:blue"), (new[bt], args.new_label, "tab:red"))
    ):
        curve = curve_of(payload)
        n = len(curve["layer"])
        x = np.arange(n) / max(n - 1, 1)
        ax.plot(x, curve["info_flow"], marker="o", ms=3, color=color, label=label)
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(f"{label} ({n} layers)", fontsize=10)
        ax.set_xlabel("normalised depth (0 = first, 1 = last)")
        ax.set_ylabel("information flow (IF)")
        ax.grid(alpha=0.3)

    # third-style combined summary printed on the right axis title
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Layer-wise information flow on the same samples - block type: {bt}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "fig_model_layer_comparison.pdf"
    save_fig(fig, path)
    plt.close(fig)
    print(f"[compare] wrote {path}")


def plot_blocktype_heatmap(old: Dict, new: Dict, args, out_dir: Path) -> None:
    """Heatmap of information flow (block type x layer) for each model."""
    block_types = [bt for bt in old if bt in new]
    if not block_types:
        print("[compare] no shared block types - skipping heatmap")
        return

    max_layers = max(payloads[bt]["meta"]["num_layers"] for payloads in (old, new) for bt in block_types)
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.0 + 0.5 * len(block_types)), sharex=True)
    for ax, (payloads, label) in zip(axes, ((old, args.old_label), (new, args.new_label))):
        grid = np.full((len(block_types), max_layers), np.nan)
        for row, bt in enumerate(block_types):
            curve = curve_of(payloads[bt])
            grid[row, : len(curve["info_flow"])] = curve["info_flow"]
        vmax = np.nanmax(np.abs(grid)) or 1.0
        im = ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label} ({payloads[block_types[0]]['meta']['num_layers']} layers)", fontsize=10)
        ax.set_yticks(range(len(block_types)))
        ax.set_yticklabels(block_types, fontsize=8)
        ax.set_xlabel("layer index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="info flow")

    fig.suptitle("Information flow per block type x layer (red = important pathway)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / "fig_blocktype_heatmap.pdf"
    save_fig(fig, path)
    plt.close(fig)
    print(f"[compare] wrote {path}")


def plot_finetuning_shift(old: Dict, new: Dict, args, out_dir: Path) -> None:
    """Paper Figure 1: base vs medical, image & question pathways on shared depth axes."""
    panels = [bt for bt in ("image_to_last", "question_to_last") if bt in old and bt in new]
    if not panels:
        print("[compare] need image_to_last/question_to_last in both models - skipping shift figure")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), squeeze=False)
    for ax, bt in zip(axes[0], panels):
        for payload, label, color in (
            (old[bt], args.old_label, "tab:blue"),
            (new[bt], args.new_label, "tab:red"),
        ):
            curve = curve_of(payload)
            n = len(curve["layer"])
            x = np.arange(n) / max(n - 1, 1)
            ax.plot(x, curve["info_flow"], marker="o", ms=3, color=color, label=label)
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(bt, fontsize=10)
        ax.set_xlabel("normalised depth (0 = first layer, 1 = last)")
        ax.set_ylabel("information flow (IF)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-layer information flow, base vs medical (absolute IF)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "fig_finetuning_shift.pdf"
    save_fig(fig, path)
    plt.close(fig)
    print(f"[compare] wrote {path}")


def paper_metrics(payloads: Dict) -> Dict:
    """Derived trust metrics for one model: image entry layer, dominance, peaks."""
    out: Dict = {}
    img = payloads.get("image_to_last")
    q = payloads.get("question_to_last")
    if img is not None:
        c = curve_of(img)
        n = len(c["layer"])
        # CANONICAL entry layer = per-sample (apply the >=50%-of-peak rule to each
        # sample's own curve, then take the median). This is the quantity that has a
        # distribution and a p-value in analysis/stats.py, so it is what the paper
        # reports. The aggregate-curve value below (rule applied to the mean curve)
        # is a different number and is kept only under an explicit *_aggcurve key so
        # the two are never confused again. See PREREGISTRATION.md section 3.
        per_sample_entries = []
        per_sample_depths = []
        for r in img.get("results", []):
            flow = [w["info_flow"] for w in r.get("windows", [])]
            e = image_entry_layer(flow)
            if e is not None:
                per_sample_entries.append(e)
                per_sample_depths.append(e / max(len(flow) - 1, 1))
        if per_sample_entries:
            out["image_entry_layer"] = float(np.median(per_sample_entries))
            out["image_entry_depth"] = float(np.median(per_sample_depths))
        else:
            out["image_entry_layer"] = None
            out["image_entry_depth"] = None
        # aggregate-curve version (rule applied to the mean curve) -- NOT the tested metric
        agg_entry = image_entry_layer(c["info_flow"])
        out["image_entry_layer_aggcurve"] = agg_entry
        out["image_entry_depth_aggcurve"] = (agg_entry / max(n - 1, 1)) if agg_entry is not None else None
        out["image_peak_layer"] = summarise(img)["peak_layer"]
        out["num_layers"] = n
    if img is not None and q is not None:
        out["question_dominance_ratio"] = round(
            question_dominance_ratio(curve_of(q)["info_flow"], curve_of(img)["info_flow"]), 3
        )
    return out


def accuracy_metrics(payload: Dict) -> Dict:
    """Accuracy and confidence breakdown from per-sample results."""
    results = payload.get("results", [])
    # prefer pre-computed run_summary when available
    rs = payload.get("run_summary")
    if rs:
        return rs

    scored = [r for r in results if r.get("correct") is not None and r.get("closed_ended")]
    if not scored:
        return {}

    acc = sum(r["correct"] for r in scored) / len(scored)
    em = sum(r.get("exact_match") or 0 for r in scored) / len(scored)
    closed = [r for r in scored if r.get("closed_ended")]
    open_  = [r for r in scored if r.get("closed_ended") is False]
    conf_c = [r["answer_confidence"] for r in scored if r["correct"] and r.get("answer_confidence") is not None]
    conf_i = [r["answer_confidence"] for r in scored if not r["correct"] and r.get("answer_confidence") is not None]

    out = {
        "accuracy": round(acc, 4),
        "exact_match": round(em, 4),
        "accuracy_closed": round(sum(r["correct"] for r in closed) / len(closed), 4) if closed else None,
        "accuracy_open": None,  # not evaluated (truncated generations)
        "n_closed": len(closed),
        "n_open": len(open_),
        "mean_conf_correct": round(sum(conf_c) / len(conf_c), 4) if conf_c else None,
        "mean_conf_incorrect": round(sum(conf_i) / len(conf_i), 4) if conf_i else None,
    }
    return out


def plot_accuracy_breakdown(old: Dict, new: Dict, args, out_dir: Path) -> None:
    """Bar chart: overall / closed / open accuracy for both models."""
    bt = next(iter(old))
    am_old = accuracy_metrics(old[bt])
    am_new = accuracy_metrics(new[bt])
    if not am_old or not am_new:
        return

    keys = ["accuracy_closed"]
    labels = ["Closed (yes/no)"]
    x = np.arange(len(keys))
    w = 0.35

    vals_old = [am_old.get(k) or 0 for k in keys]
    vals_new = [am_new.get(k) or 0 for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_old = ax.bar(x - w / 2, vals_old, w, label=args.old_label, color="tab:blue", alpha=0.8)
    bars_new = ax.bar(x + w / 2, vals_new, w, label=args.new_label, color="tab:red", alpha=0.8)
    for bar in list(bars_old) + list(bars_new):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={am_old.get('n_closed','?')})" for l in labels], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("VQA-RAD Accuracy: base vs medical model", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "fig_accuracy_breakdown.pdf"
    save_fig(fig, path)
    plt.close(fig)
    print(f"[compare] wrote {path}")


def plot_confidence_comparison(old: Dict, new: Dict, args, out_dir: Path) -> None:
    """Box plot of answer log-probability split by correct / incorrect."""
    bt = next(iter(old))

    def _split(payload):
        results = payload.get("results", [])
        c = [r["answer_confidence"] for r in results if r["correct"] and r.get("answer_confidence") is not None]
        i = [r["answer_confidence"] for r in results if r.get("correct") is False and r.get("answer_confidence") is not None]
        return c, i

    c_old, i_old = _split(old[bt])
    c_new, i_new = _split(new[bt])

    if not (c_old or i_old or c_new or i_new):
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, (c, i, label, color) in zip(
        axes,
        [(c_old, i_old, args.old_label, "tab:blue"), (c_new, i_new, args.new_label, "tab:red")],
    ):
        data = [d for d in [c, i] if d]
        tick_labels = [f"Correct\n(n={len(c)})", f"Incorrect\n(n={len(i)})"][:len(data)]
        bp = ax.boxplot(data, patch_artist=True, widths=0.4)
        for patch, col in zip(bp["boxes"], [color, "lightgray"]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Mean log-prob (answer confidence)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Model confidence on correct vs incorrect answers", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "fig_confidence_comparison.pdf"
    save_fig(fig, path)
    plt.close(fig)
    print(f"[compare] wrote {path}")


def plot_correct_vs_incorrect_if(old: Dict, new: Dict, args, out_dir: Path) -> None:
    """IF curve averaged over correct vs incorrect samples. Do they differ?"""
    bt = args.block_type
    if bt not in old or bt not in new:
        return

    def _split_curves(payload):
        results = [r for r in payload["results"] if r.get("windows") and r.get("correct") is not None]
        n_layers = max((len(r["windows"]) for r in results), default=0)
        if not n_layers:
            return None, None, 0, 0

        def mean_curve(rs):
            if not rs:
                return []
            n = min(len(r["windows"]) for r in rs)
            return [sum(r["windows"][i]["info_flow"] for r in rs) / len(rs) for i in range(n)]

        correct = [r for r in results if r["correct"]]
        incorrect = [r for r in results if not r["correct"]]
        return mean_curve(correct), mean_curve(incorrect), len(correct), len(incorrect)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (payload, label, color) in zip(
        axes, [(old[bt], args.old_label, "tab:blue"), (new[bt], args.new_label, "tab:red")]
    ):
        cc, ic, nc, ni = _split_curves(payload)
        if cc:
            x_c = np.arange(len(cc)) / max(len(cc) - 1, 1)
            ax.plot(x_c, cc, color=color, label=f"Correct (n={nc})", linewidth=2)
        if ic:
            x_i = np.arange(len(ic)) / max(len(ic) - 1, 1)
            ax.plot(x_i, ic, color=color, linestyle="--", alpha=0.6, label=f"Incorrect (n={ni})", linewidth=2)
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(f"{label}: {bt}", fontsize=10)
        ax.set_xlabel("normalised depth")
        ax.set_ylabel("mean info flow")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Information flow: correct vs incorrect predictions", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "fig_if_correct_vs_incorrect.pdf"
    save_fig(fig, path)
    plt.close(fig)
    print(f"[compare] wrote {path}")


def plot_if_auc_vs_correctness(old: Dict, new: Dict, args, out_dir: Path) -> None:
    """Scatter of IF AUC (image_to_last) vs correctness. Visual correlation check."""
    bt = "image_to_last"
    if bt not in old or bt not in new:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (payload, label, color) in zip(
        axes, [(old[bt], args.old_label, "tab:blue"), (new[bt], args.new_label, "tab:red")]
    ):
        results = [r for r in payload["results"] if r.get("correct") is not None and r.get("if_auc") is not None]
        if not results:
            continue
        aucs = [r["if_auc"] for r in results]
        correct = [int(r["correct"]) for r in results]
        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(correct))
        ax.scatter(aucs, np.array(correct) + jitter, c=color, alpha=0.5, s=25)
        # mean AUC per group
        auc_c = [a for a, c in zip(aucs, correct) if c]
        auc_i = [a for a, c in zip(aucs, correct) if not c]
        if auc_c:
            ax.axvline(np.mean(auc_c), color=color, linewidth=1.5, label=f"mean correct: {np.mean(auc_c):.4f}")
        if auc_i:
            ax.axvline(np.mean(auc_i), color=color, linewidth=1.5, linestyle="--", label=f"mean incorrect: {np.mean(auc_i):.4f}")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Incorrect", "Correct"])
        ax.set_xlabel("IF AUC (image_to_last)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Does higher image information flow → more correct answers?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "fig_if_auc_vs_correctness.pdf"
    save_fig(fig, path)
    plt.close(fig)
    print(f"[compare] wrote {path}")


def report_paper_metrics(old: Dict, new: Dict, args, out_dir: Path) -> None:
    m_old, m_new = paper_metrics(old), paper_metrics(new)
    bt = next(iter(old))
    am_old = accuracy_metrics(old[bt])
    am_new = accuracy_metrics(new[bt])

    print("\n[compare] ===== paper metrics (the H1/H2/H3 answer) =====")
    print(f"  {'metric':30s} {args.old_label:>22s} {args.new_label:>22s}")

    all_keys = [
        ("image_entry_layer", m_old, m_new),
        ("image_entry_depth", m_old, m_new),
        ("image_peak_layer", m_old, m_new),
        ("question_dominance_ratio", m_old, m_new),
        ("accuracy_closed", am_old, am_new),
        ("mean_conf_correct", am_old, am_new),
        ("mean_conf_incorrect", am_old, am_new),
    ]
    for key, d_old, d_new in all_keys:
        print(f"  {key:30s} {str(d_old.get(key)):>22s} {str(d_new.get(key)):>22s}")

    # IF curve shape stats (from image_to_last if available)
    for label, payloads in [(args.old_label, old), (args.new_label, new)]:
        if_bt = payloads.get("image_to_last")
        if if_bt:
            results = if_bt.get("results", [])
            if results:
                auc_vals = [r["if_auc"] for r in results if r.get("if_auc") is not None]
                gini_vals = [r["if_gini"] for r in results if r.get("if_gini") is not None]
                if auc_vals:
                    print(f"  {'mean IF AUC':30s} [{label}] {sum(auc_vals)/len(auc_vals):.5f}")
                if gini_vals:
                    print(f"  {'mean IF Gini':30s} [{label}] {sum(gini_vals)/len(gini_vals):.4f}")

    summary = {
        args.old_label: {**m_old, **am_old},
        args.new_label: {**m_new, **am_new},
    }
    path = out_dir / "metrics_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[compare] wrote {path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old = load_payloads(args.old)
    new = load_payloads(args.new)
    if not old or not new:
        raise SystemExit("both --old and --new must contain results_*.json files")

    old_layers = max(p["meta"]["num_layers"] for p in old.values())
    new_layers = max(p["meta"]["num_layers"] for p in new.values())
    print(f"[compare] {args.old_label}: {len(old)} block types, {old_layers} layers")
    print(f"[compare] {args.new_label}: {len(new)} block types, {new_layers} layers")

    shared = [bt for bt in old if bt in new]
    print("\n[compare] aggregate information flow (mean over layers):")
    print(f"  {'block type':18s} {args.old_label:>18s} {args.new_label:>18s}")
    for bt in shared:
        s_old, s_new = summarise(old[bt]), summarise(new[bt])
        print(
            f"  {bt:18s} mean={s_old['mean']:+.5f} "
            f"(peak L{s_old['peak_layer']:>2})   mean={s_new['mean']:+.5f} (peak L{s_new['peak_layer']:>2})"
        )

    for label, payloads in ((args.old_label, old), (args.new_label, new)):
        print(f"\n[compare] {label} generated answers:")
        for r in next(iter(payloads.values()))["results"]:
            print(f"  {r['id']} | correct={r.get('correct')} | {r['generated'][:70]!r}")

    report_paper_metrics(old, new, args, out_dir)
    plot_finetuning_shift(old, new, args, out_dir)
    plot_layer_touch_diagram(old, new, args, out_dir)
    plot_model_overlay(old, new, args, out_dir)
    plot_blocktype_heatmap(old, new, args, out_dir)
    plot_accuracy_breakdown(old, new, args, out_dir)
    plot_confidence_comparison(old, new, args, out_dir)
    plot_correct_vs_incorrect_if(old, new, args, out_dir)
    plot_if_auc_vs_correctness(old, new, args, out_dir)


if __name__ == "__main__":
    main()
