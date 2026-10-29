"""the probe experiment runner.

Causal attention-knockout tracing on a VLM, evaluated on VQA-RAD (radiology VQA).

Example
-------
python run_tracing.py --checkpoint_dir google/gemma-3-4b-it \
    --block_types image_to_last,question_to_last --k 1 \
    --num_samples 30 --answer_path outputs_gemma
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from attn_knockout.attention_blocking import BLOCK_TYPES
from attn_knockout.metrics import image_entry_layer, question_dominance_ratio
from attn_knockout.synthetic_data import (
    HF_VQARAD_REPO,
    load_annotations,
    load_hf_vqarad,
)
from attn_knockout.tracing import aggregate_curve, trace_sample
from attn_knockout.vlm import DEFAULT_MODEL_ID, QUANTIZATION_CHOICES, SmallVLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the probe: trace multimodal information flow in a VLM")
    parser.add_argument(
        "--task",
        choices=("vqa_rad",),
        default="vqa_rad",
        help="which task/dataset to run (vqa_rad = radiology VQA from the HF Hub)",
    )
    parser.add_argument("--checkpoint_dir", default=DEFAULT_MODEL_ID, help="HuggingFace model id or local path")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--dtype", default="auto", help="model dtype: 'auto', float32, float16, bfloat16")
    parser.add_argument(
        "--quantization",
        default="none",
        choices=QUANTIZATION_CHOICES,
        help="weight-only quantization: none | int8 (optimum-quanto) | int4 (hqq)",
    )
    parser.add_argument(
        "--image_max_pixels",
        type=int,
        default=None,
        help="max pixels per image (Qwen2.5-VL); lower = fewer visual tokens = faster on CPU",
    )
    parser.add_argument("--image_dir", default=None, help="directory containing the images (optional)")
    parser.add_argument("--data_path", default=None, help="annotation JSON (optional; overrides VQA-RAD)")
    parser.add_argument(
        "--hf_dataset",
        default=None,
        help="HuggingFace dataset repo id to use instead of the default, e.g. " + HF_VQARAD_REPO,
    )
    parser.add_argument("--hf_split", default="val", help="split for --hf_dataset (e.g. val/test)")
    parser.add_argument("--answer_path", default="outputs", help="output directory for the results")
    parser.add_argument(
        "--block_types",
        default="image_to_last",
        help="comma separated block types, or 'all'. Choices: " + ", ".join(BLOCK_TYPES),
    )
    parser.add_argument("--k", type=int, default=1, help="layer window size K of the sliding intervention")
    parser.add_argument("--num_samples", type=int, default=8, help="number of synthetic samples to create")
    parser.add_argument("--start_index", type=int, default=0, help="first dataset index (for splitting a run across GPUs)")
    parser.add_argument("--limit", type=int, default=None, help="process at most N samples")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="max generated answer length")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", default="data", help="where synthetic data is created / cached")
    return parser.parse_args()


def resolve_image_path(sample: dict, image_dir: str | None) -> str:
    path = Path(sample["image"])
    if image_dir:
        candidate = Path(image_dir) / path.name
        if candidate.exists():
            return str(candidate)
    return str(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.answer_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        samples = load_annotations(args.data_path)
        print(f"[the probe] using annotations: {args.data_path} ({len(samples)} samples)")
    else:
        split = "test" if args.hf_split == "val" else args.hf_split
        samples, ann_path = load_hf_vqarad(
            args.data_dir,
            num_samples=args.num_samples,
            split=split,
            start_index=args.start_index,
            repo_id=args.hf_dataset or HF_VQARAD_REPO,
        )
        print(f"[the probe] using VQA-RAD: {ann_path} ({len(samples)} samples)")
    if args.limit:
        samples = samples[: args.limit]

    block_types = list(BLOCK_TYPES) if args.block_types == "all" else [b.strip() for b in args.block_types.split(",")]
    for bt in block_types:
        if bt not in BLOCK_TYPES:
            raise SystemExit(f"invalid block type {bt!r}; choices: {', '.join(BLOCK_TYPES)}")

    print(f"[the probe] model={args.checkpoint_dir} task={args.task} samples={len(samples)} k={args.k}")
    vlm = SmallVLM(
        args.checkpoint_dir,
        device=args.device,
        dtype=args.dtype,
        image_max_pixels=args.image_max_pixels,
        quantization=args.quantization,
    )
    print(
        f"[the probe] loaded {vlm.model_id} | arch={vlm.model_type} | text layers={vlm.num_layers} | "
        f"image_token_id={vlm.image_token_id} | device={vlm.device} | "
        f"dtype={vlm.dtype} | quantization={vlm.quantization}"
    )

    curves_by_block: dict[str, list[float]] = {}
    for block_type in block_types:
        t0 = time.time()

        # resume: load any results already written for this block type
        out_file = out_dir / f"results_{args.task}_{block_type}_k{args.k}.json"
        done_ids: set = set()
        results: list = []
        if out_file.exists():
            try:
                prev = json.loads(out_file.read_text(encoding="utf-8"))
                results = prev.get("results", [])
                done_ids = {r["id"] for r in results}
                print(f"[the probe] resume: {len(done_ids)} samples already done for {block_type}, skipping them")
            except Exception:
                results = []

        for i, sample in enumerate(samples):
            if sample.get("id", f"sample_{i}") in done_ids:
                continue
            image = Image.open(resolve_image_path(sample, args.image_dir)).convert("RGB")
            result = trace_sample(
                vlm,
                image,
                sample["question"],
                reference=sample.get("answer"),
                answers=sample.get("answers"),
                block_type=block_type,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                sample_id=sample.get("id", f"sample_{i}"),
                task=args.task,
                question_type=sample.get("question_type"),
                answer_type=sample.get("answer_type"),
                image_organ=sample.get("image_organ"),
            )
            results.append(result)

            # incremental write, so a crash loses at most 1 sample
            out_file.write_text(
                json.dumps({"meta": {"model": vlm.model_id, "task": args.task, "block_type": block_type,
                                     "k": args.k, "num_layers": vlm.num_layers,
                                     "num_samples": len(results), "max_new_tokens": args.max_new_tokens},
                            "curve": {}, "results": results}, indent=2),
                encoding="utf-8",
            )

            if result["windows"]:
                peak = max(result["windows"], key=lambda w: w["info_flow"])
                print(
                    f"  [{block_type}] {i + 1}/{len(samples)} id={result['id']} "
                    f"tokens={result['num_generated_tokens']} "
                    f"IF={result['windows'][0]['info_flow']:+.4f} "
                    f"(peak layer {peak['start']}: {peak['info_flow']:+.4f})"
                )
            else:
                print(f"  [{block_type}] {i + 1}/{len(samples)} id={result['id']} (no tokens generated)")

        curve = aggregate_curve(results)

        # --- per-run summary ------------------------------------------------
        scored = [r for r in results if r.get("correct") is not None and r.get("closed_ended")]
        acc = sum(r["correct"] for r in scored) / len(scored) if scored else None
        em = sum(r.get("exact_match") or 0 for r in scored) / len(scored) if scored else None

        closed = [r for r in scored if r.get("closed_ended")]
        open_  = [r for r in scored if r.get("closed_ended") is False]
        acc_closed = sum(r["correct"] for r in closed) / len(closed) if closed else None
        acc_open   = sum(r["correct"] for r in open_) / len(open_) if open_ else None

        conf_vals = [r["answer_confidence"] for r in results if r.get("answer_confidence") is not None]
        mean_conf = sum(conf_vals) / len(conf_vals) if conf_vals else None

        conf_corr = [r["answer_confidence"] for r in scored if r.get("answer_confidence") is not None]
        conf_incorr = [r["answer_confidence"] for r in scored
                       if not r["correct"] and r.get("answer_confidence") is not None]
        conf_corr_only = [r["answer_confidence"] for r in scored
                          if r["correct"] and r.get("answer_confidence") is not None]

        print(f"[the probe] {block_type} summary:")
        if acc is not None:
            print(f"  accuracy (loose)  : {acc:.3f}  ({sum(r['correct'] for r in scored)}/{len(scored)})")
        if em is not None:
            print(f"  exact match       : {em:.3f}")
        if acc_closed is not None:
            print(f"  closed (yes/no)   : {acc_closed:.3f}  (n={len(closed)})")
        if mean_conf is not None:
            print(f"  mean log-prob     : {mean_conf:.3f}")
        if conf_corr_only and conf_incorr:
            mc = sum(conf_corr_only) / len(conf_corr_only)
            mi = sum(conf_incorr) / len(conf_incorr)
            print(f"  log-prob correct  : {mc:.3f}   incorrect: {mi:.3f}  (Δ={mc-mi:+.3f})")

        run_summary = {
            "accuracy": round(acc, 4) if acc is not None else None,
            "exact_match": round(em, 4) if em is not None else None,
            "accuracy_closed": round(acc_closed, 4) if acc_closed is not None else None,
            "accuracy_open": None,  # not evaluated (truncated generations)
            "n_closed": len(closed),
            "n_open": len(open_),
            "mean_answer_confidence": round(mean_conf, 4) if mean_conf is not None else None,
            "mean_conf_correct": round(sum(conf_corr_only) / len(conf_corr_only), 4) if conf_corr_only else None,
            "mean_conf_incorrect": round(sum(conf_incorr) / len(conf_incorr), 4) if conf_incorr else None,
        }

        payload = {
            "meta": {
                "model": vlm.model_id,
                "decoder_model": getattr(vlm, "decoder_model_id", vlm.model_id),
                "frontend_model": getattr(vlm, "frontend_model_id", vlm.model_id),
                "task": args.task,
                "block_type": block_type,
                "k": args.k,
                "num_layers": vlm.num_layers,
                "num_samples": len(results),
                "max_new_tokens": args.max_new_tokens,
            },
            "run_summary": run_summary,
            "curve": curve,
            "results": results,
        }
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        curves_by_block[block_type] = curve["info_flow"]
        print(f"[the probe] wrote {out_file}  ({time.time() - t0:.1f}s)")

    img = curves_by_block.get("image_to_last")
    if img is not None:
        entry = image_entry_layer(img)
        print(f"[the probe] image_entry_layer (>=50% of peak) = {entry} / {vlm.num_layers}")
        q = curves_by_block.get("question_to_last")
        if q is not None:
            print(f"[the probe] question_dominance_ratio = {question_dominance_ratio(q, img):.2f}")


if __name__ == "__main__":
    main()
