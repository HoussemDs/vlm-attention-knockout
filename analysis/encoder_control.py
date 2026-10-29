#!/usr/bin/env python3
"""Encoder-controlled fine-tuning effect (CPU-only, no GPU).

Answers the reviewer objection "your grounding shift is just a better encoder,
not decoder fine-tuning" WITHOUT a model rerun, by combining:
  - per-sample encoder divergence  (1 - cosine of the two models' layer-0 image
    features, from analysis/deep_xai.py's saved features), and
  - per-sample grounding            (info-flow AUC of block_type image_to_last
    from each model's trace; shift = medical - base, matched by sample id).

It then isolates the decoder/fine-tuning effect from the encoder:
  1. raw paired effect              (mean grounding shift + paired t-test)
  2. ENCODER-MATCHED effect         (same test on the half of samples whose
     encoders are MOST similar -> if it survives here, the encoder isn't the driver)
  3. correlation(divergence, shift) (is the shift explained by encoder change?)
  4. regression shift ~ divergence  (intercept = effect at zero encoder change;
     the encoder-adjusted fine-tuning effect)

Output: figures_compare_fixed/encoder_control.json  (+ a plain-language verdict).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st
except Exception:
    _st = None


def _img_auc_by_id(trace_dir: Path) -> dict:
    """sample_id -> info-flow AUC (image_to_last), the grounding proxy."""
    f = next(iter(sorted(trace_dir.glob("results_*image_to_last*.json"))), None)
    if not f:
        return {}
    data = json.loads(f.read_text())
    out = {}
    for r in data.get("results", []):
        v = r.get("if_auc")
        if r.get("id") is not None and isinstance(v, (int, float)):
            out[str(r["id"])] = float(v)
    return out


def _divergence_by_id(xai_dir: Path) -> dict:
    """sample_id -> mean encoder divergence (1 - cosine of layer-0 image feats)."""
    gj, mj = xai_dir / "xai_gemma.json", xai_dir / "xai_medgemma.json"
    gf, mf = xai_dir / "xai_gemma_imgfeat.npy", xai_dir / "xai_medgemma_imgfeat.npy"
    if not (gj.exists() and mj.exists() and gf.exists() and mf.exists()):
        return {}
    gids = [str(r.get("id")) for r in json.loads(gj.read_text()).get("per_sample", [])]
    mids = [str(r.get("id")) for r in json.loads(mj.read_text()).get("per_sample", [])]
    A, B = np.load(gf).astype(np.float32), np.load(mf).astype(np.float32)
    # rows align with per_sample order; only rows that actually produced features
    n = min(len(A), len(gids)); A, gids = A[:n], gids[:n]
    m = min(len(B), len(mids)); B, mids = B[:m], mids[:m]
    midx = {i: k for k, i in enumerate(mids)}
    out = {}
    for k, sid in enumerate(gids):
        if sid in midx:
            a = A[k].reshape(-1); b = B[midx[sid]].reshape(-1)
            if a.shape == b.shape:
                cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                out[sid] = 1.0 - cos
    return out


def _ttest(x: np.ndarray):
    if _st is None or len(x) < 2:
        return {"mean": float(np.mean(x)) if len(x) else None, "p_value": None, "n": len(x)}
    t, p = _st.ttest_1samp(x, 0.0)
    return {"mean": float(np.mean(x)), "t": float(t), "p_value": float(p), "n": int(len(x))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="outputs_gemma_fixed")
    ap.add_argument("--new_dir", default="outputs_medgemma_fixed")
    ap.add_argument("--xai_dir", default="figures_compare_fixed/xai")
    ap.add_argument("--out", default="figures_compare_fixed/encoder_control.json")
    a = ap.parse_args()

    base = _img_auc_by_id(Path(a.base_dir))
    new = _img_auc_by_id(Path(a.new_dir))
    div = _divergence_by_id(Path(a.xai_dir))
    ids = sorted(set(base) & set(new) & set(div))
    result = {"n_matched": len(ids), "base_dir": a.base_dir, "new_dir": a.new_dir}

    if len(ids) >= 3:
        shift = np.array([new[i] - base[i] for i in ids])
        dvg = np.array([div[i] for i in ids])
        result["raw_finetuning_effect"] = _ttest(shift)
        # encoder-matched half (lowest divergence)
        order = np.argsort(dvg)
        half = order[: max(2, len(order) // 2)]
        result["encoder_matched_effect"] = _ttest(shift[half])
        result["encoder_divergence_summary"] = {
            "mean": float(np.mean(dvg)), "median": float(np.median(dvg)),
            "matched_subset_max_divergence": float(dvg[half].max())}
        if _st is not None and len(ids) >= 4:
            r, p = _st.pearsonr(dvg, shift)
            result["corr_divergence_vs_shift"] = {"pearson_r": float(r), "p_value": float(p)}
            sl, ic = np.polyfit(dvg, shift, 1)
            result["regression"] = {"slope": float(sl),
                                    "intercept_effect_at_zero_encoder_change": float(ic)}
        # plain verdict
        raw = result["raw_finetuning_effect"]; mat = result["encoder_matched_effect"]
        raw_sig = (raw.get("p_value") is not None and raw["p_value"] < 0.05 and raw["mean"] > 0)
        mat_sig = (mat.get("p_value") is not None and mat["p_value"] < 0.05 and mat["mean"] > 0)
        if raw_sig and mat_sig:
            v = ("Fine-tuning grounding effect is significant AND survives on "
                 "encoder-matched samples -> the effect is NOT explained by the "
                 "vision encoder; decoder/fine-tuning attribution holds.")
        elif raw_sig and not mat_sig:
            v = ("Effect is significant overall but weakens on encoder-matched "
                 "samples -> the encoder contributes; run the causal shared-encoder "
                 "control before claiming decoder attribution.")
        else:
            v = ("Grounding shift is not robustly significant on this sample size; "
                 "increase --num_samples before drawing conclusions.")
        result["verdict"] = v
    else:
        result["verdict"] = "insufficient matched samples (need deep_xai features + both traces)"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
