#!/usr/bin/env python3
"""Deep, GPU-cheap XAI probes for the base-vs-finetuned comparison.

Everything here is derived from ONE forward pass per sample (output_hidden_states
+ optional output_attentions), so it adds seconds of GPU on top of loading the
model -- no extra per-layer intervention loop. It complements the attention-
blocking info-flow already in run_tracing.py with complementary SOTA signals:

  1. Logit Lens trajectory  -- project every decoder layer's residual at the
     answer position through the final norm + unembedding; record P(answer),
     entropy, and the layer at which the answer first becomes top-1.
     => "answer-formation depth": does fine-tuning form the answer earlier?
  2. Logit-lens Direct Logit Attribution (DLA) -- per-layer increment to the
     answer logit (lens(h_l) - lens(h_{l-1})). => which layers write the answer.
  3. Entropy trajectory -- uncertainty collapse across depth.
  4. Attention-to-image mass per layer (best-effort; needs eager attention)
     -- fraction of the answer position's attention landing on image tokens.
     => visual grounding by layer, an attention-side check on the info-flow.
  5. Encoder-divergence features -- layer-0 image-token features (shared vision
     interface) saved so --combine can compute CKA/cosine (the encoder confound
     bound, same as encoder_probe.py).
  6. Final-position logits saved (fp16) so --combine computes cross-model
     KL / Jensen-Shannon: how far fine-tuning moved the output distribution.

Per-model pass loads the model once:
    python analysis/deep_xai.py --model_dir "$GEMMA"    --label gemma    --out_dir figures_compare_fixed/xai --num_samples 100
    python analysis/deep_xai.py --model_dir "$MEDGEMMA" --label medgemma --out_dir figures_compare_fixed/xai --num_samples 100
Then (no GPU, no model load) combine across models:
    python analysis/deep_xai.py --combine --out_dir figures_compare_fixed/xai

Writes xai_<label>.json (+ _logits.npy, _imgfeat.npy) per model and, after
--combine, xai_divergence.json and encoder_probe.json (confound bound).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import glob
import json
import os
from pathlib import Path

import numpy as np

try:                        # torch is only needed for the per-model GPU pass
    import torch            # (combine/ KL/ CKA are numpy-only), so stay importable
except ImportError:         # on a machine without torch (local analysis box)
    torch = None


# ----------------------------------------------------------------------------- helpers
def _find_final_norm(model):
    """Locate the RMSNorm applied just before the unembedding, trying the common
    Gemma3 / LLaVA-style nesting. Returns the module or None (then lens is applied
    without the norm -- still informative, flagged in the output)."""
    paths = [
        "model.language_model.norm", "language_model.model.norm",
        "model.language_model.model.norm", "model.model.norm",
        "model.text_model.norm", "model.model.language_model.norm",
    ]
    for p in paths:
        obj = model
        try:
            for part in p.split("."):
                obj = getattr(obj, part)
            if obj is not None and hasattr(obj, "forward"):
                return obj
        except AttributeError:
            continue
    return None


def _entropy(logp: torch.Tensor) -> float:
    p = logp.exp()
    return float(-(p * logp).sum().item())


# ----------------------------------------------------------------------------- per model
def run_model(model_dir: str, label: str, out_dir: Path, num_samples: int,
              hf_split: str, dtype: str, device: str) -> None:
    from attn_knockout.vlm import SmallVLM
    from attn_knockout.synthetic_data import load_hf_vqarad

    out_dir.mkdir(parents=True, exist_ok=True)
    # VQA-RAD has no 'val' split; run_tracing maps val->test, so match it exactly
    # (same split + indices => same sample ids => encoder_control can pair them).
    split = "test" if hf_split == "val" else hf_split
    samples, _ = load_hf_vqarad(out_dir, split=split, num_samples=num_samples)
    vlm = SmallVLM(model_dir, device=device, dtype=dtype, quantization="none")

    lm_head = vlm.model.get_output_embeddings()
    final_norm = _find_final_norm(vlm.model)
    normed = final_norm is not None

    def lens(h_row: torch.Tensor) -> torch.Tensor:
        """(hidden,) residual -> (vocab,) log-probs via final norm + unembedding."""
        x = h_row.unsqueeze(0)
        if normed:
            x = final_norm(x)
        logits = lm_head(x).float().squeeze(0)
        return torch.log_softmax(logits, dim=-1)

    per_sample = []
    lens_lp, lens_ent, dla_all, attn_img = [], [], [], []   # aligned per-layer arrays
    last_logits = []                                        # fp16 final logits for KL
    img_feats = []                                          # layer-0 image features for CKA
    layer_feats = []                                        # per-layer pooled img feats

    for s in samples:
        try:
            inputs = vlm.build_inputs(s["image"], s["question"])
            ids = inputs["input_ids"]
            extra = {k: v for k, v in inputs.items()
                     if k not in ("input_ids", "attention_mask", "token_type_ids")}
            if inputs.get("token_type_ids") is not None:
                extra["token_type_ids"] = (ids == vlm.image_token_id).long()
            def _fwd(with_attn):
                with torch.no_grad():
                    return vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     use_cache=False, output_hidden_states=True,
                                     output_attentions=with_attn, **extra)
            try:                      # attentions need an eager backend; degrade gracefully
                out = _fwd(True)
            except Exception:
                out = _fwd(False)
            hs = out.hidden_states                      # tuple (L+1) of (1,seq,hid)
            last = ids.shape[1] - 1                      # predicts the first answer token
            final_logits = out.logits[0, last].float()
            answer_tok = int(final_logits.argmax())
            last_logits.append(final_logits.half().cpu().numpy())

            # layer-0 image features (shared vision interface) for encoder CKA
            img_mask = (ids[0] == vlm.image_token_id)
            if img_mask.any():
                img_feats.append(hs[0][0][img_mask].float().cpu().numpy())
                # per-layer mean-pooled image-token features -> GLOBAL layer-wise
                # CKA drift curve base-vs-finetuned (Phang 2021; Matsoukas CVPR'22)
                pooled = np.stack([h[0][img_mask].float().mean(0).cpu().numpy() for h in hs])
                layer_feats.append(pooled.astype(np.float16))   # (L+1, hidden)

            # logit-lens trajectory + DLA at the answer position
            lp_layers, ent_layers, dla_layers = [], [], []
            prev_logit = None
            for h in hs:
                logp = lens(h[0, last])
                lp_layers.append(float(logp[answer_tok]))
                ent_layers.append(_entropy(logp))
                cur = float(logp[answer_tok])
                dla_layers.append(0.0 if prev_logit is None else cur - prev_logit)
                prev_logit = cur
            lens_lp.append(lp_layers); lens_ent.append(ent_layers); dla_all.append(dla_layers)
            # first layer where the lens top-1 equals the final answer
            form_layer = next((i for i, h in enumerate(hs)
                               if int(lens(h[0, last]).argmax()) == answer_tok), len(hs) - 1)

            # attention-to-image per layer (best-effort: sdpa returns None)
            a2i = []
            if out.attentions and out.attentions[0] is not None:
                for att in out.attentions:                   # (1,heads,seq,seq)
                    row = att[0, :, last, :].mean(0)          # mean over heads
                    a2i.append(float(row[img_mask].sum().item()))
                attn_img.append(a2i)

            per_sample.append({
                "id": s.get("id"), "answer_token": answer_tok,
                "answer_formation_layer": int(form_layer),
                "answer_formation_frac": float(form_layer) / max(1, len(hs) - 1),
                "final_answer_logprob": lp_layers[-1],
                "final_entropy": ent_layers[-1],
            })
        except Exception as e:                               # never lose the whole run
            per_sample.append({"id": s.get("id"), "error": repr(e)})

    # aggregate curves (mean over samples), ragged-safe
    def mean_curve(rows):
        if not rows:
            return []
        L = min(len(r) for r in rows)
        return [float(np.mean([r[i] for r in rows])) for i in range(L)]

    summary = {
        "label": label, "model_dir": model_dir, "n_samples": len(per_sample),
        "n_ok": sum(1 for r in per_sample if "error" not in r),
        "lens_applied_final_norm": normed,
        "attention_available": bool(attn_img),
        "curve_lens_answer_logprob": mean_curve(lens_lp),
        "curve_lens_entropy": mean_curve(lens_ent),
        "curve_lens_dla": mean_curve(dla_all),
        "curve_attn_to_image": mean_curve(attn_img),
        "mean_answer_formation_frac": float(np.mean(
            [r["answer_formation_frac"] for r in per_sample if "error" not in r])) if per_sample else None,
        "per_sample": per_sample,
    }
    (out_dir / f"xai_{label}.json").write_text(json.dumps(summary, indent=2))
    if last_logits:
        np.save(out_dir / f"xai_{label}_logits.npy", np.stack(last_logits))
    if img_feats:
        try:
            np.save(out_dir / f"xai_{label}_imgfeat.npy", np.stack(img_feats))
        except ValueError:
            pass    # ragged image-token counts; CKA will skip
    if layer_feats:
        try:
            np.save(out_dir / f"xai_{label}_layerfeat.npy", np.stack(layer_feats))
        except ValueError:
            pass
    print(f"[deep_xai:{label}] ok={summary['n_ok']}/{summary['n_samples']} "
          f"norm={normed} attn={summary['attention_available']} "
          f"form_frac={summary['mean_answer_formation_frac']}")

    del vlm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- cross-model
def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / den) if den > 0 else float("nan")


def combine(out_dir: Path) -> None:
    # cross-model output divergence (KL / JS) from saved final-position logits
    gl = out_dir / "xai_gemma_logits.npy"; ml = out_dir / "xai_medgemma_logits.npy"
    div = {}
    if gl.exists() and ml.exists():
        a = np.load(gl).astype(np.float32); b = np.load(ml).astype(np.float32)
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        def softmax(x):
            x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)
        pa, pb = softmax(a), softmax(b)
        eps = 1e-9
        kl = float(np.mean(np.sum(pa * (np.log(pa + eps) - np.log(pb + eps)), axis=1)))
        m = 0.5 * (pa + pb)
        js = float(np.mean(0.5 * np.sum(pa * (np.log(pa + eps) - np.log(m + eps)), axis=1)
                           + 0.5 * np.sum(pb * (np.log(pb + eps) - np.log(m + eps)), axis=1)))
        div = {"n": n, "kl_base_to_medical_nats": kl, "jensen_shannon_nats": js,
               "note": "first-answer-token distribution shift from fine-tuning"}
    (out_dir / "xai_divergence.json").write_text(json.dumps(div, indent=2))

    # encoder-divergence (CKA/cosine) -- the vision-encoder confound bound
    gf = out_dir / "xai_gemma_imgfeat.npy"; mf = out_dir / "xai_medgemma_imgfeat.npy"
    if gf.exists() and mf.exists():
        fb = np.load(gf).astype(np.float32); fn = np.load(mf).astype(np.float32)
        if fb.shape == fn.shape:
            a = fb.reshape(-1, fb.shape[-1]); b = fn.reshape(-1, fn.shape[-1])
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
            enc = {"n_samples": int(fb.shape[0]), "n_image_tokens": int(fb.shape[1]),
                   "hidden_dim": int(fb.shape[2]),
                   "cosine_similarity_mean": float(np.mean(cos)),
                   "cosine_similarity_median": float(np.median(cos)),
                   "linear_cka": _linear_cka(a, b),
                   "interpretation": ("cosine/CKA near 1 => encoders produce near-identical "
                                      "decoder-input features, so the vision encoder cannot "
                                      "explain the QDR shift; lower values bound its share.")}
            # keep the canonical name the rest of the pipeline/paper expects
            Path(out_dir.parent / "encoder_probe.json").write_text(json.dumps(enc, indent=2))
            (out_dir / "encoder_probe.json").write_text(json.dumps(enc, indent=2))
            print(f"[deep_xai:combine] CKA={enc['linear_cka']:.4f} "
                  f"cos={enc['cosine_similarity_mean']:.4f}")
    # GLOBAL: layer-wise CKA(base, finetuned) -> representation-drift-by-depth curve
    glf = out_dir / "xai_gemma_layerfeat.npy"; mlf = out_dir / "xai_medgemma_layerfeat.npy"
    if glf.exists() and mlf.exists():
        A = np.load(glf).astype(np.float32); B = np.load(mlf).astype(np.float32)
        n = min(len(A), len(B)); A, B = A[:n], B[:n]          # (n, L+1, hidden)
        if A.shape[1:] == B.shape[1:]:
            L = A.shape[1]
            cka_by_layer = [_linear_cka(A[:, l, :], B[:, l, :]) for l in range(L)]
            lw = {"n_samples": int(n), "n_layers": int(L),
                  "cka_by_layer": [float(x) for x in cka_by_layer],
                  "min_cka": float(np.nanmin(cka_by_layer)),
                  "argmin_layer": int(np.nanargmin(cka_by_layer)),
                  "interpretation": ("per-layer CKA of image representations between base "
                                     "and fine-tuned decoder; dips mark the layers fine-tuning "
                                     "changed most (where medical routing is injected).")}
            (out_dir / "xai_layerwise_cka.json").write_text(json.dumps(lw, indent=2))
            print(f"[deep_xai:combine] layerwise CKA min={lw['min_cka']:.3f} @layer {lw['argmin_layer']}")
    print(f"[deep_xai:combine] divergence={div}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir"); ap.add_argument("--label")
    ap.add_argument("--out_dir", default="figures_compare_fixed/xai")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--hf_split", default="val")
    ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if args.combine:
        combine(out)
    else:
        assert args.model_dir and args.label, "need --model_dir and --label"
        run_model(args.model_dir, args.label, out, args.num_samples,
                  args.hf_split, args.dtype, args.device)


if __name__ == "__main__":
    main()
