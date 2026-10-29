"""the probe tracing loop.

For every sample we:

1. generate the free-form answer under full attention (greedy);
2. record the baseline probability of every generated token;
3. for each sliding window of K consecutive layers, re-run the forward pass
   with the attention edges of the chosen source->target pair knocked out and
   record how the token probabilities changed.

Because the model is causal, a *single* forward pass with the mask applied to
all relevant query rows reproduces exactly the per-token intervention, so no
per-token autoregressive loop is needed.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

import torch

from .attention_blocking import BLOCK_TYPES, AttentionBlocker, build_block_mask
from .metrics import (
    answer_confidence,
    answer_perplexity,
    closed_ended,
    exact_match,
    first_token_entropy,
    group_tokens_into_words,
    if_curve_stats,
    is_correct,
    normalized_information_flow,
    normalized_logprob_information_flow,
    token_probabilities,
    token_word_boundaries,
)
from .vlm import SmallVLM


def compute_groups(input_ids_row: torch.Tensor, prompt_len: int, image_token_id: int):
    """Split sequence positions into image / question / generated groups.

    Returns (image_idx, question_idx, pred_positions, gen_token_positions):
      * pred_positions     - query positions that predict a generated token
      * gen_token_positions - positions of the generated tokens themselves
    """
    seq_len = int(input_ids_row.shape[0])
    image_idx = [i for i in range(seq_len) if int(input_ids_row[i]) == image_token_id]
    image_set = set(image_idx)
    question_idx = [i for i in range(prompt_len) if i not in image_set]
    pred_positions = list(range(prompt_len - 1, seq_len - 1))
    gen_token_positions = list(range(prompt_len, seq_len))
    return image_idx, question_idx, pred_positions, gen_token_positions


def trace_sample(
    vlm: SmallVLM,
    image,
    question: str,
    reference: str | None = None,
    answers: Sequence[str] | None = None,
    block_type: str = "image_to_last",
    k: int = 1,
    max_new_tokens: int = 64,
    sample_id: str = "",
    task: str = "",
    question_type: str | None = None,
    answer_type: str | None = None,
    image_organ: str | None = None,
) -> Dict:
    """Run the probe on a single (image, question) pair."""
    if block_type not in BLOCK_TYPES:
        raise ValueError(f"unknown block_type {block_type!r}; expected one of {BLOCK_TYPES}")

    inputs = vlm.build_inputs(image, question)
    full_ids, prompt_len = vlm.generate(inputs, max_new_tokens=max_new_tokens)
    seq_len = int(full_ids.shape[1])
    generated_text = vlm.decode(full_ids[0, prompt_len:].tolist())

    if seq_len <= prompt_len:
        # The model emitted EOS immediately: nothing to trace.
        return _empty_result(sample_id, task, question, reference, block_type, k, generated_text, seq_len, prompt_len)

    image_idx, question_idx, pred_positions, gen_tokens = compute_groups(
        full_ids[0], prompt_len, vlm.image_token_id
    )
    gen_ids = full_ids[0, prompt_len:].tolist()

    # --- baseline (full attention) -----------------------------------------
    baseline_logits = vlm.forward_logits(full_ids, inputs)
    p_full, logp_full = token_probabilities(baseline_logits, pred_positions, gen_ids)

    # confidence / uncertainty metrics (free, from baseline logits already computed)
    conf = answer_confidence(logp_full)
    perp = answer_perplexity(logp_full)
    fte = first_token_entropy(baseline_logits, pred_positions[0]) if pred_positions else float("nan")

    layers = vlm.text_layers()
    num_layers = len(layers)
    device = full_ids.device

    # --- per layer-window interventions ------------------------------------
    windows: List[Dict] = []
    window_token_delta: List[List[float]] = []
    block_mask = build_block_mask(
        block_type, seq_len, image_idx, question_idx, pred_positions, gen_tokens, device
    )
    is_baseline = block_type == "full_attention"

    for start in range(num_layers):
        end = min(start + k, num_layers)
        if is_baseline:
            delta = torch.zeros_like(p_full)
            info_flow = 0.0
            info_flow_logprob = 0.0
        else:
            with AttentionBlocker(layers[start:end], block_mask):
                logits = vlm.forward_logits(full_ids, inputs)
            p_blk, logp_blk = token_probabilities(logits, pred_positions, gen_ids)
            delta = p_full - p_blk
            info_flow = normalized_information_flow(p_full, p_blk)
            info_flow_logprob = normalized_logprob_information_flow(logp_full, logp_blk)

        windows.append(
            {
                "start": start,
                "end": end,
                "info_flow": info_flow,
                "info_flow_logprob": info_flow_logprob,
            }
        )
        window_token_delta.append([float(x) for x in delta.tolist()])

    # --- whole-network intervention (for the word-level analysis) ----------
    if is_baseline:
        token_delta = [0.0] * len(gen_ids)
    else:
        with AttentionBlocker(layers, block_mask):
            logits_all = vlm.forward_logits(full_ids, inputs)
        p_all, _ = token_probabilities(logits_all, pred_positions, gen_ids)
        token_delta = [float(x) for x in (p_full - p_all).tolist()]

    words = group_tokens_into_words(vlm.processor.tokenizer, gen_ids, token_delta)
    word_ids = token_word_boundaries(vlm.processor.tokenizer, gen_ids)
    refs = list(answers) if answers else ([reference] if reference else [])

    # IF curve shape statistics
    if_vals = [w["info_flow"] for w in windows]
    curve_stats = if_curve_stats(if_vals)

    is_closed = closed_ended(reference) if reference else None

    # --- logit-based yes/no scoring for closed questions -------------------
    # Robust alternative to parsing possibly-truncated free text: read P(yes)
    # vs P(no) directly from the first answer-token logits (reuses baseline
    # logits already computed above, so this is free).
    closed_score = None
    if is_closed and pred_positions:
        closed_score = vlm.score_closed(baseline_logits[0, pred_positions[0], :])

    # Correctness: for closed questions trust the logit verdict; otherwise fall
    # back to text matching on the generated answer.
    if is_closed and closed_score is not None and reference is not None:
        correct = closed_score["closed_pred"] == reference.strip().lower()
    else:
        correct = is_correct(generated_text, refs) if refs else None

    return {
        "id": sample_id,
        "task": task,
        "model": vlm.model_id,
        "block_type": block_type,
        "k": k,
        "question": question,
        "question_type": question_type,
        "answer_type": answer_type,
        "image_organ": image_organ,
        "reference": reference,
        "generated": generated_text,
        "correct": correct,
        "exact_match": exact_match(generated_text, refs) if refs else None,
        "closed_ended": is_closed,
        "closed_pred": closed_score["closed_pred"] if closed_score else None,
        "closed_logp_yes": round(closed_score["closed_logp_yes"], 6) if closed_score else None,
        "closed_logp_no": round(closed_score["closed_logp_no"], 6) if closed_score else None,
        "closed_margin": round(closed_score["closed_margin"], 6) if closed_score else None,
        "answer_confidence": round(conf, 6),
        "answer_perplexity": round(perp, 4),
        "first_token_entropy": round(fte, 4) if not (fte != fte) else None,
        "if_auc": curve_stats["auc"],
        "if_peak_share": curve_stats["peak_share"],
        "if_early_share": curve_stats["early_share"],
        "if_late_share": curve_stats["late_share"],
        "if_gini": curve_stats["gini"],
        "prompt_len": prompt_len,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_generated_tokens": len(gen_ids),
        "num_image_tokens": len(image_idx),
        "num_question_tokens": len(question_idx),
        "windows": windows,
        "token_delta": token_delta,
        "window_token_delta": window_token_delta,
        "words": words,
        "word_ids": word_ids,
        "generated_token_ids": gen_ids,
    }


def _empty_result(sample_id, task, question, reference, block_type, k, generated_text, seq_len, prompt_len) -> Dict:
    return {
        "id": sample_id,
        "task": task,
        "block_type": block_type,
        "k": k,
        "question": question,
        "question_type": None,
        "answer_type": None,
        "image_organ": None,
        "reference": reference,
        "generated": generated_text,
        "correct": None,
        "exact_match": None,
        "closed_ended": None,
        "answer_confidence": None,
        "answer_perplexity": None,
        "first_token_entropy": None,
        "if_auc": 0.0,
        "if_peak_share": 0.0,
        "if_early_share": 0.0,
        "if_late_share": 0.0,
        "if_gini": 0.0,
        "prompt_len": prompt_len,
        "seq_len": seq_len,
        "num_layers": 0,
        "num_generated_tokens": 0,
        "num_image_tokens": 0,
        "num_question_tokens": 0,
        "windows": [],
        "token_delta": [],
        "window_token_delta": [],
        "words": [],
        "word_ids": [],
        "generated_token_ids": [],
    }


def aggregate_curve(results: Sequence[Dict]) -> Dict[str, List[float]]:
    """Average the per-layer information-flow curve over a list of samples."""
    curves = [r["windows"] for r in results if r["windows"]]
    if not curves:
        return {"layer": [], "info_flow": [], "info_flow_logprob": [], "std": []}
    n = min(len(c) for c in curves)
    layer = [c[i]["start"] for i in range(n) for c in curves[:1]]
    info = []
    info_logp = []
    std = []

    for i in range(n):
        vals = [c[i]["info_flow"] for c in curves]
        info.append(sum(vals) / len(vals))
        info_logp.append(sum(c[i]["info_flow_logprob"] for c in curves) / len(curves))
        std.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return {"layer": layer, "info_flow": info, "info_flow_logprob": info_logp, "std": std}
