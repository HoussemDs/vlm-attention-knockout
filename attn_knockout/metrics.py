"""the probe metrics: numerically stabilised information flow and word aggregation."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import math

# Subword markers used by the byte-level BPE / SentencePiece tokenizers.
_SPACE_MARKERS = ("\u0120", "\u2581")  # "Ġ" and "▁"


def token_probabilities(
    logits: torch.Tensor, pred_positions: Sequence[int], target_ids: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Probability and log-probability of each target token.

    ``logits`` has shape (1, S, vocab). ``pred_positions[i]`` is the query
    position whose next-token distribution predicts ``target_ids[i]``.
    """
    rows = logits[0, torch.as_tensor(list(pred_positions), device=logits.device), :].float()
    logp = torch.log_softmax(rows, dim=-1)
    tgt = torch.as_tensor(list(target_ids), device=logits.device).view(-1)
    idx = torch.arange(tgt.numel(), device=logits.device)
    logp_t = logp[idx, tgt]
    return torch.exp(logp_t), logp_t


def normalized_information_flow(p_full: torch.Tensor, p_blocked: torch.Tensor) -> float:
    """Length-normalised change in output probability (the the probe estimator).

    Averaging over the number of generated tokens keeps the measure bounded and
    comparable across free-form answers of different lengths.
    """
    return float((p_full - p_blocked).mean())


def normalized_logprob_information_flow(logp_full: torch.Tensor, logp_blocked: torch.Tensor) -> float:
    """Log-probability variant of the estimator (more sensitive to low-prob tokens)."""
    return float((logp_full - logp_blocked).mean())


def token_word_boundaries(tokenizer, token_ids: Sequence[int], special_ids: set | None = None) -> List[int]:
    """Return, for every token, the index of the word it belongs to (-1 = special).

    A new word starts whenever the subword carries a leading space marker (or on
    the very first token), mirroring how the paper normalises subwords before
    computing per-word information flow. Special tokens (EOS, image markers...)
    map to -1 and are ignored by the word-level aggregation.
    """
    if special_ids is None:
        special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    word_ids: List[int] = []
    current = -1
    for position, tid in enumerate(token_ids):
        tid = int(tid)
        if tid in special_ids:
            word_ids.append(-1)
            continue
        token = tokenizer.convert_ids_to_tokens(tid)
        piece = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        starts_word = (
            position == 0
            or current == -1
            or (isinstance(token, str) and token.startswith(_SPACE_MARKERS))
            or piece.startswith(" ")
        )
        if starts_word:
            current += 1
        word_ids.append(current)
    return word_ids


def group_tokens_into_words(tokenizer, token_ids: Sequence[int], deltas: Sequence[float]) -> List[Dict]:
    """Aggregate token-level information flow into word-level measures."""
    word_ids = token_word_boundaries(tokenizer, token_ids)
    words: List[Dict] = []
    for wid, tid, delta in zip(word_ids, token_ids, deltas):
        if wid < 0:  # special token: not a word
            continue
        piece = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
        if wid == len(words):
            words.append(
                {"word": piece.strip(), "delta": float(delta), "n_subwords": 1, "token_ids": [int(tid)]}
            )
        else:
            words[wid]["word"] = (words[wid]["word"] + piece).strip()
            words[wid]["delta"] += float(delta)
            words[wid]["n_subwords"] += 1
            words[wid]["token_ids"].append(int(tid))
    return words


def is_correct(generated: str, answers: Sequence[str]) -> bool:
    """Loose matching used to split samples into correct / incorrect buckets."""
    gen = (generated or "").lower().strip()
    if not gen:
        return False
    return any(a.lower().strip() and a.lower().strip() in gen for a in answers)


def image_entry_layer(info_flow: Sequence[float], frac: float = 0.5) -> int | None:
    """First layer whose information flow reaches `frac` of the curve's peak.

    CANONICAL USE: call this PER SAMPLE (on each sample's own curve) and take the
    median across samples. That per-sample median is the reported "image entry
    layer" and the one that is significance-tested (analysis/stats.py,
    analysis/stats_family.py). Applying it to a mean/aggregate curve gives a
    DIFFERENT number (a property of the averaged curve, with no distribution) --
    report that only under an explicit *_aggcurve name, never as "the entry layer".
    """
    peak = max(info_flow)
    if peak <= 0:
        return None
    thr = frac * peak
    for layer, v in enumerate(info_flow):
        if v >= thr:
            return layer
    return None


def question_dominance_ratio(question_flow: Sequence[float], image_flow: Sequence[float], eps: float = 1e-8) -> float:
    """Mean non-negative question-to-last flow divided by mean non-negative image-to-last flow."""
    q = sum(max(v, 0.0) for v in question_flow) / len(question_flow)
    i = sum(max(v, 0.0) for v in image_flow) / len(image_flow)
    return q / (i + eps)


# ---------------------------------------------------------------------------
# Additional per-sample metrics (computed from existing forward passes)
# ---------------------------------------------------------------------------

def exact_match(generated: str, answers: Sequence[str]) -> bool:
    """Strict exact match after lower-casing and stripping."""
    gen = (generated or "").lower().strip()
    return any(gen == a.lower().strip() for a in answers if a)


def closed_ended(answer: str) -> bool:
    """True when the ground-truth answer is a yes/no (closed-ended) question."""
    return answer.lower().strip() in {"yes", "no"}


def answer_confidence(logp_tokens: torch.Tensor) -> float:
    """Mean log-probability of generated tokens. Higher is more confident."""
    return float(logp_tokens.mean())


def answer_perplexity(logp_tokens: torch.Tensor) -> float:
    """Per-token perplexity. Lower means the model is more certain."""
    return float(torch.exp(-logp_tokens.mean()))


def first_token_entropy(logits: torch.Tensor, position: int) -> float:
    """Shannon entropy (nats) of the next-token distribution at `position`.

    High entropy = the model is uncertain about the first word of its answer.
    """
    logp = torch.log_softmax(logits[0, position, :].float(), dim=-1)
    p = torch.exp(logp)
    return float(-(p * logp).nansum())


def if_curve_stats(info_flow: Sequence[float]) -> Dict:
    """Shape statistics of an information-flow curve.

    Returns:
        auc          – mean over all layers (same sign as raw values)
        peak_share   – peak layer's share of total non-negative IF (concentration)
        early_share  – fraction of non-negative IF in the first third of layers
        late_share   – fraction of non-negative IF in the last third of layers
        gini         – Gini coefficient of non-negative IF (0 = uniform, 1 = single layer)
    """
    n = len(info_flow)
    if n == 0:
        return {"auc": 0.0, "peak_share": 0.0, "early_share": 0.0, "late_share": 0.0, "gini": 0.0}

    pos = [max(v, 0.0) for v in info_flow]
    total = sum(pos) or 1e-8
    third = max(n // 3, 1)

    sorted_pos = sorted(pos)
    gini_num = sum((2 * i - n - 1) * v for i, v in enumerate(sorted_pos, 1))
    gini = gini_num / (n * total)

    return {
        "auc": round(sum(info_flow) / n, 6),
        "peak_share": round(max(pos) / total, 4),
        "early_share": round(sum(pos[:third]) / total, 4),
        "late_share": round(sum(pos[n - third:]) / total, 4),
        "gini": round(gini, 4),
    }
