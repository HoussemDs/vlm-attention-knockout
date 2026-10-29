"""Re-evaluate accuracy from already-saved 'generated' text. No model rerun needed.

Loads results JSONs from outputs_gemma/ and outputs_medgemma/, applies a smarter
yes/no extraction regex for closed-ended samples, then writes patched copies to
outputs_gemma_fixed/ and outputs_medgemma_fixed/ with corrected 'correct' fields
and updated 'run_summary' accuracy stats.

Usage:
    python reeval_accuracy.py
    python reeval_accuracy.py --old outputs_gemma --new outputs_medgemma
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_clean_answer(generated: str, closed: bool) -> str:
    """Pull a short canonical answer out of verbose generated text."""
    g = (generated or "").lower().strip()
    if not g:
        return ""

    if closed:
        # First standalone yes/no wins.
        m = re.search(r'\b(yes|no)\b', g)
        return m.group(1) if m else g[:20]

    # Open-ended: try to catch a short answer after common preambles.
    patterns = [
        r'(?:the answer is|diagnosis is|finding is)[:\s]+([a-z][a-z\s,\-]+?)[\.\n,;]',
        r'(?:shows?|reveals?|demonstrates?|indicates?|appears? to be)[:\s]+([a-z][a-z\s,\-]+?)[\.\n,;]',
        r'(?:consistent with|compatible with)[:\s]+([a-z][a-z\s,\-]+?)[\.\n,;]',
        r'(?:yes|no)[,\.\s]+(?:there is|it is|this is|i see)[:\s]+([a-z][a-z\s,\-]+?)[\.\n,;]',
    ]
    for pat in patterns:
        m = re.search(pat, g)
        if m:
            candidate = m.group(1).strip()
            if 2 <= len(candidate.split()) <= 6:
                return candidate
    # Fallback: first meaningful word (skip "the", "a", "an", "based", "okay")
    skip = {"the", "a", "an", "based", "okay", "let", "i", "in", "this", "there"}
    for word in g.split():
        w = re.sub(r'[^\w]', '', word)
        if w and w not in skip:
            return w
    return g.split()[0] if g.split() else g


def is_correct_extracted(extracted: str, references: Sequence[str]) -> bool:
    """Check extracted answer against reference answers.

    Whole-word / exact matching only. The earlier `r in ext` substring rule
    produced false positives ("no" matched "normal"/"nodule", "ap" matched
    "appears"), inflating accuracy; it is removed here.
    """
    ext = extracted.lower().strip()
    if not ext:
        return False
    for ref in references:
        r = ref.lower().strip()
        if not r:
            continue
        if r == ext:
            return True
        # whole-word containment (word boundaries), both directions
        if re.search(rf'\b{re.escape(r)}\b', ext):
            return True
        if re.search(rf'\b{re.escape(ext)}\b', r):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-file patching
# ---------------------------------------------------------------------------

def patch_json(src: Path, dst_dir: Path) -> dict:
    """Load a results JSON, re-evaluate accuracy, write patched copy, return stats."""
    data = json.loads(src.read_text(encoding="utf-8"))
    results = data.get("results", [])

    n_total = n_correct_old = n_correct_new = 0
    n_closed = n_closed_correct = 0
    n_open = n_open_correct = 0
    conf_correct = []
    conf_incorrect = []

    for r in results:
        generated = r.get("generated", "") or ""
        reference = r.get("reference", "") or ""
        answers = [reference] if isinstance(reference, str) else list(reference)
        closed = r.get("closed_ended", False)

        old_correct = bool(r.get("correct"))
        extracted = extract_clean_answer(generated, closed)
        if closed and r.get("closed_pred") is not None:
            # logit yes/no verdict is primary: text is truncated at max_new_tokens
            new_correct = str(r["closed_pred"]).lower() == str(reference).strip().lower()
        else:
            new_correct = None  # no logit verdict (open-ended, or no logit score): not evaluated

        if not closed:
            new_correct = None  # open-ended: not evaluated (generation truncated)
        r["correct"] = new_correct
        r["correct_source"] = ("not_evaluated" if not closed
                               else "logit" if r.get("closed_pred") is not None else "text")
        r["extracted_answer"] = extracted  # keep for audit

        if closed:
            n_total += 1
            n_closed += 1
            if old_correct:
                n_correct_old += 1
            if new_correct:
                n_correct_new += 1
                n_closed_correct += 1
        else:
            n_open += 1

        conf = r.get("answer_confidence")
        if conf is not None:
            (conf_correct if new_correct else conf_incorrect).append(conf)

    # Rebuild run_summary
    acc = n_closed_correct / n_closed if n_closed else None  # headline = closed yes/no only
    data["run_summary"] = {
        "accuracy": round(acc, 4) if acc is not None else None,
        "exact_match": round(acc, 4) if acc is not None else None,
        "accuracy_closed": round(acc, 4) if acc is not None else None,
        "accuracy_open": None,  # not evaluated
        "n_closed": n_closed,
        "n_open": n_open,
        "mean_answer_confidence": None,
        "mean_conf_correct": round(sum(conf_correct) / len(conf_correct), 4) if conf_correct else None,
        "mean_conf_incorrect": round(sum(conf_incorrect) / len(conf_incorrect), 4) if conf_incorrect else None,
    }

    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / src.name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "file": src.name,
        "n": n_total,
        "old_acc": round(n_correct_old / n_total, 4) if n_total else 0,
        "new_acc": round(n_correct_new / n_total, 4) if n_total else 0,
        "acc_closed": round(n_closed_correct / n_closed, 4) if n_closed else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate accuracy without rerunning the model.")
    parser.add_argument("--old", default="outputs_gemma", help="Gemma results directory")
    parser.add_argument("--new", default="outputs_medgemma", help="MedGemma results directory")
    parser.add_argument("--old_fixed", default="outputs_gemma_fixed")
    parser.add_argument("--new_fixed", default="outputs_medgemma_fixed")
    args = parser.parse_args()

    pairs = [
        (Path(args.old), Path(args.old_fixed), "Gemma-3-4B"),
        (Path(args.new), Path(args.new_fixed), "MedGemma-4B"),
    ]

    all_stats = {}
    for src_dir, dst_dir, label in pairs:
        jsons = sorted(src_dir.glob("results_*.json"))
        if not jsons:
            print(f"  [SKIP] {src_dir}: no results_*.json files found")
            continue
        print(f"\n=== {label} ({src_dir}) ===")
        for jf in jsons:
            stats = patch_json(jf, dst_dir)
            delta = stats["new_acc"] - stats["old_acc"]
            print(f"  {stats['file']}")
            print(f"    accuracy: {stats['old_acc']:.1%} -> {stats['new_acc']:.1%}  (Δ {delta:+.1%})")
            print(f"    closed (yes/no, n={stats['n']}): {stats['acc_closed']:.1%}  |  open: not evaluated")
        all_stats[label] = stats

    print("\nPatched JSONs written to:", args.old_fixed, "and", args.new_fixed)
    print("Run compare_models.py with --old", args.old_fixed, "--new", args.new_fixed, "--output_dir figures_compare_fixed")


if __name__ == "__main__":
    main()
