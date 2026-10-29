# Accuracy measure and audit

This note records how accuracy is computed, and the earlier numbers that were
wrong. The paper uses the corrected measure only.

## Measure used in the paper
- Closed-ended yes/no questions only.
- Verdict: the logit score from `run_tracing.py` / `attn_knockout/tracing.py` (`closed_pred`),
  not text extraction from the generated answer.
- Open-ended questions are not scored. Generations are capped at 64 new tokens, so many
  answers are cut off before the answer appears.
- `reeval_accuracy.py` applies this verdict to saved outputs. Pass `--old_fixed` and
  `--new_fixed` explicitly. Without them the script overwrites the discovery folders.

## Earlier numbers that were invalid
An early pipeline scored correctness by text extraction from the generated answer. It
produced accuracy of about 4-10%, and it ignored the logit verdict. Those numbers came
from a 50-sample pilot and from a 100-sample check, and they are not reported anywhere in
the paper. The discrepancy came from the scoring, not from the models.

## Checks
- In the 250-sample runs, all 138 closed questions have a non-null `closed_pred`, for both
  models. Check this after any new run.
- The majority-class baseline matters. On the confirmation split the majority answer is
  "no", so always-"yes" scores 44.3%. The paper reports both the model accuracy and that
  baseline.
