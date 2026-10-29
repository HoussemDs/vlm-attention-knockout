# Analysis pre-registration: attention-knockout study

> Fixed analysis rules: discovery 0000-0099, confirmation 0100-0249 (see Amendment 1), primary test, correction.


Written: 2026-10-24, before the 100-sample MedGemma output was analysed.
Scope: how results will be tested and reported. Any change after the date below must be
logged as a dated amendment, not edited in place.

Honesty note: before writing this, the author had already seen the Gemma 100-sample trace,
the 50-sample MedGemma trace, and smoke-test outputs (QDR, closed accuracy, encoder
divergence). The primary metric below was chosen with that knowledge. The confirmation
set is the only part that has not been seen.

## 1. Question

Does medical fine-tuning (Gemma-3-4B to MedGemma-4B) change how the model routes image
evidence versus question text in its decision layers, and can that change be attributed to
the vision encoder, the decoder, or both?

## 2. Data split (VQA-RAD test split, absolute indices)

| Set | Indices | Use | Status |
|---|---|---|---|
| Discovery | 0000-0099 | All metrics in Section 3 are computed and tested | Complete |
| Confirmation | 0100-0199 → **0100-0249** (see Amendment 1, §10) | Only the primary test and any discovery candidates | Complete |
| Reserve | 0200-0450 → **0250-0450** (see Amendment 1, §10) | Not used unless the confirmation set is inconclusive (then amend this document first). Reserved for future work. | Not run |

Indices are absolute (`vqarad_test_NNNN`). Tracked in `SAMPLE_MANIFEST.md`.

## 3. Metric family (discovery)

These are the metrics in the family. Adding a metric after this date makes it exploratory
(Section 6).

Routing (per model, paired by sample):
1. Question dominance ratio (QDR), per sample, `image_to_last` and `question_to_last`
2. Image entry layer (`image_entry_layer`) and depth
3. Image peak layer
4. Information-flow AUC, `image_to_last`
5. Information-flow Gini, `image_to_last`

Answer quality:
6. Closed yes/no accuracy (logit verdict, n=59 in discovery). Not used for the primary test.
7. Mean answer log-prob, correct vs incorrect (closed questions only)

Encoder:
8. Linear CKA between encoders (layer-0 image features)
9. Mean per-sample encoder divergence (1 − cosine)
10. Jensen-Shannon divergence and KL, first answer token
11. Encoder-control regression intercept (`analysis/encoder_control.py`)

Discovery testing:
- Mann-Whitney U (two-sided), as in `analysis/stats.py`. Permutation p-values are reported alongside.
- Holm correction across the 11 metrics above (report raw and adjusted p).
- A metric is a **discovery candidate** if its Holm-adjusted p < 0.05.

## 4. Primary confirmatory test (pre-specified)

**Primary claim:** the median per-sample QDR is lower for MedGemma than for Gemma
(image_to_last), tested on the confirmation set (0100-0249).

- Test: Mann-Whitney U, two-sided, α = 0.05.
- Effect direction must match discovery (MedGemma lower).
- This is the only result that counts as confirmatory without further correction.

## 5. Discovery candidates

Any discovery candidate (Holm-adjusted p < 0.05 on 0000-0099) is re-tested on the
confirmation set with the same test and direction.

- A candidate is **confirmed** if its raw p < 0.05 on 0100-0249 in the discovery direction.
- Confirmed candidates are then Holm-adjusted across the set of candidates that were
  tested on confirmation.
- A candidate that fails confirmation is reported as "not replicated," not dropped silently.

## 6. Exploratory metrics

Measuring additional metrics during the GPU runs is allowed. They must be:
- logged with their date,
- reported in a separate table labelled "exploratory,"
- not counted in the family, the primary test, or the candidate list,
- tested on the confirmation set before any claim is made about them.

## 7. Reporting rules

- Report every metric in Section 3, with raw and adjusted p-values, whatever the result.
- Report the number of tests run in each family.
- Report sample sizes for each set. The MedGemma discovery set is 100 only after the
  100-sample run is pulled and checked against `SAMPLE_MANIFEST.md`.
- Do not report a result as a finding unless it is confirmed or is the primary test.
- If confirmation is inconclusive, say so. Do not extend the sample until this document is
  amended and dated.

## 8. Decision outcomes (fixed in advance)

| Confirmation result | Claim allowed |
|---|---|
| Primary QDR test significant, same direction | Routing differs between base and medical models (confirmed) |
| Primary test not significant | Routing difference is not confirmed on fresh samples. Report discovery result as exploratory. |
| Candidate metrics confirmed | Report each confirmed metric. Attribution to encoder or decoder remains open and is future work. |
| No candidates confirmed | Report the discovery results as exploratory only. |

## 9. Known limitations (to state in the paper)

- Metrics come from the same traces and are correlated, so the Holm correction is
  conservative in some places and approximate in others.
- The confirmation set is 100 samples. Power is limited: the encoder-control shift is small
  relative to its per-sample spread, and a 100-sample test may not detect it.
- VQA-RAD test items are a small, radiology-only sample.

## 10. Amendments

### Amendment 1, 2026-10-24: confirmation set extended to 0100-0249
- **Change:** the confirmation set is extended from 0100-0199 (100 samples) to
  **0100-0249 (150 samples)**. The reserve set becomes **0250-0450**.
- **Reason:** increase power for the confirmatory tests (the discovery QDR effect is
  small and borderline; closed accuracy at n=59 is underpowered). Logged before the
  confirmation output was pulled or analysed.
- **Scope unchanged:** the primary confirmatory test (QDR, image_to_last, MedGemma
  lower, Mann-Whitney two-sided, alpha=0.05) and the candidate-replication rules
  (section 5) are unchanged; only the sample range they run on is larger.
- **What has NOT been seen:** at the time of this amendment the author has seen only
  the discovery set 0000-0099 (both models). No sample in 0100-0249 has been traced
  or inspected. Both models run the identical indices 0100-0249.

### Amendment 2, written 2026-10-26, recording the timeline (no analysis rule changed)
- **Change:** none to the analysis rules. This amendment records the sequence of work, which
  the header and Amendment 1 only imply.
- **Sequence:**
  1. A 50-sample smoke test on both models gave the idea for the QDR metric.
  2. Discovery on 0000-0099 (100 samples per model) generated the candidate metrics.
  3. 2026-10-24: this protocol was written, after the Gemma 100-sample trace and the 50-sample
     MedGemma trace had been seen, and before the MedGemma 100-sample output was analysed.
  4. Confirmation on 0100-0249 (150 samples, none used in discovery). Amendment 1 extended the
     set from 0100-0199 before any confirmation output was analysed.
- **Why:** an earlier draft of the paper said the protocol was written together with
  the discovery analysis. That was not accurate, and this entry replaces that claim.

### Amendment 3, written 2026-10-26, recording events of 2026-10-23 to 2026-10-24: selection sequence and aborted launch (documentation only)
- **No analysis rule changed.** This amendment records how the work ran.
- **Selection of QDR:** the 50-sample smoke test measured closed-question accuracy, F1 and QDR
  on both models. Accuracy and F1 came from a text-extraction scorer, which was later found to
  be invalid and replaced by the logit verdict (see `RESULTS_SCENARIOS.md`). The pilot showed a
  QDR drop, larger than the drop later seen on discovery and confirmation, and it did not
  replicate at n=100. QDR was then chosen as the primary metric.
- **Aborted launch:** a run on 0100-0199 was started and stopped after about three minutes,
  before the model loaded. It produced no output and no sample from it was analysed.
- **Confirmation:** run once on 0100-0249 (150 samples, none used in discovery).
- **Ranges in sections 2, 4 and 5:** the 0100-0199 ranges there are the originally planned
  confirmation range. Amendment 1 superseded them with 0100-0249. Only 0100-0249 was analysed;
  the aborted 0100-0199 launch produced no output.
- **Header note:** the honesty note above describes what the author had seen by 2026-10-24 and is
  unchanged. This amendment adds the pilot and launch details that the note does not cover.

### Amendment 4, written 2026-10-29: hypotheses added, multiplicity count, and sample size (documentation only)
- **No analysis rule changed.** This amendment records three things that the original text did not state correctly.
- **Directional expectations H1 to H3 are not in the protocol file as it stood on 2026-10-26**, although the paper states them from that date. They are recorded here on 2026-10-29. The repository does not establish when they were first written, so they are not claimed as pre-specified:
  - H1 (Image Entry Layer): fine-tuning shifts image evidence to earlier decoder depths.
  - H2 (Question Dominance): fine-tuning lowers the QDR.
  - H3 (Peak Layer): the peak image-information layer shifts earlier.
- **Multiplicity count (Section 3).** The original text said Holm correction across the 11 metrics. Two of the 11 are exact deterministic transforms of another metric and carry identical statistics: answer perplexity (of mean answer log-prob) and image entry depth (of image entry layer). The correction is therefore applied across the nine distinct quantities. The paper reports the same five discovery candidates under the 11-test count as a robustness note. The nine-quantity count appears in the paper from 2026-10-26, and it is recorded in this file on 2026-10-29.
- **Section 9 sample size.** "The confirmation set is 100 samples" is out of date. The confirmation set is 0100-0249, which is 150 samples (Amendment 1).
