# Sample manifest

Sample IDs are absolute indices into the VQA-RAD test split (`vqarad_test_NNNN`).
Both models use the same indices. Counts were checked against the output files on
2026-10-26.

## Main study (250 samples)

| Run | Block types | Samples | Index range | Contiguous |
|---|---|---|---|---|
| base Gemma-3-4B (`outputs_gemma_full`) | image_to_last, question_to_last | 250 | 0000-0249 | yes |
| MedGemma-4B (`outputs_medgemma_full`) | image_to_last, question_to_last | 250 | 0000-0249 | yes |

- Discovery split: 0000-0099 (100 samples). Candidate metrics were generated here.
- Confirmation split: 0100-0249 (150 samples). Pre-registered tests were run here.
- Closed-ended questions: 138 of the 250 samples (same for both models).

