# Medical Fine-Tuning Changes Answer Commitment and Information Flow in a Vision-Language Model: A Split-Sample Study on VQA-RAD

Code, analysis and paper source for an attention-knockout study of **Gemma-3-4B** (base)
vs **MedGemma-4B** (medical fine-tune) on the VQA-RAD radiology benchmark.

Per-layer attention knockout blocks image-to-answer or question-to-answer attention one
decoder layer at a time, and measures how much information about the answer flows through
each pathway. The study applies this as a pre-registered before/after measurement on 250
samples. Discovery (0000-0099) generated candidate metrics. Confirmation (0100-0249) tested
them. The paper's headline test is the question-dominance ratio on the confirmation split.
Whether the change lies in the vision encoder or the language decoder is left as follow-up
work.

## Pre-registration

- Written protocol: [`PREREGISTRATION.md`](PREREGISTRATION.md).
- Timeline:
  1. A 50-sample smoke test (pilot) on both models. It measured closed-question accuracy, F1
     and QDR. Accuracy and F1 used a text-extraction scorer that was later found to be
     invalid, so the paper reports only the logit verdict (see `RESULTS_SCENARIOS.md`). The
     pilot's QDR drop was larger than later effects and did not replicate at n=100. QDR was
     then chosen as the primary metric.
  2. The full pipeline was rerun on discovery, 0000-0099 (100 samples per model). This
     produced the candidate metrics.
  3. 2026-10-24: the protocol was written. The Gemma 100-sample trace had been analysed, the
     50-sample MedGemma trace and the smoke-test outputs had been seen, and the MedGemma
     100-sample output had not been analysed.
  4. A launch on 0100-0199 was started and stopped after about three minutes, before the
     model loaded. It produced no output. Confirmation then ran once on 0100-0249 (150
     samples, none used in discovery).
- The timing above rests on the dated protocol header and the amendment log.
- Public timestamp: the paper is archived on Zenodo, DOI [10.5281/zenodo.23766104](https://doi.org/10.5281/zenodo.23766104).

## Splits

Indices refer to the row order of the VQA-RAD `test` split (451 rows). Both models use the
same indices.

| Split | Indices | Samples | Role |
|---|---|---|---|
| Discovery | 0000-0099 | 100 | Candidate metrics |
| Confirmation | 0100-0249 | 150 | Pre-registered tests |
| Total analysed | 0000-0249 | 250 | Main results |
| Reserve | 0250-0450 | - | Not used in this paper |

The 138 closed yes/no questions are the scored set (same for both models). Per-sample
indices are in `SAMPLE_MANIFEST.md`.

## Models, data and software

| Item | Value |
|---|---|
| Base model | `google/gemma-3-4b-it`, licence: Gemma terms |
| Medical model | `google/medgemma-4b-it`, licence: MedGemma terms |
| Dataset | `flaviagiammarino/vqa-rad`, `test` split, licence CC0-1.0 (per Hugging Face metadata) |
| Revisions used locally | gemma `093f9f388b31de276ce2de164bdc2081324b9767`, medgemma `290cda5eeccbee130f987c4ad74a59ae6f196408`, vqa-rad `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9` |
| Base model source for the Kaggle runs | Kaggle's copy of `google/gemma-3/transformers/gemma-3-4b-it/1` (its revision is not recorded in the outputs) |
| Software | `requirements.txt`: `torch>=2.2`, `transformers>=4.50,<5`. The exact versions used on Kaggle were not recorded |
| Hardware | Kaggle free tier, two NVIDIA T4 GPUs per kernel session (12-hour cap). No institutional compute or funding. Single author |

Both models are gated on Hugging Face. Accept each licence on its model page first.

## Probe settings

- Precision: bfloat16, no weight quantization.
- Window: `K=1`, all 34 decoder layers, one window per layer.
- Block types: `image_to_last` and `question_to_last`.
- Decoding: greedy (`do_sample=False`), `max_new_tokens=64`. Generation is deterministic up
  to hardware numerics, so the `--seed` flag of `run_tracing.py` does not affect it.
- Randomness in the statistics: the bootstrap CIs and the permutation checks in
  `analysis/stats.py` use `numpy.random.default_rng(0)` (`--seed` defaults to 0). The
  Mann-Whitney and Holm tests in `analysis/stats_family.py` (Tables 1-2) are deterministic.
  The permutation p-values in the family tables come from `analysis/perm_family.py`
  (10,000 permutations, seed 0, reads the exported CSVs). Its output is in
  `figures_compare_fixed/perm_family_*.json`.
- Hook: `attn_knockout/attention_blocking.py` installs a `forward_pre_hook` on each decoder
  layer's self-attention. It applies the block mask built by `build_block_mask`.
- Accuracy: closed yes/no questions only, scored from the logit verdict (`closed_pred`).

## Reproducing the paper

1. Install dependencies: `pip install -r requirements.txt numpy matplotlib`.
2. Generate the run outputs with `run_tracing.py`. Always pass `--checkpoint_dir`. The
   default in `attn_knockout/vlm.py` is the SmolVLM pilot model, not the paper's models. See
   `SETUP.md` for the commands. The Kaggle kernels in `kaggle/` were used for the GPU runs.
3. Run `bash reproduce/reproduce.sh`. It regenerates Tables 1-2, Figures 1-4, and the
   numeric export in `results/`.

Output mapping:

| Paper item | Script |
|---|---|
| Table 1 (metric family, discovery) | `analysis/stats_family.py` on `outputs_*_fixed` |
| Table 2 (confirmation) | `analysis/stats_family.py` on `outputs_*_confirm` |
| Figure 1 (metric family) | `analysis/fig_metric_family.py` |
| Figure 2 (per-layer curves, both models) | `compare_models.py` on `outputs_*_full` |
| Figure 3 (fine-tuning shift) | `compare_models.py` on `outputs_*_full` |
| Figure 4 (confirmation) | `analysis/fig_confirmation.py` |
| Raw per-layer curves | `reproduce/export_results.py` |

Verification status: Tables 1 and 2 regenerate to the committed JSON values. Figure 1 was regenerated from `figures_compare_fixed/stats_family.json`. Figures 3 and 4 were not regenerated in this pass.
Figure 2 (`fig_model_layer_comparison`) was redrawn with a shared y-axis from the exported
mean curves (`kaggle/runs/`), because `compare_models.py` needs torch. The `sharey=True`
change in `compare_models.py` produces the same layout, but that script has not been rerun
end to end.

## What is in `results/`

- `curves/`: the mean per-layer information-flow curve for each model and block type.
- `per_layer_if/`: information flow for every sample at every layer (250 rows per file).
- `per_sample/`: numeric per-sample metrics. Questions, reference answers and generated
  text are removed, because they are VQA-RAD content.

The raw run JSON files contain the question, reference and generated text, so they are not
included.

## Limitations

- Single author, one benchmark, one model pair. The findings may not transfer to other
  medical fine-tunes or other radiology datasets.
- 250 samples: 100 for discovery and 150 for confirmation. This gives limited power for
  small effects. The primary effect is small (Cliff's delta = 0.18).
- The protocol was written after the Gemma 100-sample trace and the 50-sample MedGemma
  trace had been seen, but before the MedGemma 100-sample output. The primary metric was
  chosen with that knowledge. Only the confirmation split is a clean test of the
  pre-specified claims.
- Accuracy uses closed yes/no questions only. Open-ended answers are not scored, and the
  64-token cap truncates long generations.
- The question-dominance ratio is relative to each model's own language prior. It does not
  measure absolute reliance on the text.
- The encoder-versus-decoder question is unresolved. The encoder change is documented, but
  correlational analyses cannot attribute the routing change to either part.
- Exact software versions used on Kaggle were not recorded.
- Compute: all runs used free Kaggle GPUs (two T4s per run). Open-ended answers were not
  scored: raising `max_new_tokens` from 8 to 64 did not produce scorable answers in our runs.
- Some Weights & Biases logs were not recorded in the cloud. The Kaggle logs and exported
  files are the complete record. The `n_samples` field in the W&B config of the confirmation
  notebooks reads 100; the actual sample ranges are in `kaggle/runs/README.md`.

## Repository layout

| Path | Contents |
|---|---|
| `attn_knockout/` | Probe library: attention blocking, metrics, model loading, tracing |
| `run_tracing.py` | Runs the knockout tracing for one model on one split |
| `merge_outputs.py` | Merges sharded run outputs |
| `reeval_accuracy.py` | Re-evaluates closed yes/no accuracy from saved outputs |
| `compare_models.py` | Base vs medical per-layer comparison and figures |
| `plot_information_flow.py`, `log_to_*.py` | Plotting and experiment-tracker logging |
| `analysis/` | Statistics and figure scripts used in the paper |
| `reproduce/` | Reproduction script and numeric export |
| `results/` | Exported numeric results (no VQA-RAD text) |
| `figures_compare_fixed/` | Statistics JSON used for Tables 1-2 and the encoder analysis |
| `paper/` | LaTeX source of the paper |
| `kaggle/` | Kaggle kernels and launch script used for the GPU runs |
| `requirements.txt` | Python dependencies |
| `PREREGISTRATION.md` | Pre-registered analysis plan |
| `RESULTS_SCENARIOS.md` | Accuracy measure and the earlier invalid numbers |
| `SAMPLE_MANIFEST.md` | Sample indices per run |
| `SETUP.md` | Setup and run commands |
| `LICENSE` | MIT licence for the code |
| `.gitignore` | Excludes run outputs, virtualenvs, caches and the compiled PDF |

## Kaggle runs and timestamps

The GPU runs behind every number in the paper were done on Kaggle. To check them:

- [`kaggle/README.md`](kaggle/README.md): the kernel versions, with links to each Kaggle version.
- [`kaggle/runs/README.md`](kaggle/runs/README.md): which run is which, the sample ranges, and the wall-clock time of each run from its log.
- `kaggle/runs/*/`: the notebook, kernel metadata and full log of each run, with the timestamps.

## Paper

The paper source is in `paper/`. Build it with a standard LaTeX toolchain:

```
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`paper/main.pdf` is git-ignored because it is regenerated from the source.

## Licence

- Code: MIT (see `LICENSE`).
- Paper text and figures: CC BY 4.0.
- VQA-RAD data: not included. Its licence applies to anyone who downloads it.
- The model weights keep the licences listed above.
