# Kaggle runs behind the paper

This folder holds the Kaggle notebooks, kernel metadata and run logs for the GPU traces used
in the paper, plus numeric exports of their per-sample results. Anyone can open the
notebooks and logs to see how each number was produced.

## Which version is which

| Folder | Kaggle kernel and version | Model | Sample range | Used for |
|---|---|---|---|---|
| `fgtracer-gemma-v7` | `fgtracer-gemma` v7 | Gemma-3-4B | 0000-0099 (2 shards of 50) | Discovery |
| `fgtracer-medgemma-v13` | `fgtracer-medgemma` v13 | MedGemma-4B | 0000-0099 (2 shards of 50) | Discovery |
| `fgtracer-gemma-v9` | `fgtracer-gemma` v9 | Gemma-3-4B | 0100-0249 (2 shards of 75) | Confirmation |
| `fgtracer-medgemma-v15` | `fgtracer-medgemma` v15 | MedGemma-4B | 0100-0249 (2 shards of 75) | Confirmation |

Each folder contains:
- `fgtracer-<model>.ipynb`: the notebook as it ran, with its cell outputs.
- `kernel-metadata.json`: the Kaggle kernel settings (GPU on, attached inputs).
- `fgtracer-<model>.log`: the full log of that run.

`fgtracer-encoder/` is the encoder comparison kernel (latest version at download, 2026-10-26).
Its `encoder_probe.json` and `xai_layerwise_cka.json` outputs match the committed copies in
`figures_compare_fixed/xai/` exactly.

The 50-sample smoke test that gave the idea for the QDR metric was run before these versions
and is not included here.

## Numeric exports

- `export-disc/`: per-sample metrics for the discovery split, from v7 and v13.
- `export-conf/`: per-sample metrics for the confirmation split, from v9 and v15.

Each export has `per_sample/`, `per_layer_if/` and `curves/`. They contain only numbers and
the sample id. They have no question, reference answer or generated text. They are made by
`reproduce/export_results.py`.

The raw run JSON files are not in the repository. They contain generated text and VQA-RAD
content (see the README). They are needed to rerun the statistics from scratch, so reproduce
them on Kaggle with the notebooks above.

## Reproducing the statistics

The statistics in `figures_compare_fixed/stats_family.json` and
`figures_compare_fixed/stats_family_confirm.json` were checked against these runs. Rerunning
`analysis/stats_family.py` on the v7/v13 pair and on the v9/v15 pair gives the same medians,
Mann-Whitney p-values and Holm-adjusted p-values, to the last digit.

## Running it yourself

1. Accept the Gemma and MedGemma licences on Hugging Face.
2. Create a Kaggle account and a Hugging Face token, then upload the token as a private
   Kaggle dataset named `fgtracer-hf-token` (file `hf_token.txt`).
3. Open the notebook in Kaggle, attach the code dataset and the token dataset, and run all.
   The notebook reads the token from the attached file. It never stores the token in the
   notebook.
4. Download the run output and merge with `merge_outputs.py`, then run
   `analysis/stats_family.py`. See `SETUP.md`.

Kernel wall time from the logs, on two T4 GPUs: the Gemma discovery run (v7, 100 samples) took 5.1 h and the MedGemma discovery run (v13, 100 samples) took 5.6 h. The Gemma confirmation run (v9, 150 samples) and the MedGemma confirmation run (v15, 150 samples) each took 7.6 h. These times include model loading.
