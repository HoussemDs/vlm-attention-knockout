# Running on Kaggle

The GPU runs use Kaggle kernels. Each kernel runs on two T4 GPUs, one shard per GPU, and
merges its outputs. Kaggle sessions are capped at 12 hours.

## Kernels

| Kernel | What it runs | Sample range |
|---|---|---|
| `fgtracer-gemma` | base Gemma-3-4B tracing | discovery 0000-0099 (v7), confirmation 0100-0249 (v9) |
| `fgtracer-medgemma` | MedGemma-4B tracing | discovery 0000-0099 (v13), confirmation 0100-0249 (v15) |

Each confirmation kernel runs 2 shards of 75 samples. Each discovery kernel runs 2 shards of 50. The full 250-sample outputs are the merge of both splits.

The notebooks, logs and numeric exports of these exact versions are in `kaggle/runs/`. See `kaggle/runs/README.md`.

Kaggle versions (each link opens that exact version):

| Kernel | Version | Link |
|---|---|---|
| `fgtracer-gemma` | v7 (discovery) | https://www.kaggle.com/code/houssemeddineds/fgtracer-gemma?scriptVersionId=361544970 |
| `fgtracer-gemma` | v9 (confirmation) | https://www.kaggle.com/code/houssemeddineds/fgtracer-gemma?scriptVersionId=362045713 |
| `fgtracer-medgemma` | v13 (discovery) | https://www.kaggle.com/code/houssemeddineds/fgtracer-medgemma?scriptVersionId=361769710 |
| `fgtracer-medgemma` | v15 (confirmation) | https://www.kaggle.com/code/houssemeddineds/fgtracer-medgemma?scriptVersionId=361869138 |
| `fgtracer-encoder` | v9 | https://www.kaggle.com/code/houssemeddineds/fgtracer-encoder |

## Attached inputs

Each kernel's `kernel-metadata.json` lists its inputs:
- `fgtracer-code`: the project code, uploaded as a private Kaggle dataset. `run_kaggle.sh`
  builds the zip from the repository on each run (the file list is in the script), so the
  zip itself is not stored in the repository.
- `fgtracer-hf-token`: a private dataset holding a Hugging Face token in `hf_token.txt`.
  The token is read from the file and is never written into the notebook. This dataset
  was used for the runs listed above. It has since been deleted and the token revoked.
  New runs must read `HF_TOKEN` from Kaggle Secrets instead.
- The models are downloaded from Hugging Face. Accept the Gemma and MedGemma licences on
  Hugging Face first.

Optional: a Kaggle Secret named `WANDB_API_KEY` enables Weights & Biases logging. Without
it, the run still logs to MLflow and to the raw JSON outputs.

## Running

1. Authenticate the Kaggle CLI with your own account. Do not commit the key file.
2. Upload the code dataset and the token dataset, then push a kernel:
   ```bash
   kaggle kernels push -p kaggle/fgtracer-gemma
   ```
3. Wait for the kernel to finish, then download its output:
   ```bash
   kaggle kernels output <your-username>/fgtracer-gemma -p <scratch-dir>
   ```
4. Merge the shards and re-evaluate accuracy locally with `merge_outputs.py` and
   `reeval_accuracy.py`. See `SETUP.md`.

`kaggle/run_kaggle.sh` does the same steps in one command.

## Memory

A 4B model in bfloat16 takes about 14.4 GB, which fills a 16 GB T4. Load one model per GPU.
Do not load two full models on one GPU.
