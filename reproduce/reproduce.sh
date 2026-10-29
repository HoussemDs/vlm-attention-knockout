#!/usr/bin/env bash
# Regenerates the statistics, tables and figures of the paper from saved run outputs.
# Run from anywhere; paths are relative to the repository root.
#
# Needs the run folders outputs_gemma_fixed, outputs_medgemma_fixed (discovery, 0000-0099),
# outputs_gemma_confirm, outputs_medgemma_confirm (confirmation, 0100-0249) and
# outputs_gemma_full, outputs_medgemma_full (both splits, 0000-0249). These are not in the
# repository (see README). Install requirements.txt first.
set -euo pipefail
cd "$(dirname "$0")/.."

# Table 1: pre-registered metric family, discovery split (n=100)
python analysis/stats_family.py \
    --base_dir outputs_gemma_fixed --new_dir outputs_medgemma_fixed \
    --out figures_compare_fixed/stats_family.json

# Table 2: confirmation split (n=150)
python analysis/stats_family.py \
    --base_dir outputs_gemma_confirm --new_dir outputs_medgemma_confirm \
    --out figures_compare_fixed/stats_family_confirm.json

# Figure 1: metric family (discovery)
python analysis/fig_metric_family.py

# Figure 4: confirmation split
python analysis/fig_confirmation.py

# Figures 2 and 3: per-layer curves on the full 250 samples (pixel-identical to the paper)
python compare_models.py \
    --old outputs_gemma_full --new outputs_medgemma_full \
    --output_dir figures_regenerated

# Raw numeric export: per-layer information flow and per-sample metrics (no text)
python reproduce/export_results.py \
    --runs outputs_gemma_full outputs_medgemma_full --out results
