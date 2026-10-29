# Statistics outputs

| File | Written by | Split | What it holds |
|---|---|---|---|
| `stats_family.json` | `analysis/stats_family.py` | discovery, 0000–0099 (n=100) | Medians, Mann–Whitney p (raw), Holm p across the nine distinct quantities |
| `stats_family_confirm.json` | `analysis/stats_family.py` | confirmation, 0100–0249 (n=150) | Same fields as above, on the confirmation split |
| `perm_family_discovery.json` | `analysis/perm_family.py` | discovery | Permutation p (10,000 permutations, seed 0) |
| `perm_family_confirmation.json` | `analysis/perm_family.py` | confirmation | Same, confirmation split |
| `xai/*.json`, `xai/*.pdf`, `xai/*.png` | the encoder and XAI scripts | both | Encoder CKA and related outputs |

## Holm correction: which family

The two `stats_family*.json` files correct across all nine distinct quantities. The paper's
confirmation table (`paper/sections/results.tex`) corrects across the five discovery
candidates only, as `analysis/fig_confirmation.py` does. So the `p_holm` values in
`stats_family_confirm.json` are larger than the values in the paper's confirmation table.
The paper's confirmation Holm values are reproduced by hand from the five raw p-values in
the JSON: for example the log-prob value 3.0e-14 × 5 = 1.5e-13.

## Not in these files

- Question dominance ratio and image entry layer permutation p-values. They need the per-layer
  flow data, so they come from `analysis/stats.py` on the raw run JSON, which is not in the
  repository. The Kaggle runs in `kaggle/runs/` produce that JSON.
