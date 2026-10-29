#!/usr/bin/env bash
# One-shot Kaggle driver for Attention-knockout probe.
#
# Does everything via the Kaggle CLI so you never download outputs by hand:
#   1. packages the live code as a private Kaggle dataset (fgtracer-code)
#   2. pushes the two kernels (gemma, medgemma) -> two separate 12h sessions
#   3. waits for both to finish
#   4. pulls their outputs back into the project (outputs_gemma/, outputs_medgemma/)
#   5. runs the local compare + stats so results are ready to read
#
# Prereqs (one-time):
#   - Kaggle CLI authenticated (~/.config/kaggle/kaggle.json)  [you said this is set]
#   - Attach the models to each kernel once in the Kaggle UI, OR make sure the
#     model_sources slugs in kernel-metadata.json are correct.
#
# Usage:
#   bash kaggle/run_kaggle.sh            # full run
#   bash kaggle/run_kaggle.sh --push     # just upload code + push kernels
#   bash kaggle/run_kaggle.sh --pull     # just pull outputs + analyse

set -euo pipefail
cd "$(dirname "$0")/.."                       # project root
PROJ="$(pwd)"
KUSER="${KAGGLE_USERNAME:-houssemeddineds}"
GEMMA_SLUG="$KUSER/fgtracer-gemma"
MED_SLUG="$KUSER/fgtracer-medgemma"

upload_code() {
  echo "==> packaging code dataset (single zip, known structure)"
  local d="kaggle/fgtracer-code"
  rm -rf "$d"; mkdir -p "$d"
  zip -qr "$d/attn_knockout_code.zip" attn_knockout analysis run_tracing.py merge_outputs.py \
      compare_models.py reeval_accuracy.py -x '*__pycache__*'
  cat > "$d/dataset-metadata.json" <<JSON
{
  "title": "fgtracer-code",
  "id": "$KUSER/fgtracer-code",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON
  if kaggle datasets status "$KUSER/fgtracer-code" >/dev/null 2>&1; then
    kaggle datasets version -p "$d" -m "update $(date -u +%FT%TZ)"
  else
    kaggle datasets create -p "$d"
  fi
}

push_kernels() {
  echo "==> pushing kernels"
  kaggle kernels push -p kaggle/fgtracer-gemma
  kaggle kernels push -p kaggle/fgtracer-medgemma
  echo "    pushed. Track at https://www.kaggle.com/$GEMMA_SLUG and /$MED_SLUG"
}

wait_for() {
  local slug="$1"
  echo "==> waiting on $slug"
  while true; do
    local st
    st="$(kaggle kernels status "$slug" 2>/dev/null | tr -d '\r' || true)"
    echo "    $st"
    case "$st" in
      *complete*) return 0 ;;
      *error*|*cancel*) echo "!! $slug failed"; return 1 ;;
    esac
    sleep 120
  done
}

pull_outputs() {
  echo "==> pulling outputs into the project"
  local stage; stage="$(mktemp -d)"
  kaggle kernels output "$GEMMA_SLUG" -p "$stage/gemma"
  kaggle kernels output "$MED_SLUG"   -p "$stage/medgemma"
  # kernel output preserves /kaggle/working structure -> copy the outputs_* dirs back
  rsync -a "$stage"/gemma/outputs_gemma/        "$PROJ/outputs_gemma/"        2>/dev/null || \
    cp -r "$stage"/gemma/outputs_gemma "$PROJ/" 2>/dev/null || true
  rsync -a "$stage"/medgemma/outputs_medgemma/  "$PROJ/outputs_medgemma/"     2>/dev/null || \
    cp -r "$stage"/medgemma/outputs_medgemma "$PROJ/" 2>/dev/null || true
  rm -rf "$stage"
  echo "    outputs_gemma/ and outputs_medgemma/ updated"
}

analyse() {
  echo "==> local analysis (no GPU needed)"
  # reeval reads --old/--new (raw) and writes the *_fixed dirs used by the paper
  python reeval_accuracy.py --old outputs_gemma --new outputs_medgemma \
      --old_fixed outputs_gemma_fixed --new_fixed outputs_medgemma_fixed || true
  python compare_models.py --old outputs_gemma_fixed --new outputs_medgemma_fixed \
      --old_label "Gemma-3-4B (base)" --new_label "MedGemma-4B (medical)" \
      --block_type image_to_last --output_dir figures_compare_fixed || true
  python analysis/stats.py --base_dir outputs_gemma_fixed --new_dir outputs_medgemma_fixed \
      --out figures_compare_fixed/stats_summary.json
  echo "    see figures_compare_fixed/ (figures + stats_summary.json)"
}

case "${1:-all}" in
  --push) upload_code; push_kernels ;;
  --pull) pull_outputs; analyse ;;
  all|"")
    upload_code; push_kernels
    wait_for "$GEMMA_SLUG"; wait_for "$MED_SLUG"
    pull_outputs; analyse ;;
  *) echo "usage: $0 [--push|--pull]"; exit 1 ;;
esac
echo "==> done"
