#!/usr/bin/env bash
# Run the Hausa models on the same excerpt and compare them side by side:
#   current  CLEAR-Global w2v-bert-2.0 Hausa (transcribe_ctc.py)
#   ncair    NCAIR1/Hausa-ASR, Whisper fine-tune (transcribe_whisper_hf.py; needs HF_TOKEN, gated model)
#   omni     Meta Omnilingual ASR (transcribe_omni.py; skipped if omnilingual_asr is not installed)
# Omnilingual ASR needs Python 3.10-3.12 with PyTorch 2.8, so it may live in its own environment:
# OMNI_PYTHON points at that interpreter (default: PYTHON). OMNI_MODEL picks the model card and
# OMNI_HF a Hugging Face copy of it (for when Meta's download server is unreachable).
# Usage: pipeline/compare_models.sh [LECTURE_ID] [START_SECONDS] [DURATION_SECONDS]
set -euo pipefail
ID="${1:-fatihah01_baqara05}"; START="${2:-600}"; DUR="${3:-120}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PY="${PYTHON:-python3}"; OPY="${OMNI_PYTHON:-$PY}"
AUDIO="$ROOT/player/audio/$ID.mp3"; OUT="$ROOT/work/compare/$ID-$START"
mkdir -p "$OUT/current" "$OUT/ncair"
"$PY" "$ROOT/pipeline/transcribe_ctc.py" "$AUDIO" "$OUT/current" --start "$START" --duration "$DUR"
"$PY" "$ROOT/pipeline/transcribe_whisper_hf.py" "$AUDIO" "$OUT/ncair" --start "$START" --duration "$DUR"
RUNS=("$OUT/current/raw_ha.json" "$OUT/ncair/raw_ha.json")
if "$OPY" -c "import omnilingual_asr" 2>/dev/null; then
  mkdir -p "$OUT/omni"
  "$OPY" "$ROOT/pipeline/transcribe_omni.py" "$AUDIO" "$OUT/omni" --start "$START" --duration "$DUR" \
    ${OMNI_MODEL:+--model "$OMNI_MODEL"} ${OMNI_HF:+--hf "$OMNI_HF"}
  RUNS+=("$OUT/omni/raw_ha.json")
else
  echo "Skipping Omnilingual ASR: omnilingual_asr is not installed for $OPY (set OMNI_PYTHON)." >&2
fi
"$PY" "$ROOT/pipeline/compare_asr.py" "${RUNS[@]}" "$OUT/comparison.md" > /dev/null
echo "Saved: $OUT/comparison.md"
