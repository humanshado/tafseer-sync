#!/usr/bin/env bash
# Run NCAIR's Hausa-ASR and the current Hausa model on the same excerpt and compare them.
# Needs HF_TOKEN (read token, with NCAIR1/Hausa-ASR terms accepted on Hugging Face).
# Usage: pipeline/compare_ncair.sh [LECTURE_ID] [START_SECONDS] [DURATION_SECONDS]
set -euo pipefail
ID="${1:-fatihah01_baqara05}"; START="${2:-600}"; DUR="${3:-120}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PY="${PYTHON:-python3}"
AUDIO="$ROOT/player/audio/$ID.mp3"; OUT="$ROOT/work/compare/$ID-$START"
mkdir -p "$OUT/current" "$OUT/ncair"
"$PY" "$ROOT/pipeline/transcribe_ctc.py" "$AUDIO" "$OUT/current" --start "$START" --duration "$DUR"
"$PY" "$ROOT/pipeline/transcribe_whisper_hf.py" "$AUDIO" "$OUT/ncair" --start "$START" --duration "$DUR"
"$PY" "$ROOT/pipeline/compare_asr.py" "$OUT/current/raw_ha.json" "$OUT/ncair/raw_ha.json" "$OUT/comparison.md"
echo "Saved: $OUT/comparison.md"
