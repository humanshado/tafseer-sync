#!/usr/bin/env bash
# Process one lecture end to end:  clean audio -> transcribe (Hausa + Arabic) -> build player JSON.
# Set FOCUS=1,2 (surahs the lecture covers) to resolve phrases that occur in several surahs.
# A third pass with Meta's Omnilingual ASR (CTC v2, transformers conversion) finds recitation more
# accurately and often writes clearer Hausa; build.py combines it with the others. OMNI_MODEL picks
# the model; set it empty (OMNI_MODEL= ) to skip this pass.
# Usage: pipeline/run_lecture.sh ORIGINAL.mp3 LECTURE_ID "Title" [extra transcribe.py args, e.g. --start 600 --duration 120]
set -euo pipefail
SRC="$1"; ID="$2"; TITLE="$3"; shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
MODELS="$ROOT/pipeline/models"
CLEAN="$ROOT/player/audio/$ID.mp3"
mkdir -p "$ROOT/work/$ID" "$ROOT/player/audio" "$ROOT/player/data"

if [ ! -f "$CLEAN" ]; then
  echo "== Cleaning audio"
  ffmpeg -hide_banner -loglevel error -y -i "$SRC" -af "pan=mono|c0=0.5*c0+0.5*c1,aresample=48000:resampler=soxr,\
highpass=f=80:p=2,highpass=f=80,arnndn=m=$MODELS/cb.rnnn,lowpass=f=4000,\
equalizer=f=250:t=q:w=1:g=-2,equalizer=f=2500:t=q:w=1.5:g=2,loudnorm=I=-16:TP=-1.5:LRA=9" \
    -ar 44100 -c:a libmp3lame -q:a 4 "$CLEAN"
fi

echo "== Transcribing Hausa commentary (CLEAR-Global w2v-bert Hausa model)"
"$PY" "$ROOT/pipeline/transcribe_ctc.py" "$CLEAN" "$ROOT/work/$ID" "$@"
OMNI_MODEL="${OMNI_MODEL-aadel4/omniASR-CTC-1B-v2}"
if [ -n "$OMNI_MODEL" ]; then
  echo "== Transcribing again with Omnilingual ASR ($OMNI_MODEL): recitation and a second Hausa reading"
  "$PY" "$ROOT/pipeline/transcribe_ctc.py" "$CLEAN" "$ROOT/work/$ID" --model "$OMNI_MODEL" --name raw_omni.json "$@"
fi
echo "== Transcribing Arabic (Whisper large-v3; the slow step) to find Quran recitation"
"$PY" "$ROOT/pipeline/transcribe.py" "$CLEAN" "$ROOT/work/$ID" --langs ar "$@"

DPY="${DIARIZE_PYTHON:-$PY}"
if "$DPY" -c "import speechbrain, sklearn" 2>/dev/null; then
  echo "== Telling the reciter's voice from the lecturer's (for the explanation of each verse)"
  "$DPY" "$ROOT/pipeline/diarize.py" "$CLEAN" "$ROOT/work/$ID"
else
  echo "== Skipping speaker detection: speechbrain is not installed for $DPY (set DIARIZE_PYTHON)"
fi

echo "== Building transcript"
"$PY" "$ROOT/pipeline/build.py" "$ROOT/work/$ID" "$ID" "audio/$ID.mp3" "$ROOT/player/data/$ID.json" --title "$TITLE" --focus "${FOCUS:-}"
if [ -f "$ROOT/work/$ID/speakers.json" ]; then
  echo "== Cutting the explanation of each verse into its own file"
  "$PY" "$ROOT/pipeline/export_explanations.py" "$ROOT/player/data/$ID.json"
fi
echo "Done. Preview: cd $ROOT/player && python3 serve.py, then open http://localhost:8000/?lecture=$ID"
