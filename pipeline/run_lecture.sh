#!/usr/bin/env bash
# Process one lecture end to end:  clean audio -> transcribe (Hausa + Arabic) -> build player JSON.
# Set FOCUS=1,2 (surahs the lecture covers) to resolve phrases that occur in several surahs.
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
echo "== Transcribing Arabic (Whisper large-v3; the slow step) to find Quran recitation"
"$PY" "$ROOT/pipeline/transcribe.py" "$CLEAN" "$ROOT/work/$ID" --langs ar "$@"

echo "== Building transcript"
"$PY" "$ROOT/pipeline/build.py" "$ROOT/work/$ID" "$ID" "audio/$ID.mp3" "$ROOT/player/data/$ID.json" --title "$TITLE" --focus "${FOCUS:-}"
echo "Done. Preview: cd $ROOT/player && python3 serve.py, then open http://localhost:8000/?lecture=$ID"
