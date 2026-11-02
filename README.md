# Tafsir synced-transcript player

A Hausa tafsir lecture plays while its transcript scrolls along with it. Quran verses appear in
official Arabic text (Hafs or Warsh) with Abubakar Gumi's Hausa translation, and common Arabic
expressions are shown in Arabic script.

## Layout
```
pipeline/
  run_lecture.sh   one command per lecture: clean audio -> transcribe -> build JSON
  transcribe_ctc.py Hausa commentary: CLEAR-Global w2v-bert-2.0 Hausa model (CC BY-SA 4.0), word timings
  transcribe.py    Whisper large-v3 (faster-whisper), Arabic pass used to find Quran recitation
  build.py         merges both passes, detects Quran recitation, tags Arabic phrases, flags low confidence
  glossary.json    Arabic expressions to detect in the Hausa text (add to it as you review)
  quran_data/      Uthmani + Warsh text, Hausa translation (Gumi), surah names
  models/cb.rnnn   noise-reduction model used for audio cleanup
player/
  index.html       the player (no build step, no dependencies)
  serve.py         local preview server (supports seeking)
  audio/<id>.mp3   cleaned audio
  data/<id>.json   transcript per lecture
work/<id>/         raw Whisper output (kept for re-building)
```

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install faster-whisper rapidfuzz torch transformers
```
The first run downloads the models from huggingface.co (Whisper large-v3 about 3 GB, the Hausa model about 2.3 GB).

Why two models: Whisper is poor at Hausa (it produced mostly gibberish on this lecture), while the
dedicated Hausa model gives usable text but cannot write Arabic. So the Hausa model transcribes the
commentary, and Whisper's Arabic output is used only to find which verses are recited.
`NCAIR1/Hausa-ASR` (Nigeria's National Centre for AI and Robotics) is another candidate, but it is
gated: it needs a Hugging Face login and acceptance of its terms.

## Process a lecture
```bash
PYTHON=.venv/bin/python pipeline/run_lecture.sh ORIGINAL.mp3 fatihah01_baqara05 "Tafsiri 1: Fatiha – Baƙara 5"
```
For a quick quality test on 2 minutes, add `--start 600 --duration 120`.
Re-run only the build step (for example after editing glossary.json):
```bash
.venv/bin/python pipeline/build.py work/fatihah01_baqara05 fatihah01_baqara05 audio/fatihah01_baqara05.mp3 player/data/fatihah01_baqara05.json --title "..."
```

## Preview
```bash
cd player && python3 serve.py
```
Then open http://localhost:8000/?lecture=fatihah01_baqara05

## Transcript format (player/data/<id>.json)
- `segments[]`: `{id, start, end, type, ...}`, where `type` is one of:
  - `speech`: Hausa `text`, plus `phrases[]` (detected Arabic expressions)
  - `quran`: `refs` such as `["2:2", "2:3"]`, plus `heard` (what speech recognition heard, kept for review)
  - `arabic`: other Arabic `text` (added by hand during review)
- `review: true` marks segments speech recognition was unsure about. Toggle **Review** in the player to see them.
- `verses{}`: the text for each referenced verse (`uthmani`, `warsh`, `hausa`).

## Player features
- Highlights the current segment and scrolls with the audio. Tap any segment to jump there.
- Scrolling by hand pauses auto-scroll; **Back to current** resumes it.
- Hafs/Warsh toggle, Hausa translation on/off, playback speed, light/dark mode.
- Remembers the playback position for each lecture.
- Keyboard: Space plays/pauses; ← and → skip 5 s.

## Sources and licences
- Quran text: Tanzil.net Uthmani (CC BY 3.0, must be used verbatim). Warsh: KFGQPC via Quranpedia.net.
  Hausa translation: Abubakar Mahmoud Gumi, via QuranEnc.com. All collected through
  [quran-json](https://github.com/risan/quran-json) (see `pipeline/quran_data/LICENSE-quran-json.txt`).
- Font: Amiri (SIL Open Font License, `player/fonts/amiri.OFL.txt`).
- Lecture audio: confirm permission from the rights holder before publishing.
