# Tafsir synced-transcript player

A Hausa tafsir lecture plays while its transcript scrolls along with it. Quran verses appear in
official Arabic text (Hafs or Warsh) with Abubakar Gumi's Hausa translation, and common Arabic
expressions are shown in Arabic script.

## Layout
```
pipeline/
  run_lecture.sh   one command per lecture: clean audio -> transcribe -> build JSON
  transcribe_ctc.py Hausa commentary: CLEAR-Global w2v-bert-2.0 Hausa model (CC BY-SA 4.0), word timings
  transcribe_whisper_hf.py  alternative Hausa pass with a Hugging Face Whisper fine-tune (default NCAIR1/Hausa-ASR)
  compare_ncair.sh / compare_asr.py  run two Hausa models on the same excerpt and compare side by side
  transcribe.py    Whisper large-v3 (faster-whisper), Arabic pass used to find Quran recitation
  build.py         merges both passes, detects Quran recitation, tags Arabic phrases, flags low confidence
  glossary.json    Arabic expressions to detect in the Hausa text (add to it as you review)
  formulas.json    set Arabic openings (e.g. Khutbat al-Hajah) shown as "Opening praise", not as Quran verses
  quran_data/      Uthmani + Warsh text, Hausa translation (Gumi), surah names
  models/cb.rnnn   noise-reduction model used for audio cleanup
review/
  server.py        local review server: serves the tool, validates and saves edits, keeps backups
  index.html       the review tool
player/
  index.html       the player (no build step, no dependencies)
  serve.py         local preview server (supports seeking)
  audio/<id>.mp3   cleaned audio
  data/<id>.json   transcript per lecture
work/<id>/         raw speech-recognition output (for re-building) and review backups
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
gated: accept its terms on Hugging Face and set a read token in `HF_TOKEN`. To compare it with the
current model on a 2-minute excerpt (writes `work/compare/<id>-<start>/comparison.md`):
```bash
PYTHON=.venv/bin/python pipeline/compare_ncair.sh fatihah01_baqara05 600 120
```

## Process a lecture
```bash
PYTHON=.venv/bin/python pipeline/run_lecture.sh ORIGINAL.mp3 fatihah01_baqara05 "Tafsiri 1: Fatiha – Baƙara 5"
```
For a quick quality test on 2 minutes, add `--start 600 --duration 120`.
Re-run only the build step (for example after editing glossary.json or formulas.json).
Segments already marked reviewed in the existing transcript are kept; everything else is regenerated
(add `--discard-reviewed` to start over):
```bash
.venv/bin/python pipeline/build.py work/fatihah01_baqara05 fatihah01_baqara05 audio/fatihah01_baqara05.mp3 player/data/fatihah01_baqara05.json --title "..." --focus 1,2
```

## Preview
```bash
cd player && python3 serve.py
```
Then open http://localhost:8000/?lecture=fatihah01_baqara05

## Review and correct a transcript
```bash
python3 review/server.py
```
Then open http://localhost:8001 (plain Python is enough; no virtualenv needed).

- The list starts with the segments not yet reviewed. **Alt+P** plays the selected segment.
  Correct the text, then **Ctrl+Enter** marks it reviewed and plays the next one.
- Buttons insert the Hausa letters ɓ ɗ ƙ ƴ.
- Change a segment's type to **Quran verse** and enter references such as `2:3-5`. The verse text always
  comes from the official data, and a reference that doesn't exist is rejected.
- Add the Arabic expressions you hear. A new one can also go into `pipeline/glossary.json`, so
  future lectures detect it automatically.
- Nudge start and end times, or set them from the playhead. **Split**, **Merge with next**, **Insert after**,
  **Delete** and **Undo** handle segmentation mistakes. Press **?** for all keyboard shortcuts.
- It saves automatically (and with **Ctrl+S**). The previous 30 versions are kept in `work/<id>/backups/`.
- Commit `player/data/<id>.json` after a review session so the work is on GitHub.

Run the review tool only on your own computer: it can write files and has no login.

## Transcript format (player/data/<id>.json)
- `segments[]`: `{id, start, end, type, ...}`, where `type` is one of:
  - `speech`: Hausa `text`, plus `phrases[]` (detected Arabic expressions)
  - `quran`: `refs` such as `["2:2", "2:3"]`, plus `heard` (what speech recognition heard, kept for review)
  - every segment: `reviewed: true` once a person has checked it
  - `arabic`: other Arabic `text`, with an optional `label` (e.g. "Opening praise"); `formula` marks a draft from formulas.json
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
