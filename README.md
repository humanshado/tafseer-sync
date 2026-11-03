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
  transcribe_omni.py  alternative Hausa pass with Meta's Omnilingual ASR
  compare_models.sh / compare_asr.py  run the Hausa models on the same excerpt and compare side by side
  suggest_corrections.py  optional: N-ATLaS suggests word-spacing and spelling fixes for the review tool
  transcribe.py    Whisper large-v3 (faster-whisper), Arabic pass used to find Quran recitation
  build.py         merges the passes, detects Quran recitation, tags Arabic phrases, flags low confidence
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
work/<id>/         raw speech-recognition output (for re-building), review backups, suggestions.json
```

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install faster-whisper rapidfuzz torch transformers
```
The first run downloads the models from huggingface.co (Whisper large-v3 about 3 GB, the Hausa model about
2.3 GB, Omnilingual ASR about 3.9 GB).

Why three passes: Whisper is poor at Hausa (it produced mostly gibberish on this lecture), while the
dedicated Hausa model gives usable text but cannot write Arabic. So the Hausa model transcribes the
commentary, and Whisper's Arabic output is used to find which verses are recited. The third pass, Meta's
Omnilingual ASR (CTC v2 via `aadel4/omniASR-CTC-1B-v2`, a community conversion stated to match Meta's
outputs), writes recited verses far more accurately than Whisper but switches to Arabic letters for about a
third of the Hausa commentary. `build.py` therefore combines them:
- **Recitation:** searched for in both Whisper's and Omnilingual's Arabic. Where both find the same passage
  it becomes one, with all verses either heard; its start and end come from Omnilingual, whose word timings
  are cut at pauses, while Whisper's segments often run on into the commentary.
- **Hausa:** each segment keeps the Hausa model's text unless Omnilingual wrote the same stretch in Latin
  letters with at least the same confidence. The other model's text is kept in `alt`, and the review tool
  shows it under the text box ("Other") with a button to swap it in.

On lecture 1 the combined build found 41 recitation passages where Whisper alone found 31 (all 31 kept, most
with tighter times; the new ones are mostly verses quoted during the explanation), and used Omnilingual's text
for 216 of 689 Hausa segments. For example the Fatiha at 11:05–11:34, where Whisper alone gave 11:07–11:46 and
hid the commentary after it. The isti'adha ("a'udhu billahi mina sh-shaytani r-rajim") is not matched as a
verse, and where the passes disagree on a surah, one the lecture covers (`FOCUS`) wins. The Omnilingual pass
takes about 25 minutes per hour of audio on a 4-core CPU. Skip it with `OMNI_MODEL=` (empty); `build.py` then
works as before.
Two other Hausa models can be compared with the current one on a 2-minute excerpt (writes
`work/compare/<id>-<start>/comparison.md`, one column per model):
- `NCAIR1/Hausa-ASR` (Nigeria's National Centre for AI and Robotics), a Whisper fine-tune. It is gated:
  accept its terms on Hugging Face and set a read token in `HF_TOKEN`.
- Meta's Omnilingual ASR (Apache 2.0, 1,600+ languages). It needs Python 3.10–3.12 with PyTorch 2.8, so give
  it its own environment and point `OMNI_PYTHON` at it; without it, the comparison skips this model.
  `OMNI_MODEL` picks the model (default `omniASR_LLM_1B`; the `_v2` models are newer). If Meta's download
  server is unreachable, `OMNI_HF` loads the same checkpoint from Hugging Face (v1 models only).
  The newer CTC v2 model also exists as a community conversion to the standard transformers format
  (`aadel4/omniASR-CTC-1B-v2`, stated to match Meta's outputs). It runs in the main environment with word
  timings, no separate setup needed:
  `.venv/bin/python pipeline/transcribe_ctc.py player/audio/ID.mp3 OUT_DIR --model aadel4/omniASR-CTC-1B-v2`.
  This is the model `run_lecture.sh` uses for its third pass (see above).
```bash
python3.12 -m venv .venv-omni && .venv-omni/bin/pip install omnilingual-asr faster-whisper
PYTHON=.venv/bin/python OMNI_PYTHON=.venv-omni/bin/python pipeline/compare_models.sh fatihah01_baqara05 600 120
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

## Optional: spelling and spacing suggestions (N-ATLaS)
The Hausa model often runs words together ("waɗansuharuffa") or misspells them ("akwi").
`suggest_corrections.py` asks N-ATLaS, NCAIR's Hausa/Yoruba/Igbo/English language model, for a tidied
version of each unreviewed Hausa segment. It never changes the transcript. It writes
`work/<id>/suggestions.json`, and the review tool shows each suggestion under the segment as a word diff
with **Use suggestion** (Alt+U, press again to restore the original) and **Dismiss**.
A language model can "improve" what the scholar said or invent words where the audio was unclear, so a
suggestion is thrown away unless it keeps essentially the same letters (only spacing and a few letters may
change). Still listen before accepting one.
```bash
.venv/bin/pip install llama-cpp-python
.venv/bin/python pipeline/suggest_corrections.py player/data/fatihah01_baqara05.json --from 600 --to 720
```
By default it runs a 4-bit GGUF copy of N-ATLaS (`QuantFactory/N-ATLaS-GGUF`, 4.9 GB download, about 6 GB of
RAM), which works on an ordinary CPU but slowly: about 40 s per segment on 4 cores, so a whole lecture
(about 700 segments) takes 7–8 hours. Run it overnight or on part of a lecture; an interrupted run continues
where it stopped. With a GPU, `--hf NCAIR1/N-ATLaS` uses the official weights
instead (gated like Hausa-ASR: accept the terms and set `HF_TOKEN`). `--flagged-only` limits it to segments
flagged as uncertain.

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
