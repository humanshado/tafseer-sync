"""Step 1: speech-to-text with timestamps.

Runs Whisper (faster-whisper) twice over the same audio:
  * a Hausa pass   -> the commentary (main timeline)
  * an Arabic pass -> used only to find Quran recitation and Arabic phrases

Usage:
  python transcribe.py AUDIO.mp3 OUT_DIR [--model large-v3] [--langs ha,ar] [--start S --duration D]

Writes OUT_DIR/raw_<lang>.json for each language.
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

from faster_whisper import WhisperModel

# Domain words help Whisper spell recurring names and terms.
PROMPTS = {
    "ha": ("Tafsirin Alkur'ani mai girma da harshen Hausa. Suratul Fatiha, Suratul Baƙara. "
           "Allah Subhanahu wa Ta'ala, Annabi sallallahu alaihi wa sallam, ayoyi, malamai."),
    "ar": "بسم الله الرحمن الرحيم. الحمد لله رب العالمين. سورة الفاتحة. سورة البقرة.",
}


def cut(audio, start, duration):
    """Decode (optionally an excerpt, for quick tests) to 16 kHz mono float32 samples."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", audio, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--langs", default="ha,ar")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    wav = cut(a.audio, a.start, a.duration)
    model = WhisperModel(a.model, device="auto", compute_type="int8", cpu_threads=a.threads)

    for lang in a.langs.split(","):
        t0 = time.time()
        segments, info = model.transcribe(
            wav,
            language=lang,
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,  # avoids runaway repetition loops
            initial_prompt=PROMPTS.get(lang),
        )
        out = []
        for s in segments:
            out.append({
                "start": round(s.start + a.start, 2),
                "end": round(s.end + a.start, 2),
                "text": s.text.strip(),
                "avg_logprob": round(s.avg_logprob, 3),
                "no_speech_prob": round(s.no_speech_prob, 3),
                "words": [{"w": w.word, "s": round(w.start + a.start, 2),
                           "e": round(w.end + a.start, 2), "p": round(w.probability, 3)}
                          for w in (s.words or [])],
            })
            print(f"[{lang}] {s.start + a.start:8.1f}  {s.text.strip()[:90]}", flush=True)
        path = os.path.join(a.out_dir, f"raw_{lang}.json")
        json.dump({"language": lang, "model": a.model, "segments": out},
                  open(path, "w"), ensure_ascii=False, indent=1)
        print(f"[{lang}] {len(out)} segments in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
