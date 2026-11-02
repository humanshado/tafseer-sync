"""Hausa transcription with a Hugging Face Whisper fine-tune (default: NCAIR1/Hausa-ASR).

Writes OUT_DIR/raw_ha.json in the same format as transcribe_ctc.py, so build.py can use it.
Speech is split at pauses (Silero VAD from faster-whisper) into chunks of at most 25 s,
each chunk is transcribed, and confidence comes from the decoder's token probabilities.

NCAIR1/Hausa-ASR is gated: accept its terms on Hugging Face, then provide a read token
through the HF_TOKEN environment variable (or `huggingface-cli login`).

Usage:
  python transcribe_whisper_hf.py AUDIO.mp3 OUT_DIR [--model ID] [--start S --duration D]
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import torch
from faster_whisper.vad import VadOptions, get_speech_timestamps
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from transcribe import cut

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="NCAIR1/Hausa-ASR")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--beams", type=int, default=4, help="beam search width (>1 also gives a confidence score)")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    os.makedirs(a.out_dir, exist_ok=True)
    audio = cut(a.audio, a.start, a.duration)
    token = os.environ.get("HF_TOKEN")
    proc = WhisperProcessor.from_pretrained(a.model, token=token)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, token=token).eval()

    # Ask for Hausa transcription if the model knows the language tokens; fine-tunes sometimes don't.
    gen = {"num_beams": a.beams, "max_new_tokens": 200, "return_dict_in_generate": True, "output_scores": True}
    try:
        model.generate(proc(np.zeros(SR, np.float32), sampling_rate=SR, return_tensors="pt").input_features,
                       language="hausa", task="transcribe", max_new_tokens=2)
        gen.update(language="hausa", task="transcribe")
    except Exception as e:  # noqa: BLE001
        print(f"(model does not take a language setting: {e.__class__.__name__}; using its defaults)")

    chunks = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300,
                                                     max_speech_duration_s=25, speech_pad_ms=150))
    out, t0 = [], time.time()
    for c in chunks:
        x = audio[c["start"]:c["end"]]
        feats = proc(x, sampling_rate=SR, return_tensors="pt").input_features
        with torch.inference_mode():
            r = model.generate(feats, **gen)
        text = proc.batch_decode(r.sequences, skip_special_tokens=True)[0].strip()
        if not text:
            continue
        # beam search reports a length-normalised log-probability for the chosen transcript
        ss = r.get("sequences_scores")
        avg = float(ss[0]) if ss is not None else -1.0
        # Whisper models can get stuck repeating themselves ("napapapa..."); text that compresses
        # this well is a loop, not speech, so mark it as very uncertain.
        if len(text) > 20 and len(text) / len(zlib.compress(text.encode())) > 2.4:
            avg = min(avg, -5.0)
        base = c["start"] / SR + a.start
        out.append({"start": round(base, 2), "end": round(base + len(x) / SR, 2), "text": text,
                    "avg_logprob": round(avg, 3), "no_speech_prob": 0.0, "words": []})
        print(f"[ha-whisper] {base:8.1f}  {text[:100]}", flush=True)

    path = os.path.join(a.out_dir, "raw_ha.json")
    json.dump({"language": "ha", "model": a.model, "review_below_logprob": -0.6, "segments": out},
              open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[ha-whisper] {len(out)} segments in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
