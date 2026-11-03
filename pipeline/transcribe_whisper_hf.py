"""Hausa transcription with a Hugging Face Whisper fine-tune (default: NCAIR1/Hausa-ASR).

Writes OUT_DIR/raw_ha.json in the same format as transcribe_ctc.py, so build.py can use it.
Speech is split at pauses (Silero VAD from faster-whisper) into chunks of at most 25 s,
each chunk is transcribed, and confidence comes from the decoder's token probabilities.

Repetition guard: Whisper fine-tunes sometimes get stuck repeating a syllable or word
("kafafafa...", "ƴan ƴan ƴan ..."). When a chunk's text looks like such a loop, the chunk is
decoded again with stronger settings (blocking repeated 4-token sequences, then sampling at
rising temperatures), as OpenAI's Whisper does. If every attempt still loops, the repeated
run is collapsed to one copy and the segment is marked uncertain.

NCAIR1/Hausa-ASR is gated: accept its terms on Hugging Face, then provide a read token
through the HF_TOKEN environment variable (or `huggingface-cli login`).

Usage:
  python transcribe_whisper_hf.py AUDIO.mp3 OUT_DIR [--model ID] [--start S --duration D]
"""
import argparse
import json
import os
import re
import time
import zlib

import numpy as np
import torch
from faster_whisper.vad import VadOptions, get_speech_timestamps
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from transcribe import cut

SR = 16000
LOOP_RATIO = 2.4  # text that compresses better than this (zlib) is a loop, not speech
# Decoding settings tried in turn while the output still loops; the first is plain beam search.
FALLBACKS = [
    {},
    {"no_repeat_ngram_size": 4, "repetition_penalty": 1.2},
    {"do_sample": True, "num_beams": 1, "temperature": 0.4, "no_repeat_ngram_size": 4},
    {"do_sample": True, "num_beams": 1, "temperature": 0.8, "no_repeat_ngram_size": 4},
]
# A run of 1-30 characters repeated 4 or more times in a row (optionally space-separated).
REPEAT_RUN = re.compile(r"(\S.{0,29}?)(?:\s*\1){3,}", re.S)


def is_loop(text):
    return len(text) > 20 and len(text) / len(zlib.compress(text.encode())) > LOOP_RATIO


def collapse_repeats(text):
    return re.sub(r"\s{2,}", " ", REPEAT_RUN.sub(r"\1", text)).strip()


def decode(model, proc, feats, gen):
    """Transcribe one chunk; returns (text, mean log-probability per token)."""
    with torch.inference_mode():
        r = model.generate(feats, **gen)
    text = proc.batch_decode(r.sequences, skip_special_tokens=True)[0].strip()
    ss = r.get("sequences_scores")
    if ss is not None:  # beam search reports a length-normalised log-probability directly
        return text, float(ss[0])
    steps = model.compute_transition_scores(r.sequences, r.scores, normalize_logits=True)[0]
    steps = steps[torch.isfinite(steps)]
    return text, float(steps.mean()) if len(steps) else -1.0


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
    torch.manual_seed(0)  # the sampling fallbacks should give the same result on every run

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
        for attempt, extra in enumerate(FALLBACKS):
            text, avg = decode(model, proc, feats, {**gen, **extra})
            if not is_loop(text):
                break
        looped = is_loop(text)
        if looped:  # every attempt looped: keep one copy of the repeated run, flag as very uncertain
            text, avg = collapse_repeats(text), min(avg, -5.0)
        if not text:
            continue
        if attempt or looped:
            print(f"[ha-whisper] repetition guard at {c['start'] / SR + a.start:.1f}s: "
                  f"{'still looping, collapsed' if looped else f'clean on retry {attempt}'}", flush=True)
        base = c["start"] / SR + a.start
        out.append({"start": round(base, 2), "end": round(base + len(x) / SR, 2), "text": text,
                    "avg_logprob": round(avg, 3), "no_speech_prob": 0.0, "words": [],
                    "decode_retries": attempt, "loop_collapsed": looped})
        print(f"[ha-whisper] {base:8.1f}  {text[:100]}", flush=True)

    path = os.path.join(a.out_dir, "raw_ha.json")
    json.dump({"language": "ha", "model": a.model, "review_below_logprob": -0.6, "segments": out},
              open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[ha-whisper] {len(out)} segments in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
