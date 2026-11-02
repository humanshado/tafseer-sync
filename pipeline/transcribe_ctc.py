"""Hausa transcription with a CTC model (default: CLEAR-Global w2v-bert-2.0 Hausa, CC BY-SA 4.0).

Writes OUT_DIR/raw_ha.json in the same format as transcribe.py, so build.py can use it.
Speech is split at pauses (Silero VAD from faster-whisper), each chunk is transcribed,
and word timings come from the CTC frame alignment.

Usage:
  python transcribe_ctc.py AUDIO.mp3 OUT_DIR [--model ID] [--start S --duration D]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from faster_whisper.vad import VadOptions, get_speech_timestamps
from transformers import AutoModelForCTC, AutoProcessor

from transcribe import cut

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="CLEAR-Global/w2v-bert-2.0-clearglobal-hausa-asr-1.0.0")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--max-chunk", type=float, default=20.0, help="longest chunk in seconds")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    os.makedirs(a.out_dir, exist_ok=True)
    audio = cut(a.audio, a.start, a.duration)
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForCTC.from_pretrained(a.model).eval()

    chunks = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300,
                                                     max_speech_duration_s=a.max_chunk, speech_pad_ms=150))
    blank = proc.tokenizer.pad_token_id
    delim = proc.tokenizer.word_delimiter_token
    out, t0 = [], time.time()
    for c in chunks:
        x = audio[c["start"]:c["end"]]
        feats = proc(x, sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**feats).logits[0]
        probs = logits.softmax(-1)
        ids = probs.argmax(-1).tolist()
        sec_per_frame = len(x) / SR / len(ids)
        base = c["start"] / SR + a.start

        # Greedy CTC decode, keeping frame positions to time each word.
        words, cur, cur_s, cur_p, prev = [], "", None, [], None
        for f, i in enumerate(ids):
            if i != prev and i != blank:
                ch = proc.tokenizer.convert_ids_to_tokens(i)
                if ch == delim or ch == " ":
                    if cur:
                        words.append((cur, cur_s, f, float(np.mean(cur_p))))
                    cur, cur_s, cur_p = "", None, []
                else:
                    if not cur:
                        cur_s = f
                    cur += ch
                    cur_p.append(float(probs[f, i]))
            prev = i
        if cur:
            words.append((cur, cur_s, len(ids), float(np.mean(cur_p))))
        if not words:
            continue
        conf = float(np.mean([w[3] for w in words]))
        out.append({
            "start": round(base, 2), "end": round(base + len(x) / SR, 2),
            "text": " ".join(w[0] for w in words),
            # build.py expects Whisper-style confidence; map mean char probability onto it
            "avg_logprob": round(float(np.log(max(conf, 1e-6))), 3), "no_speech_prob": 0.0,
            "words": [{"w": w, "s": round(base + s * sec_per_frame, 2), "e": round(base + e * sec_per_frame, 2),
                       "p": round(p, 3)} for w, s, e, p in words],
        })
        print(f"[ha-ctc] {base:8.1f}  {out[-1]['text'][:100]}", flush=True)

    path = os.path.join(a.out_dir, "raw_ha.json")
    json.dump({"language": "ha", "model": a.model, "review_below_logprob": round(float(np.log(0.8)), 3),
               "segments": out}, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[ha-ctc] {len(out)} segments in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
