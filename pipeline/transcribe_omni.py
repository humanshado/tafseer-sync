"""Hausa transcription with Meta's Omnilingual ASR (default: omniASR_LLM_1B, Apache 2.0).

Writes OUT_DIR/raw_ha.json in the same format as transcribe_ctc.py, so compare_asr.py and
build.py can use it. Speech is split at pauses (Silero VAD from faster-whisper) into chunks of at
most 30 s (the models accept up to 40 s) and each chunk is transcribed with the language set to
Hausa (hau_Latn). Segment times come from the chunks; there are no word timings.

Confidence: for the CTC models (omniASR_CTC_*) it is the mean probability of the emitted
characters, as in transcribe_ctc.py. The LLM models (omniASR_LLM_*) give no score through
Meta's pipeline, so their segments are never marked uncertain.

Needs `pip install omnilingual-asr` (Python 3.10-3.12, PyTorch 2.8; see its README for the CPU-only
variant of fairseq2). Model cards download from Meta's server (dl.fbaipublicfiles.com). Where that
is blocked, --hf takes the same checkpoint from Hugging Face instead (the v1 models are mirrored
there, e.g. --model omniASR_CTC_1B --hf facebook/omniASR-CTC-1B).

Usage:
  python transcribe_omni.py AUDIO.mp3 OUT_DIR [--model CARD] [--hf REPO] [--start S --duration D]
"""
import argparse
import json
import os
import tempfile
import time

import numpy as np

from transcribe import cut

SR = 16000


def use_hf_copy(card, repo):
    """Register a fairseq2 asset card that loads `card` from a Hugging Face copy of its files.

    Must run before omnilingual_asr is imported: fairseq2 reads FAIRSEQ2_USER_ASSET_DIR then."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo)
    ckpt = hf_hub_download(repo, next(f for f in files if f.endswith(".pt")))
    tok = hf_hub_download(repo, next(f for f in files if f.endswith("tokenizer.model")))
    d = tempfile.mkdtemp(prefix="omniasr-cards-")
    with open(os.path.join(d, "hf.yaml"), "w") as f:
        f.write(f"name: {card}_hf_tokenizer\ntokenizer_family: char_tokenizer\ntokenizer: file://{tok}\n"
                f"---\nname: {card}_hf\nbase: {card}\ncheckpoint: file://{ckpt}\n"
                f"tokenizer_ref: {card}_hf_tokenizer\n")
    os.environ["FAIRSEQ2_USER_ASSET_DIR"] = d
    return f"{card}_hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="omniASR_LLM_1B", help="model card, e.g. omniASR_CTC_1B, omniASR_LLM_1B_v2")
    ap.add_argument("--hf", help="Hugging Face repo with the same checkpoint, e.g. facebook/omniASR-LLM-1B")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--max-chunk", type=float, default=30.0, help="longest chunk in seconds (model limit 40)")
    ap.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto",
                    help="auto: bfloat16 on a GPU, float32 on a CPU (faster there, but needs twice the RAM; "
                         "the LLM_1B model needs bfloat16 below about 20 GB)")
    a = ap.parse_args()

    card = use_hf_copy(a.model, a.hf) if a.hf else a.model
    import torch
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    torch.set_num_threads(os.cpu_count() or 4)
    cuda = torch.cuda.is_available()
    os.makedirs(a.out_dir, exist_ok=True)
    audio = cut(a.audio, a.start, a.duration)
    dtype = a.dtype if a.dtype != "auto" else ("bfloat16" if cuda else "float32")
    pipe = ASRInferencePipeline(model_card=card, device="cuda" if cuda else "cpu", dtype=getattr(torch, dtype))
    is_ctc = "CTC" in a.model.upper()
    if is_ctc:  # keep the CTC output so a confidence can be computed (the CTC blank is <s>, i.e. bos)
        logits = []
        pipe.model.register_forward_hook(lambda m, i, o: logits.append(o[0][0].float()))
        blank = pipe.tokenizer.vocab_info.bos_idx

    chunks = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300,
                                                     max_speech_duration_s=a.max_chunk, speech_pad_ms=150))
    out, t0 = [], time.time()
    for c in chunks:
        x = audio[c["start"]:c["end"]]
        if is_ctc:
            logits.clear()
        text = pipe.transcribe([{"waveform": x, "sample_rate": SR}], lang=["hau_Latn"], batch_size=1)[0].strip()
        if not text:
            continue
        avg = 0.0
        if is_ctc and logits:
            probs = logits[0].softmax(-1)
            p, ids = probs.max(-1)
            keep = (ids != blank) & torch.cat([torch.tensor([True]), ids[1:] != ids[:-1]])
            if keep.any():  # build.py expects Whisper-style confidence: log of mean char probability
                avg = float(np.log(max(float(p[keep].mean()), 1e-6)))
        base = c["start"] / SR + a.start
        out.append({"start": round(base, 2), "end": round(base + len(x) / SR, 2), "text": text,
                    "avg_logprob": round(avg, 3), "no_speech_prob": 0.0, "words": []})
        print(f"[ha-omni] {base:8.1f}  {text[:100]}", flush=True)

    path = os.path.join(a.out_dir, "raw_ha.json")
    json.dump({"language": "ha", "model": a.model + (f" ({a.hf})" if a.hf else ""),
               "review_below_logprob": round(float(np.log(0.8)), 3), "segments": out},
              open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[ha-omni] {len(out)} segments in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
