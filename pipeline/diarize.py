"""Tell the voices apart: label every speech chunk of a lecture with a speaker number.

A tafsir lecture has two voices: a reciter who recites a bloc of verses, and the lecturer who then
explains them one by one. build.py uses the labels to find the reciter's blocs and so the verse
each explanation belongs to (see explanations.py).

Each chunk of the Omnilingual pass (raw_omni.json, else the Hausa pass) gets a voice print
(SpeechBrain ECAPA, speechbrain/spkrec-ecapa-voxceleb, Apache 2.0); the prints are clustered into
--speakers groups. Chunks under a second are too short for a reliable print and are left out.
Which group is the reciter is decided later, from where the Quran recitation is.

Needs `pip install speechbrain scikit-learn` (and torchaudio matching your PyTorch).
Usage:
  python diarize.py AUDIO.mp3 WORK_DIR [--speakers 2]      writes WORK_DIR/speakers.json
"""
import argparse
import json
import os
import time

import numpy as np

from transcribe import cut

SR = 16000
MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("work_dir")
    ap.add_argument("--speakers", type=int, default=2)
    a = ap.parse_args()

    import torch
    from sklearn.cluster import AgglomerativeClustering
    from speechbrain.inference.speaker import EncoderClassifier

    src = next(p for p in ("raw_omni.json", "raw_ha.json") if os.path.exists(os.path.join(a.work_dir, p)))
    chunks = json.load(open(os.path.join(a.work_dir, src)))["segments"]
    audio = cut(a.audio, 0, 0)
    torch.set_num_threads(os.cpu_count() or 4)
    enc = EncoderClassifier.from_hparams(MODEL, run_opts={"device": "cpu"},
                                         savedir=os.path.join(os.path.expanduser("~"), ".cache", "spkrec-ecapa"))
    t0, prints, used = time.time(), [], []
    for c in chunks:
        x = audio[int(c["start"] * SR):int(c["end"] * SR)]
        if len(x) < SR:
            continue
        with torch.inference_mode():
            e = enc.encode_batch(torch.from_numpy(x.copy())[None])[0, 0].numpy()
        prints.append(e / np.linalg.norm(e))
        used.append(c)
    labels = AgglomerativeClustering(n_clusters=a.speakers, metric="cosine",
                                     linkage="average").fit_predict(np.stack(prints))
    out = [{"start": c["start"], "end": c["end"], "speaker": int(k)} for c, k in zip(used, labels)]
    path = os.path.join(a.work_dir, "speakers.json")
    json.dump({"model": MODEL, "source": src, "segments": out}, open(path, "w"), indent=1)
    mins = [round(sum(o["end"] - o["start"] for o in out if o["speaker"] == k) / 60, 1) for k in range(a.speakers)]
    print(f"[diarize] {len(out)} chunks, minutes per speaker {mins}, {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
