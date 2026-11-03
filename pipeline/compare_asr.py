"""Compare Hausa transcriptions of the same audio side by side.

Usage: python compare_asr.py A/raw_ha.json B/raw_ha.json [C/raw_ha.json ...] OUT.md [--window 15]
Groups every transcript into fixed time windows so the columns line up.
"""
import argparse
import json
import math


def load(p):
    d = json.load(open(p))
    return d.get("model", p), d["segments"], d.get("review_below_logprob", -1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="two or more raw_ha.json files, then the output .md")
    ap.add_argument("--window", type=float, default=15)
    x = ap.parse_args()
    if len(x.paths) < 3:
        ap.error("give at least two transcripts and the output file")
    *inputs, out = x.paths
    runs = [load(p) for p in inputs]
    names = [chr(ord("A") + i) for i in range(len(runs))]
    every = [s for _, segs, _ in runs for s in segs]
    t0 = min(s["start"] for s in every)
    t1 = max(s["end"] for s in every)

    def cell(segs, thr, lo, hi):
        parts = [("⚠ " if s["avg_logprob"] < thr else "") + s["text"].replace("|", "/")
                 for s in segs if lo <= (s["start"] + s["end"]) / 2 < hi]
        return " ".join(parts) or "—"

    def stats(segs, thr):
        words = sum(len(s["text"].split()) for s in segs)
        unsure = sum(1 for s in segs if s["avg_logprob"] < thr)
        return f"{len(segs)} segments, {words} words, {unsure} marked uncertain (⚠)"

    lines = [f"# Hausa transcription comparison ({t0:.0f}–{t1:.0f} s)", ""]
    lines += [f"- **{n}:** `{model}`: {stats(segs, thr)}" for n, (model, segs, thr) in zip(names, runs)]
    lines += ["", "| Time | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for i in range(math.ceil((t1 - t0) / x.window)):
        lo, hi = t0 + i * x.window, t0 + (i + 1) * x.window
        m, sec = divmod(int(lo), 60)
        lines.append(f"| {m}:{sec:02d} | " + " | ".join(cell(segs, thr, lo, hi) for _, segs, thr in runs) + " |")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
