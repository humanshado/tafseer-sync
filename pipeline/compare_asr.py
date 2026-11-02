"""Compare two Hausa transcriptions of the same audio side by side.

Usage: python compare_asr.py A/raw_ha.json B/raw_ha.json OUT.md [--window 15]
Groups both transcripts into fixed time windows so the columns line up.
"""
import argparse
import json
import math


def load(p):
    d = json.load(open(p))
    return d.get("model", p), d["segments"], d.get("review_below_logprob", -1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--window", type=float, default=15)
    x = ap.parse_args()
    (na, sa, ta), (nb, sb, tb) = load(x.a), load(x.b)
    t0 = min(s["start"] for s in sa + sb)
    t1 = max(s["end"] for s in sa + sb)

    def cell(segs, thr, lo, hi):
        parts = [("⚠ " if s["avg_logprob"] < thr else "") + s["text"].replace("|", "/")
                 for s in segs if lo <= (s["start"] + s["end"]) / 2 < hi]
        return " ".join(parts) or "—"

    def stats(segs, thr):
        words = sum(len(s["text"].split()) for s in segs)
        unsure = sum(1 for s in segs if s["avg_logprob"] < thr)
        return f"{len(segs)} segments, {words} words, {unsure} marked uncertain (⚠)"

    lines = [f"# Hausa transcription comparison ({t0:.0f}–{t1:.0f} s)", "",
             f"- **A:** `{na}`: {stats(sa, ta)}", f"- **B:** `{nb}`: {stats(sb, tb)}", "",
             "| Time | A | B |", "|---|---|---|"]
    n = math.ceil((t1 - t0) / x.window)
    for i in range(n):
        lo, hi = t0 + i * x.window, t0 + (i + 1) * x.window
        m, sec = divmod(int(lo), 60)
        lines.append(f"| {m}:{sec:02d} | {cell(sa, ta, lo, hi)} | {cell(sb, tb, lo, hi)} |")
    open(x.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
