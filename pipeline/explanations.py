"""Find the lecturer's explanation of each verse.

A lecture goes in rounds: the reciter recites a bloc of verses, then the lecturer takes them one at
a time, repeating each verse and explaining it (quoting other verses as he goes) until the reciter
recites the next bloc. So:

1. The reciter is the voice (from speakers.json) whose speech lies most inside Quran recitation.
   His stretches of speech, with the verses recited in them, are the blocs.
2. After each bloc the lecturer repeats its verses in order. For every verse the candidates are
   places where either pass wrote its opening words in Arabic (or the Hausa pass wrote them in Latin
   letters, compared by consonants), and the verses' positions must increase. A quick run through
   three or more of the bloc's verses in a row (as when he quotes a hadith that lists them) is a
   quotation, not his repeats.
3. A verse's explanation runs from its repeat to the next verse's repeat; the first verse's starts
   when the bloc ends, the last runs until the next bloc. A verse whose repeat was not found is
   listed as missing rather than guessed; its explanation is then inside the previous one.
"""
import re
from collections import Counter

from rapidfuzz import fuzz

from build import match_quran, norm_ar, verse_text

AR_CONS = str.maketrans({"ب": "b", "ت": "t", "ث": "s", "ج": "j", "ح": "h", "خ": "k", "د": "d", "ذ": "z",
                         "ر": "r", "ز": "z", "س": "s", "ش": "s", "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
                         "غ": "g", "ف": "f", "ق": "k", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h"})
LATIN_SUBS = [("sh", "s"), ("ts", "s"), ("kh", "k"), ("gh", "g"), ("dh", "z"), ("th", "s"), ("q", "k"),
              ("ɓ", "b"), ("ɗ", "d"), ("ƙ", "k"), ("c", "s")]
BASMALA = "بسم الله الرحمن الرحيم "


def cons_ar(t):
    """Consonants of Arabic text as a Hausa speaker would write them: صراط الذين -> srtlzn."""
    t = " ".join(w[2:] if w.startswith("ال") and len(w) > 3 else w for w in norm_ar(t).split())
    return re.sub(r"(.)\1+", r"\1", "".join(c for c in t.translate(AR_CONS) if "a" <= c <= "z"))


def cons_latin(t):
    t = t.lower()
    for a, b in LATIN_SUBS:
        t = t.replace(a, b)
    t = re.sub(r"\b(a|e|i)?l(?=[a-z])", "", t)  # article al-/el-, written or assimilated
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^bdfghjklmnrstz]", "", t))


def verse_head(ref, words=4):
    """Opening words of a verse, as the lecturer starts it (2:1 in the data begins with the basmala)."""
    t = verse_text[ref]
    if ref[0] != 1 and t.startswith(BASMALA):
        t = t[len(BASMALA):]
    return " ".join(t.split()[:words])


def score_arabic(text, ref):
    x = norm_ar(text)
    if len(x.split()) < 2 and len(verse_head(ref).split()) > 1:
        return 0
    head, full = verse_head(ref), verse_text[ref]
    if len(head.split()) <= 2:  # very short verse ("الم", "الرحمن الرحيم"): must appear word for word
        return 100 if re.search(rf"(^| ){re.escape(head)}( |$)", x) else 0
    return max(fuzz.partial_ratio(head, x) if len(x) >= len(head) else fuzz.partial_ratio(x, head),
               fuzz.partial_ratio(x, full) if len(x.split()) >= 3 and len(x) < len(full) else 0)


def score_latin(text, ref):
    v, t = cons_ar(verse_head(ref)), cons_latin(text)
    if len(v) < 7 or len(t) < len(v):
        return 0
    return fuzz.partial_ratio(v, t)


def reciter_blocs(speakers, quran, gap=8.0, min_len=4.0):
    """(reciter id, [bloc]) where a bloc is {start, end, refs} of one stretch of the reciter's voice."""
    def inside(seg):
        return sum(max(0.0, min(seg["end"], q["end"]) - max(seg["start"], q["start"])) for q in quran)
    share = {}
    for k in {s["speaker"] for s in speakers}:
        mine = [s for s in speakers if s["speaker"] == k]
        share[k] = sum(inside(s) for s in mine) / max(1e-6, sum(s["end"] - s["start"] for s in mine))
    reciter = max(share, key=share.get)
    runs = []
    for s in sorted((s for s in speakers if s["speaker"] == reciter), key=lambda s: s["start"]):
        if runs and s["start"] - runs[-1]["end"] < gap:
            runs[-1]["end"] = s["end"]
        else:
            runs.append({"start": s["start"], "end": s["end"]})
    blocs = []
    for r in runs:
        if r["end"] - r["start"] < min_len:
            continue
        refs = [tuple(map(int, x.split(":"))) for q in quran
                if q["end"] > r["start"] and q["start"] < r["end"] for x in q["refs"]]
        if not refs:
            continue
        surah = Counter(c for c, _ in refs).most_common(1)[0][0]
        ayahs = [v for c, v in refs if c == surah]
        span = [q for q in quran if q["end"] > r["start"] and q["start"] < r["end"]]
        blocs.append({"start": round(min([r["start"]] + [q["start"] for q in span]), 2),
                      "end": round(max([r["end"]] + [q["end"] for q in span]), 2),
                      "refs": [f"{surah}:{v}" for v in range(min(ayahs), max(ayahs) + 1)]})
    return reciter, blocs


def find_repeats(bloc, window_end, arabic, latin, min_score=80, min_latin=85):
    """{ref: (time, heard)} for the lecturer's repeats of the bloc's verses after it."""
    refs = [tuple(map(int, r.split(":"))) for r in bloc["refs"]]
    others = set(refs)
    cands = {r: [] for r in refs}
    for s in arabic + latin:
        if not bloc["end"] <= s["start"] < window_end:
            continue
        is_latin = s in latin
        scores = {r: (score_latin if is_latin else score_arabic)(s["text"], r) for r in refs}
        r = max(scores, key=scores.get)
        if scores[r] < (min_latin if is_latin else min_score):
            continue
        if not is_latin:  # a quote of some other verse that happens to share words is not a repeat
            m = match_quran(s["text"], *_QURAN)
            if m and m[0] >= 90 and not set(m[1]) & others:
                continue
        cands[r].append((s["start"], s["text"]))
    # drop quick runs through the bloc (a hadith or summary listing its verses)
    flat = sorted((t, r) for r, cs in cands.items() for t, _ in cs)
    quoted = set()
    for i in range(len(flat)):
        run = [flat[i]]
        for t, r in flat[i + 1:]:
            if t - run[-1][0] > 15:
                break
            if refs.index(r) == refs.index(run[-1][1]) + 1:
                run.append((t, r))
        if len(run) >= 3:
            quoted.update(run)
    for r in cands:
        cands[r] = sorted(c for c in cands[r] if (c[0], r) not in quoted)
    cands[refs[0]] = []  # the first verse's explanation starts when the bloc ends; a later mention is not its start
    # the longest chain of repeats in verse order (earliest on ties); the first verse may start at the bloc
    best = {}

    def chain(i, after):
        if i == len(refs):
            return (0, 0.0, ())
        key = (i, after)
        if key not in best:
            options = [chain(i + 1, after)]
            for t, heard in cands[refs[i]]:
                if t > after:
                    n, total, picks = chain(i + 1, t)
                    options.append((n + 1, total + t, ((refs[i], t, heard),) + picks))
            best[key] = max(options, key=lambda o: (o[0], -o[1]))
        return best[key]
    return {r: (t, h) for r, t, h in chain(0, bloc["end"])[2]}


_QURAN = ()


def explanation_rounds(segs, speakers, arabic_passes, hausa_passes, quran_index, lecture_end):
    """Rounds of [bloc + explanations]; see the module docstring."""
    global _QURAN
    _QURAN = quran_index
    quran = [s for s in segs if s["type"] == "quran"]
    if not speakers or not quran:
        return None
    reciter, blocs = reciter_blocs(speakers, quran)
    latin = [s for p in hausa_passes for s in p if s["text"].strip() and not re.search(r"[؀-ۿ]", s["text"])]
    arabic = [s for p in arabic_passes for s in p if re.search(r"[؀-ۿ]", s["text"])]
    rounds = []
    for i, b in enumerate(blocs):
        end = blocs[i + 1]["start"] if i + 1 < len(blocs) else lecture_end
        reps = find_repeats(b, end, arabic, latin)
        refs = [tuple(map(int, r.split(":"))) for r in b["refs"]]
        starts = []
        for k, r in enumerate(refs):
            if k == 0:
                starts.append((r, b["end"], "after bloc", ""))
            elif r in reps:
                starts.append((r, reps[r][0], "repeat", reps[r][1]))
        expl = []
        for k, (r, t, how, heard) in enumerate(starts):
            stop = starts[k + 1][1] if k + 1 < len(starts) else end
            expl.append({"ref": f"{r[0]}:{r[1]}", "start": round(t, 2), "end": round(stop, 2), "found": how,
                         "heard": heard})
        found = {e["ref"] for e in expl}
        rounds.append({"bloc": b, "explanations": expl, "missing": [r for r in b["refs"] if r not in found]})
    return {"reciter_speaker": reciter, "rounds": rounds}
