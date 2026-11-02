"""Step 2: turn raw Whisper output into the player's transcript JSON.

* Uses the Hausa pass as the main timeline.
* Finds Quran recitation in the Arabic pass by matching it against the full
  Quran text, and replaces it with the official text (by surah:ayah reference).
* Tags known Arabic expressions in the Hausa text using glossary.json.
* Flags low-confidence segments for human review.

Usage:
  python build.py WORK_DIR LECTURE_ID AUDIO_URL OUT.json [--title "..."]
"""
import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

from rapidfuzz import fuzz

HERE = os.path.dirname(os.path.abspath(__file__))
QD = os.path.join(HERE, "quran_data")

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")


def norm_ar(t):
    t = DIACRITICS.sub("", t)
    t = re.sub("[أإآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"[^ء-ي\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_quran():
    clean = json.load(open(os.path.join(QD, "simple-clean.json")))
    words, owner = [], []  # flat word stream of the whole Quran, and (surah, ayah) of each word
    for ch in clean.values():
        for v in ch:
            for w in norm_ar(v["text"]).split():
                words.append(w)
                owner.append((v["chapter"], v["verse"]))
    index = defaultdict(list)
    for i in range(len(words) - 1):
        index[(words[i], words[i + 1])].append(i)
    return words, owner, index


def match_quran(text, words, owner, index, verse_pairs, min_words=3, min_score=72):
    """Return (score, [(surah, ayah), ...]) if the Arabic text is Quran recitation."""
    seg = norm_ar(text).split()
    if len(seg) < min_words:
        return None
    votes = Counter()
    for i in range(len(seg) - 1):
        for p in index.get((seg[i], seg[i + 1]), ())[:400]:  # very common bigrams are capped
            votes[p - i] += 1
    if votes:
        # pool votes of nearby offsets (ASR drops/merges words), then take the best exact offset
        pooled = Counter()
        for off, n in votes.items():
            pooled[off // 3] += n
        bucket, n = pooled.most_common(1)[0]
        if n >= max(2, 0.25 * (len(seg) - 1)):
            off = max((o for o in votes if o // 3 == bucket), key=votes.get)
            off = max(0, off)
            lo, hi = max(0, off - 3), min(len(words), off + len(seg) + 3)
            score = fuzz.partial_ratio(" ".join(seg), " ".join(words[lo:hi]))
            if score >= min_score:
                refs = []
                for r in owner[off:min(len(words), off + len(seg))]:
                    if r not in refs:
                        refs.append(r)
                return score, refs
    # Fallback for garbled recitation: character-level fuzzy match against every pair of consecutive verses.
    s = " ".join(seg)
    if len(s) < 20:
        return None
    best = max(((fuzz.partial_ratio(s, t, score_cutoff=80), refs) for t, refs in verse_pairs),
               key=lambda x: x[0])
    if best[0] < 80:
        return None
    # keep only the verses of the pair that were actually heard
    refs = [r for r in best[1] if len(verse_text[r]) >= 8 and fuzz.partial_ratio(verse_text[r], s) >= 80]
    return best[0], refs or best[1][:1]


verse_text = {}


def load_verse_pairs():
    clean = json.load(open(os.path.join(QD, "simple-clean.json")))
    flat = [((v["chapter"], v["verse"]), norm_ar(v["text"])) for ch in clean.values() for v in ch]
    verse_text.update(flat)
    pairs = []
    for i, (ref, t) in enumerate(flat):
        nxt = flat[i + 1] if i + 1 < len(flat) and flat[i + 1][0][0] == ref[0] else None
        pairs.append((t + (" " + nxt[1] if nxt else ""), [ref] + ([nxt[0]] if nxt else [])))
    return pairs


def skeleton(t):
    """Spelling-tolerant form of Latin-script text: 'Bismillahi rRahmani' -> 'bismilahirahmani'."""
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if c.isalpha() and not unicodedata.combining(c))
    return re.sub(r"(.)\1+", r"\1", t)


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("lecture_id")
    ap.add_argument("audio_url")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    ha_raw = json.load(open(os.path.join(a.work_dir, "raw_ha.json")))
    ha = ha_raw["segments"]
    review_below = ha_raw.get("review_below_logprob", -1.0)  # CTC models report confidence differently
    ar_path = os.path.join(a.work_dir, "raw_ar.json")
    ar = json.load(open(ar_path))["segments"] if os.path.exists(ar_path) else []
    words, owner, index = load_quran()
    verse_pairs = load_verse_pairs()
    glossary = json.load(open(os.path.join(HERE, "glossary.json")))["phrases"]
    for g in glossary:
        g["re"] = re.compile(g["pattern"])

    # 1. Quran recitation from the Arabic pass; merge neighbours that continue the same passage.
    quran = []
    for s in ar:
        m = match_quran(s["text"], words, owner, index, verse_pairs)
        if not m:
            continue
        score, refs = m
        prev = quran[-1] if quran else None
        if prev and s["start"] - prev["end"] < 2.5 and refs[0][0] == prev["refs"][-1][0] \
                and 0 <= refs[0][1] - prev["refs"][-1][1] <= 1:
            prev["end"] = s["end"]
            prev["heard"] += " " + s["text"]
            prev["score"] = min(prev["score"], score)
            prev["refs"] += [r for r in refs if r not in prev["refs"]]
        else:
            quran.append({"start": s["start"], "end": s["end"], "refs": refs,
                          "score": score, "heard": s["text"]})

    # 2. Hausa commentary, minus the parts covered by recitation.
    segs = []
    for s in ha:
        dur = max(0.01, s["end"] - s["start"])
        if any(overlap(s["start"], s["end"], q["start"], q["end"]) / dur > 0.5 for q in quran):
            continue
        phrases = []
        for g in glossary:
            if g["re"].search(skeleton(s["text"])):
                phrases.append({"id": g["id"], "arabic": g["arabic"],
                                "translit": g["translit"], "meaning_en": g["meaning_en"],
                                "meaning_ha": g["meaning_ha"]})
        segs.append({"start": s["start"], "end": s["end"], "type": "speech", "text": s["text"],
                     "phrases": phrases,
                     "review": s["avg_logprob"] < review_below or s["no_speech_prob"] > 0.6})

    for q in quran:
        segs.append({"start": q["start"], "end": q["end"], "type": "quran",
                     "refs": [f"{c}:{v}" for c, v in q["refs"]], "heard": q["heard"],
                     "match": q["score"], "review": q["score"] < 85})

    segs.sort(key=lambda x: x["start"])
    for i, s in enumerate(segs, 1):
        s["id"] = i

    # 3. Verse texts the lecture needs: Hafs (Uthmani), Warsh, and the Hausa translation (Gumi).
    needed = {(int(r.split(":")[0]), int(r.split(":")[1])) for s in segs if s["type"] == "quran" for r in s["refs"]}
    uth = json.load(open(os.path.join(QD, "uthmani.json")))
    hau = json.load(open(os.path.join(QD, "hausa_gummi.json")))
    warsh = json.load(open(os.path.join(QD, "warsh.json")))
    names = {c["id"]: c for c in json.load(open(os.path.join(QD, "chapters.json")))["chapters"]}
    verses = {}
    for c, v in sorted(needed):
        w = [x["text"] for x in warsh[str(c)] if v in x.get("number_in_hafs", [])]
        verses[f"{c}:{v}"] = {
            "surah": c, "ayah": v,
            "surah_name": names[c]["name"], "surah_translit": names[c]["transliteration"],
            "uthmani": uth[str(c)][v - 1]["text"],
            "warsh": w[0] if w else "",
            "hausa": hau[str(c)][v - 1]["text"],
        }

    out = {
        "lecture": a.lecture_id, "title": a.title or a.lecture_id, "audio": a.audio_url,
        "language": "ha", "status": "draft - machine transcript, not yet reviewed",
        "sources": {
            "quran_text": "Tanzil.net Uthmani (CC BY 3.0, verbatim) via quran-json",
            "warsh": "KFGQPC Warsh via Quranpedia.net, via quran-json",
            "hausa_translation": "Abubakar Mahmoud Gumi, via QuranEnc.com",
        },
        "segments": segs, "verses": verses,
    }
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(segs)} segments ({sum(s['type'] == 'quran' for s in segs)} Quran passages, "
          f"{sum(len(s.get('phrases', [])) for s in segs)} tagged phrases, "
          f"{sum(s['review'] for s in segs)} flagged for review) -> {a.out}")


if __name__ == "__main__":
    main()
