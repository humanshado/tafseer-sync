"""Step 2: turn raw Whisper output into the player's transcript JSON.

* Uses the Hausa pass as the main timeline.
* Finds Quran recitation in the Arabic pass by matching it against the full
  Quran text, and replaces it with the official text (by surah:ayah reference).
* Optional third pass, raw_omni.json (Meta's Omnilingual ASR, see run_lecture.sh). Its Arabic-script
  output is searched for recitation too (it writes recited verses far more accurately than Whisper),
  and where it wrote Hausa in Latin letters with at least the Hausa model's confidence, its text is
  used. Each segment keeps the other model's text in `alt` for the reviewer.
* Tags known Arabic expressions in the Hausa text using glossary.json.
* Flags low-confidence segments for human review.

Usage:
  python build.py WORK_DIR LECTURE_ID AUDIO_URL OUT.json [--title "..."]
"""
import argparse
import json
import os
import re
import sys
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
    t = re.sub(r"\b(ال)?رحمان\b", lambda m: (m.group(1) or "") + "رحمن", t)  # plain spelling of الرحمن
    # Recognisers often hear a plural in -una for -ina (النبيون for النبيين) and drop the "wa-" of
    # "wa-l-..." (الصديقين for والصديقين). Fold both on either side; matching only compares letters.
    t = " ".join(re.sub(r"ون$", "ين", re.sub(r"^وال", "ال", w)) for w in t.split())
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


ISTIADHA = re.compile(r"(اعوذ بالله )?من الشيطان الرجيم")  # recognisers sometimes split off "a'udhu billahi"


def match_quran(text, words, owner, index, verse_pairs, focus=(), min_words=3, min_score=72):
    """Return (score, [(surah, ayah), ...]) if the Arabic text is Quran recitation.

    `focus` lists the surahs the lecture is about; when a phrase occurs in several places
    (e.g. "al-hamdu lillahi rabbil-alamin" is 1:2, 6:45, 10:10, 37:182, ...), those win.
    """
    # The isti'adha ("a'udhu billahi mina sh-shaytani r-rajim") is said before reciting; it is not a
    # verse, although it echoes the end of 16:98. Drop it so only what follows is matched.
    seg = ISTIADHA.sub(" ", norm_ar(text)).split()
    if len(seg) < min_words:
        return None
    hits = defaultdict(list)  # offset -> [(seg index, quran position)]
    for i in range(len(seg) - 1):
        for p in index.get((seg[i], seg[i + 1]), ())[:400]:  # very common bigrams are capped
            hits[p - i].append((i, p))
    if hits:
        # pool nearby offsets (speech recognition drops or merges words)
        pooled = defaultdict(int)
        for off, h in hits.items():
            pooled[off // 3] += len(h)
        top = max(pooled.values())
        if top >= max(2, 0.25 * (len(seg) - 1)):
            cands = [b for b, n in pooled.items() if n == top]
            b = max(cands, key=lambda b: (owner[max(0, b * 3)][0] in focus, -b))
            off = max((o for o in hits if o // 3 == b), key=lambda o: len(hits[o]))
            # only the Quran words that were actually heard decide which verses are shown
            pos = sorted({q for o in hits if abs(o - off) <= 2 for _, p in hits[o] for q in (p, p + 1)})
            score = fuzz.partial_ratio(" ".join(seg), " ".join(words[pos[0]:pos[-1] + 1]))
            if score >= min_score:
                refs = []
                for q in pos:
                    if owner[q] not in refs:
                        refs.append(owner[q])
                return score, refs
    # Fallback for garbled recitation: character-level fuzzy match against pairs of consecutive verses.
    s = " ".join(seg)
    if len(s) < 20:
        return None
    best = (0, False, None)
    for t, refs in verse_pairs:
        # the verse text must be comparable in length, or a short verse (e.g. "حم") matches anywhere
        if len(t) < 15 or len(t) < 0.5 * len(s):
            continue
        sc = fuzz.partial_ratio(s, t, score_cutoff=80)
        if sc and (sc, refs[0][0] in focus) > best[:2]:
            best = (sc, refs[0][0] in focus, refs)
    if best[0] < 80:
        return None
    # keep only the verses of the pair that were actually heard
    refs = [r for r in best[2] if len(verse_text[r]) >= 8 and fuzz.partial_ratio(verse_text[r], s) >= 80]
    return best[0], refs or best[2][:1]


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


def find_openings(ar, formulas, min_score=75, continue_score=60, max_gap=4.0):
    """Arabic-pass segments that form a known opening formula at the start of the lecture.

    Returns (blocks, indices of the Arabic segments used). A block starts with a matching segment
    inside the formula's start window and continues while consecutive segments keep matching.
    """
    blocks, used = [], set()
    for f in formulas:
        target = " ".join(norm_ar(m) for m in f["match"])
        block = None
        for k, s in enumerate(ar):
            seg = norm_ar(s["text"])
            score = fuzz.partial_ratio(seg, target) if seg else 0
            ok = len(seg.split()) >= 2 and score >= min_score
            if block is None:
                if s["start"] > f.get("starts_within_seconds", 90):
                    break
                if ok and len(seg.split()) >= 4:
                    block = {"id": f["id"], "label": f["label"], "text": f["text"],
                             "start": s["start"], "end": s["end"], "heard": s["text"]}
                    used.add(k)
            elif s["start"] - block["end"] <= max_gap and (
                    score >= continue_score or len(seg.split()) <= 2):  # a started block tolerates rougher text
                block["end"] = s["end"]
                block["heard"] += " " + s["text"]
                used.add(k)
            else:
                break
        if block:
            blocks.append(block)
    return blocks, used


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def detect_quran(segs, skip, words, owner, index, verse_pairs, focus, source):
    """Recitation passages in one Arabic-capable pass; neighbours continuing the same passage merge."""
    quran = []
    for k, s in enumerate(segs):
        if k in skip:
            continue
        if source == "omni" and len(re.findall(r"[A-Za-z]", s["text"])) > 2:
            continue  # Hausa garbled into a mix of Latin and Arabic letters, not recitation
        m = match_quran(s["text"], words, owner, index, verse_pairs, focus)
        if not m:
            continue
        score, refs = m
        start, end = s["start"], s["end"]
        if source == "omni" and s.get("words"):
            # CTC word timings are accurate: trim speech before and after the recited words
            vw = {w for r in refs for w in verse_text.get(r, "").split()}
            hit = [w for w in s["words"] if norm_ar(w["w"]) in vw]
            if hit:
                start, end = hit[0]["s"], hit[-1]["e"]
        prev = quran[-1] if quran else None
        if prev and start - prev["end"] < 2.5 and refs[0][0] == prev["refs"][-1][0] \
                and 0 <= refs[0][1] - prev["refs"][-1][1] <= 1:
            prev["end"] = end
            prev["heard"] += " " + s["text"]
            prev["score"] = min(prev["score"], score)
            prev["refs"] += [r for r in refs if r not in prev["refs"]]
        else:
            quran.append({"start": start, "end": end, "refs": refs,
                          "score": score, "heard": s["text"], "source": source})
    return quran


def merge_passages(found, focus=()):
    """Combine passages found by different passes into one list.

    Passages of the same surah that overlap, or follow on within 2.5 s with the next verse, form one
    passage with all their verses. Its start comes from the part holding the first verse and its end
    from the part holding the last; Omnilingual parts win there, because its chunks are cut at pauses
    while Whisper segments often run on into the commentary. Where passes disagree on the surah for
    the same stretch (overlapping by more than half), a surah the lecture covers wins, then the better
    match: "الم" opens six surahs."""
    groups = []
    for q in sorted(found, key=lambda q: q["start"]):
        g = groups[-1] if groups else None
        if g and q["refs"][0][0] != g["refs"][0][0] and overlap(g["start"], g["end"], q["start"], q["end"]) > \
                0.5 * min(g["end"] - g["start"], q["end"] - q["start"]):
            rank = lambda x: (x["refs"][0][0] in focus, x["score"])  # noqa: E731
            if rank(q) > max(rank(x) for x in g["parts"]):
                groups[-1] = {"start": q["start"], "end": q["end"], "refs": list(q["refs"]), "parts": [q]}
            continue
        if g and q["refs"][0][0] == g["refs"][0][0] and (
                overlap(g["start"], g["end"], q["start"], q["end"]) > 0
                or (q["start"] - g["end"] < 2.5 and 0 <= q["refs"][0][1] - g["refs"][-1][1] <= 1)):
            g["parts"].append(q)
            g["refs"] = sorted(set(g["refs"]) | set(q["refs"]))
            g["start"], g["end"] = min(g["start"], q["start"]), max(g["end"], q["end"])
        else:
            groups.append({"start": q["start"], "end": q["end"], "refs": list(q["refs"]), "parts": [q]})
    out = []
    for g in groups:
        def edge(ref, pick):
            parts = [x for x in g["parts"] if ref in x["refs"]]
            omni = [x for x in parts if x["source"] == "omni"]
            return pick(omni or parts)
        best = max(g["parts"], key=lambda x: x["score"])
        out.append({"start": edge(g["refs"][0], lambda xs: min(x["start"] for x in xs)),
                    "end": edge(g["refs"][-1], lambda xs: max(x["end"] for x in xs)),
                    "refs": g["refs"], "score": best["score"], "source": best["source"],
                    "heard": " ".join(x["heard"] for x in sorted(g["parts"], key=lambda x: x["start"])
                                      if x["source"] == best["source"])})
    return out


NON_LATIN = re.compile(r"[^\x00-\x7fɓɗƙƴƁƊƘƳʼ’]")


def pick_hausa(s, omni, review_below, omni_below):
    """(text, confidence flag, source, other model's text) for one Hausa-pass segment.

    The Omnilingual segment cut from the same stretch of audio replaces the Hausa model's text only
    if it is written in Latin letters (it sometimes writes Hausa in Arabic script, or invents text in
    other scripts) and is at least as confident; both report the log of the mean character probability."""
    dur = max(0.01, s["end"] - s["start"])
    o = max(omni, key=lambda o: overlap(s["start"], s["end"], o["start"], o["end"]), default=None)
    if o is None or overlap(s["start"], s["end"], o["start"], o["end"]) < 0.8 * max(dur, o["end"] - o["start"]):
        return s["text"], s["avg_logprob"] < review_below or s["no_speech_prob"] > 0.6, "hausa", None
    latin = o["text"].strip() and not NON_LATIN.search(o["text"])
    if latin and o["avg_logprob"] >= s["avg_logprob"]:
        return o["text"], o["avg_logprob"] < omni_below, "omni", s["text"]
    return (s["text"], s["avg_logprob"] < review_below or s["no_speech_prob"] > 0.6, "hausa",
            o["text"] if latin else None)


_qdata = {}


def verse_entries(refs):
    """{"2:3": {surah, ayah, surah_name, surah_translit, uthmani, warsh, hausa}} for the given refs.

    Raises KeyError/ValueError for a reference that does not exist.
    """
    if not _qdata:
        _qdata["uth"] = json.load(open(os.path.join(QD, "uthmani.json")))
        _qdata["hau"] = json.load(open(os.path.join(QD, "hausa_gummi.json")))
        _qdata["warsh"] = json.load(open(os.path.join(QD, "warsh.json")))
        _qdata["names"] = {c["id"]: c for c in json.load(open(os.path.join(QD, "chapters.json")))["chapters"]}
    q = _qdata
    out = {}
    for c, v in sorted({tuple(int(x) for x in r.split(":")) for r in refs}):
        if c not in q["names"] or not 1 <= v <= q["names"][c]["total_verses"]:
            raise ValueError(f"no such verse: {c}:{v}")
        w = [x["text"] for x in q["warsh"][str(c)] if v in x.get("number_in_hafs", [])]
        out[f"{c}:{v}"] = {
            "surah": c, "ayah": v,
            "surah_name": q["names"][c]["name"], "surah_translit": q["names"][c]["transliteration"],
            "uthmani": q["uth"][str(c)][v - 1]["text"],
            "warsh": w[0] if w else "",
            "hausa": q["hau"][str(c)][v - 1]["text"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir")
    ap.add_argument("lecture_id")
    ap.add_argument("audio_url")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--focus", default="", help="surahs the lecture covers, e.g. 1,2")
    ap.add_argument("--discard-reviewed", action="store_true",
                    help="regenerate everything, throwing away reviewed segments in the existing output")
    a = ap.parse_args()

    ha_raw = json.load(open(os.path.join(a.work_dir, "raw_ha.json")))
    ha = ha_raw["segments"]
    review_below = ha_raw.get("review_below_logprob", -1.0)  # CTC models report confidence differently
    ar_path = os.path.join(a.work_dir, "raw_ar.json")
    ar = json.load(open(ar_path))["segments"] if os.path.exists(ar_path) else []
    omni_path = os.path.join(a.work_dir, "raw_omni.json")
    omni_raw = json.load(open(omni_path)) if os.path.exists(omni_path) else {"segments": []}
    omni = omni_raw["segments"]
    omni_below = omni_raw.get("review_below_logprob", review_below)
    words, owner, index = load_quran()
    verse_pairs = load_verse_pairs()
    focus = {int(x) for x in a.focus.split(",") if x.strip()}
    glossary = json.load(open(os.path.join(HERE, "glossary.json")))["phrases"]
    for g in glossary:
        g["re"] = re.compile(g["pattern"])

    # 0. Set opening formulas (praise, testimony, prayers) are Arabic but not Quran recitation,
    #    even where they quote a verse. Find them first and keep them out of the verse matching.
    formulas = json.load(open(os.path.join(HERE, "formulas.json")))["formulas"]
    openings, used = find_openings(ar, formulas)

    # 1. Quran recitation from the Arabic pass (and the Omnilingual pass, outside the openings).
    quran = detect_quran(ar, used, words, owner, index, verse_pairs, focus, "whisper")
    in_opening = {k for k, o in enumerate(omni)
                  if any(overlap(o["start"], o["end"], b["start"], b["end"]) > 0 for b in openings)}
    quran = merge_passages(quran + detect_quran(omni, in_opening, words, owner, index, verse_pairs, focus, "omni"),
                           focus)

    # 2. Hausa commentary, minus the parts covered by recitation.
    segs = []
    for s in ha:
        dur = max(0.01, s["end"] - s["start"])
        if any(overlap(s["start"], s["end"], q["start"], q["end"]) / dur > 0.5 for q in quran + openings):
            continue
        text, review, source, alt = pick_hausa(s, omni, review_below, omni_below)
        phrases = []
        for g in glossary:
            if g["re"].search(skeleton(text)):
                phrases.append({"id": g["id"], "arabic": g["arabic"],
                                "translit": g["translit"], "meaning_en": g["meaning_en"],
                                "meaning_ha": g["meaning_ha"]})
        seg = {"start": s["start"], "end": s["end"], "type": "speech", "text": text, "phrases": phrases,
               "review": review}
        if omni:
            seg["asr"] = source
        if alt:
            seg["alt"] = alt
        segs.append(seg)

    for q in quran:
        segs.append({"start": q["start"], "end": q["end"], "type": "quran",
                     "refs": [f"{c}:{v}" for c, v in q["refs"]], "heard": q["heard"],
                     "match": q["score"], "review": q["score"] < 85})

    for o in openings:
        segs.append({"start": o["start"], "end": o["end"], "type": "arabic", "label": o["label"],
                     "formula": o["id"], "text": o["text"], "heard": o["heard"], "review": True})

    # Keep human work: if the output exists, its reviewed segments win over freshly generated ones.
    kept = 0
    if os.path.exists(a.out) and not a.discard_reviewed:
        reviewed = [s for s in json.load(open(a.out))["segments"] if s.get("reviewed")]
        def clashes(s):
            d = max(0.01, s["end"] - s["start"])
            return any(overlap(s["start"], s["end"], r["start"], r["end"]) / min(d, max(0.01, r["end"] - r["start"])) > 0.3
                       for r in reviewed)
        segs = [s for s in segs if not clashes(s)] + reviewed
        kept = len(reviewed)

    segs.sort(key=lambda x: x["start"])
    for i, s in enumerate(segs, 1):
        s["id"] = i

    # 2b. Rounds: the reciter's blocs and the lecturer's explanation of each verse (needs speakers.json).
    rounds = None
    sp_path = os.path.join(a.work_dir, "speakers.json")
    if os.path.exists(sp_path):
        sys.modules.setdefault("build", sys.modules[__name__])  # explanations.py imports this module
        from explanations import explanation_rounds
        rounds = explanation_rounds(segs, json.load(open(sp_path))["segments"], [ar, omni], [ha, omni],
                                    (words, owner, index, verse_pairs, focus), max(s["end"] for s in segs))

    # 3. Verse texts the lecture needs: Hafs (Uthmani), Warsh, and the Hausa translation (Gumi).
    verses = verse_entries(r for s in segs if s["type"] == "quran" for r in s["refs"])

    out = {
        "lecture": a.lecture_id, "title": a.title or a.lecture_id, "audio": a.audio_url,
        "language": "ha", "status": (f"draft - {kept} of {len(segs)} segments reviewed" if kept else "draft - machine transcript, not yet reviewed"),
        "sources": {
            "quran_text": "Tanzil.net Uthmani (CC BY 3.0, verbatim) via quran-json",
            "warsh": "KFGQPC Warsh via Quranpedia.net, via quran-json",
            "hausa_translation": "Abubakar Mahmoud Gumi, via QuranEnc.com",
        },
        "segments": segs, "verses": verses,
    }
    if rounds:
        out["rounds"] = rounds["rounds"]
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    if rounds:
        for r in rounds["rounds"]:
            b = r["bloc"]
            print(f"Bloc {b['refs'][0]}-{b['refs'][-1].split(':')[1]} at {b['start']:.0f}s: explanations for "
                  f"{', '.join(e['ref'] for e in r['explanations'])}"
                  + (f"; not found: {', '.join(r['missing'])}" if r["missing"] else ""))
    if omni:
        sp = [s for s in segs if s["type"] == "speech" and not s.get("reviewed")]
        print(f"Omnilingual pass: text used for {sum(s.get('asr') == 'omni' for s in sp)} of {len(sp)} Hausa segments; "
              f"Quran passages found by Whisper {sum(q['source'] == 'whisper' for q in quran)}, "
              f"by Omnilingual {sum(q['source'] == 'omni' for q in quran)} (best match of each kept)")
    print(f"{len(segs)} segments ({sum(s['type'] == 'quran' for s in segs)} Quran passages, "
          f"{sum(len(s.get('phrases', [])) for s in segs)} tagged phrases, "
          f"{sum(bool(s.get('review')) for s in segs)} flagged for review, {kept} reviewed segments kept) -> {a.out}")


if __name__ == "__main__":
    main()
