"""Review tool server: proofread a lecture transcript and save it back.

Run from the project root:
    python3 review/server.py            then open http://localhost:8001

Needs only the Python standard library (no virtualenv).
Serves the project folder (so the review page can load player/audio and player/data), plus:
  GET  /api/lectures            list of lectures in player/data
  GET  /api/verses?refs=2:3,2:4 verse texts for references (400 if one does not exist)
  GET  /api/glossary            the Arabic phrase glossary
  POST /api/glossary            add a phrase to the glossary
  POST /api/save/<lecture>      validate and save a transcript (previous version kept in work/<id>/backups/)
  GET  /api/peaks/<lecture>      waveform outline for the editor (computed once with ffmpeg, cached in work/<id>/)
  GET  /api/suggestions/<lecture>          spelling/spacing suggestions from pipeline/suggest_corrections.py
  POST /api/suggestions/<lecture>/dismiss  {"start", "original"}: hide a suggestion for good
"""
import array
import datetime
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "player"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from serve import QuietServer, RangeHandler  # noqa: E402  (player/serve.py: seeking support)

DATA = os.path.join(ROOT, "player", "data")
GLOSSARY = os.path.join(ROOT, "pipeline", "glossary.json")
LECTURE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
REF = re.compile(r"^\d{1,3}:\d{1,3}$")
KEEP_BACKUPS = 30


PEAKS_PER_SECOND = 50


def peaks(lid):
    """Loudest sample per 1/50 s, scaled 0-100: enough to draw the waveform at any zoom without the
    browser decoding the whole lecture. Cached next to the lecture's other work files."""
    doc = json.load(open(os.path.join(DATA, lid + ".json")))
    audio = os.path.join(ROOT, "player", doc["audio"])
    cache = os.path.join(ROOT, "work", lid, "peaks.json")
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(audio):
        return json.load(open(cache))
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is needed to draw the waveform (macOS: brew install ffmpeg)")
    sr = 8000
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", audio, "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    x = array.array("h")
    x.frombytes(raw[:len(raw) // 2 * 2])
    if sys.byteorder != "little":
        x.byteswap()
    step = sr // PEAKS_PER_SECOND
    loud = [max(max(c), -min(c)) if c else 0 for c in (x[i:i + step] for i in range(0, len(x), step))]
    top = max(loud) or 1
    out = {"rate": PEAKS_PER_SECOND, "duration": round(len(x) / sr, 3),
           "peaks": [round(100 * v / top) for v in loud]}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(out, open(cache, "w"), separators=(",", ":"))
    return out


def suggestions_path(lid):
    return os.path.join(ROOT, "work", lid, "suggestions.json")


def verse_entries(refs):
    # build.py needs rapidfuzz at import time; import lazily so the server runs on plain Python
    try:
        from build import verse_entries as ve
    except ImportError:
        import types
        sys.modules.setdefault("rapidfuzz", types.SimpleNamespace(fuzz=None))
        from build import verse_entries as ve
    return ve(refs)


def skeleton(t):
    import unicodedata
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if c.isalpha() and not unicodedata.combining(c))
    return re.sub(r"(.)\1+", r"\1", t)


def validate_rounds(rounds):
    """The verse explanations: each has a real verse reference or a label, and they do not overlap."""
    if rounds is None:
        return
    if not isinstance(rounds, list):
        raise ValueError("rounds must be a list")
    for n, rnd in enumerate(rounds, 1):
        b = rnd.get("bloc") or {}
        if not 0 <= float(b.get("start", -1)) < float(b.get("end", -1)):
            raise ValueError(f"round {n}: the bloc's start must be before its end")
        prev_end = -1.0
        for e in rnd.get("explanations", []):
            name = e.get("ref") or e.get("label") or "?"
            start, end = float(e["start"]), float(e["end"])
            if not 0 <= start < end:
                raise ValueError(f"round {n}, {name}: start must be before end")
            if start < prev_end - 0.05:
                raise ValueError(f"round {n}, {name}: overlaps the explanation before it")
            prev_end = end
            if "ref" in e:
                if not isinstance(e["ref"], str) or not REF.match(e["ref"]):
                    raise ValueError(f"round {n}: {e['ref']!r} is not a verse reference like 2:3")
                verse_entries([e["ref"]])  # raises for a verse that does not exist
            elif not str(e.get("label", "")).strip():
                raise ValueError(f"round {n}: an explanation needs a verse reference or a label")
        have = {e.get("ref") for e in rnd.get("explanations", [])}
        rnd["missing"] = [r for r in b.get("refs", []) if r not in have]


def validate(doc):
    segs = doc.get("segments")
    if not isinstance(segs, list):
        raise ValueError("segments missing")
    prev = -1
    for i, s in enumerate(segs, 1):
        where = f"segment {i}"
        if s.get("type") not in ("speech", "quran", "arabic"):
            raise ValueError(f"{where}: unknown type {s.get('type')!r}")
        start, end = float(s["start"]), float(s["end"])
        if not 0 <= start < end:
            raise ValueError(f"{where}: start must be before end")
        if start < prev:
            raise ValueError(f"{where}: segments are out of order")
        prev = start
        if s["type"] == "quran":
            refs = s.get("refs") or []
            if not refs or not all(isinstance(r, str) and REF.match(r) for r in refs):
                raise ValueError(f"{where}: Quran segment needs references like 2:3")
        elif not str(s.get("text", "")).strip():
            raise ValueError(f"{where}: text is empty")
        s["id"] = i
    validate_rounds(doc.get("rounds"))
    # verse texts always come from the official data, never from the browser
    doc["verses"] = verse_entries([r for s in segs if s["type"] == "quran" for r in s["refs"]] +
                                  [e["ref"] for rnd in doc.get("rounds") or [] for e in rnd["explanations"] if "ref" in e])
    done = sum(1 for s in segs if s.get("reviewed"))
    doc["status"] = "reviewed" if done == len(segs) else f"draft - {done} of {len(segs)} segments reviewed"
    return doc


class Handler(RangeHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            super().log_message(fmt, *args)

    def reply(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/review/")
            self.end_headers()
            return
        if u.path == "/api/lectures":
            items = []
            for f in sorted(os.listdir(DATA)):
                if f.endswith(".json"):
                    d = json.load(open(os.path.join(DATA, f)))
                    segs = d.get("segments", [])
                    items.append({"id": f[:-5], "title": d.get("title", f[:-5]), "segments": len(segs),
                                  "reviewed": sum(1 for s in segs if s.get("reviewed"))})
            return self.reply(200, items)
        if u.path == "/api/verses":
            refs = [r.strip() for r in urllib.parse.parse_qs(u.query).get("refs", [""])[0].split(",") if r.strip()]
            try:
                if not all(REF.match(r) for r in refs):
                    raise ValueError("use references like 2:3")
                return self.reply(200, verse_entries(refs))
            except (ValueError, KeyError) as e:
                return self.reply(400, {"error": str(e)})
        if u.path == "/api/glossary":
            return self.reply(200, json.load(open(GLOSSARY))["phrases"])
        m = re.match(r"^/api/peaks/([^/]+)$", u.path)
        if m:
            lid = m.group(1)
            if not LECTURE_ID.match(lid) or not os.path.exists(os.path.join(DATA, lid + ".json")):
                return self.reply(404, {"error": "unknown lecture"})
            try:
                return self.reply(200, peaks(lid))
            except (RuntimeError, OSError, subprocess.CalledProcessError) as e:
                return self.reply(503, {"error": str(e)})
        m = re.match(r"^/api/suggestions/([^/]+)$", u.path)
        if m:
            path = suggestions_path(m.group(1))
            if not LECTURE_ID.match(m.group(1)) or not os.path.exists(path):
                return self.reply(200, {"model": None, "items": []})
            d = json.load(open(path))
            items = [i for i in d.get("items", []) if i.get("status") == "suggested" and not i.get("dismissed")]
            return self.reply(200, {"model": d.get("model"), "items": items})
        return super().do_GET()

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self.reply(400, {"error": "invalid JSON"})

        m = re.match(r"^/api/save/([^/]+)$", u.path)
        if m:
            lid = m.group(1)
            path = os.path.join(DATA, lid + ".json")
            if not LECTURE_ID.match(lid) or not os.path.exists(path):
                return self.reply(404, {"error": "unknown lecture"})
            try:
                doc = validate(body)
            except (ValueError, KeyError, TypeError) as e:
                return self.reply(400, {"error": str(e)})
            bdir = os.path.join(ROOT, "work", lid, "backups")
            os.makedirs(bdir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            os.replace(path, os.path.join(bdir, f"{lid}.{stamp}.json"))
            for old in sorted(os.listdir(bdir))[:-KEEP_BACKUPS]:
                os.remove(os.path.join(bdir, old))
            tmp = path + ".tmp"
            json.dump(doc, open(tmp, "w"), ensure_ascii=False, indent=1)
            os.replace(tmp, path)
            return self.reply(200, {"ok": True, "status": doc["status"], "verses": doc["verses"]})

        m = re.match(r"^/api/suggestions/([^/]+)/dismiss$", u.path)
        if m:
            path = suggestions_path(m.group(1))
            if not LECTURE_ID.match(m.group(1)) or not os.path.exists(path):
                return self.reply(404, {"error": "no suggestions for this lecture"})
            d = json.load(open(path))
            hit = [i for i in d["items"] if abs(i["start"] - float(body.get("start", -1))) < 0.01
                   and i["original"] == body.get("original")]
            if not hit:
                return self.reply(404, {"error": "suggestion not found"})
            for i in hit:
                i["dismissed"] = True
            tmp = path + ".tmp"
            json.dump(d, open(tmp, "w"), ensure_ascii=False, indent=1)
            os.replace(tmp, path)
            return self.reply(200, {"ok": True})

        if u.path == "/api/glossary":
            arabic, translit = str(body.get("arabic", "")).strip(), str(body.get("translit", "")).strip()
            if not arabic or not translit:
                return self.reply(400, {"error": "Arabic text and transliteration are required"})
            g = json.load(open(GLOSSARY))
            pid = re.sub(r"[^a-z0-9]+", "_", skeleton(translit))[:40] or "phrase"
            if any(p["id"] == pid or p["arabic"] == arabic for p in g["phrases"]):
                return self.reply(409, {"error": "this phrase is already in the glossary"})
            entry = {"id": pid, "pattern": re.escape(skeleton(translit)), "arabic": arabic, "translit": translit,
                     "meaning_en": str(body.get("meaning_en", "")).strip(),
                     "meaning_ha": str(body.get("meaning_ha", "")).strip()}
            g["phrases"].append(entry)
            json.dump(g, open(GLOSSARY, "w"), ensure_ascii=False, indent=1)
            return self.reply(200, entry)
        return self.reply(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"Review tool: http://localhost:{port}")
    QuietServer(("127.0.0.1", port), Handler).serve_forever()
