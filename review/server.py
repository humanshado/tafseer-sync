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
"""
import datetime
import http.server
import json
import os
import re
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "player"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from serve import RangeHandler  # noqa: E402  (player/serve.py: seeking support)

DATA = os.path.join(ROOT, "player", "data")
GLOSSARY = os.path.join(ROOT, "pipeline", "glossary.json")
LECTURE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
REF = re.compile(r"^\d{1,3}:\d{1,3}$")
KEEP_BACKUPS = 30


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
    # verse texts always come from the official data, never from the browser
    doc["verses"] = verse_entries(r for s in segs if s["type"] == "quran" for r in s["refs"])
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
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
