"""Cut the lecturer's explanation of each verse into its own audio file, with its transcript.

Reads the rounds that build.py adds to a transcript (see explanations.py) and writes, per lecture:
  OUT/<id>/<surah>_<ayah>.mp3   the explanation, from the lecturer's repeat to the next verse's
  OUT/<id>/<surah>_<ayah>.txt   its transcript (Hausa, and the verses he quotes, by reference)
  OUT/<id>/index.md             every bloc and explanation with its times
Default OUT is work/explanations (not committed: the audio is large).

Usage:
  python export_explanations.py player/data/ID.json [--out work/explanations] [--audio player/audio/ID.mp3]
"""
import argparse
import json
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clock(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "explanations"))
    ap.add_argument("--audio", help="default: the transcript's audio under player/")
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed to cut the audio but was not found. Install it (macOS: brew install ffmpeg; "
                         "Debian/Ubuntu: sudo apt install ffmpeg; Windows: winget install ffmpeg), then run this again.")
    doc = json.load(open(a.transcript))
    if not doc.get("rounds"):
        raise SystemExit("No rounds in this transcript: run diarize.py, then build.py.")
    audio = a.audio or os.path.join(ROOT, "player", doc["audio"])
    out = os.path.join(a.out, doc["lecture"])
    os.makedirs(out, exist_ok=True)
    index = [f"# {doc['title']}: explanations verse by verse", "",
             "Machine-made from the speakers and the recitation found; check the boundaries by listening.", ""]
    for rnd in doc["rounds"]:
        b = rnd["bloc"]
        index += [f"## Bloc {b['refs'][0]}–{b['refs'][-1]} (recited {clock(b['start'])}–{clock(b['end'])})", ""]
        for k, e in enumerate(rnd["explanations"], 1):
            title = e.get("ref") or e.get("label", "")
            name = e["ref"].replace(":", "_") if e.get("ref") else \
                f"round{doc['rounds'].index(rnd) + 1}_{k:02d}_" + (re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "part")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(e["start"]),
                            "-to", str(e["end"]), "-i", audio, "-c:a", "libmp3lame", "-q:a", "4",
                            os.path.join(out, name + ".mp3")], check=True)
            lines = []
            for s in doc["segments"]:
                if e["start"] <= (s["start"] + s["end"]) / 2 < e["end"]:
                    body = ("[Quran " + ", ".join(s["refs"]) + "]") if s["type"] == "quran" else s.get("text", "")
                    lines.append(f"[{clock(s['start'])}] {body}")
            open(os.path.join(out, name + ".txt"), "w").write(
                f"Explanation of {title}: {clock(e['start'])}–{clock(e['end'])}\n\n" + "\n".join(lines) + "\n")
            how = {"after bloc": "starts after the bloc", "repeat": f"repeat heard: {e.get('heard', '')[:60]}"}.get(
                e.get("found"), e.get("found") or "set by hand") + ("; checked by ear" if e.get("checked") else "")
            index.append(f"- **{title}** {clock(e['start'])}–{clock(e['end'])} "
                         f"({(e['end'] - e['start']) / 60:.1f} min; {how}) [{name}.mp3]({name}.mp3)")
        if rnd["missing"]:
            index.append(f"- Not found: {', '.join(rnd['missing'])} (inside the explanation before it, if explained)")
        index.append("")
    open(os.path.join(out, "index.md"), "w").write("\n".join(index))
    n = sum(len(r["explanations"]) for r in doc["rounds"])
    print(f"{n} explanations -> {out}")


if __name__ == "__main__":
    main()
