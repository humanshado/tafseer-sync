"""Suggest spelling and word-spacing corrections for a Hausa transcript with N-ATLaS.

The speech model often runs words together ("waɗansuharuffa") or misspells them ("akwi").
This asks N-ATLaS (NCAIR's Hausa/Yoruba/Igbo/English language model) for a tidied version of
each unreviewed Hausa segment. It only ever SUGGESTS: the transcript is not changed. Suggestions
go to work/<id>/suggestions.json, and the review tool shows each one beside the original so the
reviewer accepts or dismisses it while listening to the audio.

A language model may "improve" the scholar's words or invent text where the audio was unclear,
so every suggestion is checked mechanically and dropped if it adds, removes or changes more than
spacing and a few letters (see `check`). Segments too garbled to tidy are best left to the reviewer.

Backends:
  default    a quantised GGUF copy run with llama-cpp-python (practical on a CPU, about 5 GB RAM)
  --hf ID    the official weights with transformers (NCAIR1/N-ATLaS: gated, about 16 GB; for a GPU)

Usage:
  python suggest_corrections.py player/data/ID.json [--from 600 --to 720] [--flagged-only] [--limit N]
Re-running continues where it stopped: segments already processed with the same text are skipped.
"""
import argparse
import datetime
import difflib
import json
import os
import re
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GGUF = "QuantFactory/N-ATLaS-GGUF:N-ATLaS.Q4_K_M.gguf"

SYSTEM = (
    "You proofread Hausa text produced by a speech recogniser from a recorded Islamic lecture (tafsir). "
    "The recogniser often joins separate words together and misspells words. "
    "Return the same text with words correctly separated and obvious spelling mistakes fixed, "
    "using standard Hausa spelling including the letters ɓ, ɗ, ƙ and ƴ. "
    "Do not translate, paraphrase, reorder, add or remove words, and do not add punctuation or capital letters. "
    "Keep Arabic words as they are spoken. If a part is unclear or meaningless, leave that part exactly as it is. "
    "Reply with the corrected text only."
)
# Few-shot examples taken from this project's transcripts; the last one shows leaving garble alone.
EXAMPLES = [
    ("kuma ya kasance an yi intizain waɗansuharuffa ne daga kalmobi guda biyu",
     "kuma ya kasance an yi intizain waɗansu haruffa ne daga kalmomi guda biyu"),
    ("wato da alhamdu da ashukuru akwi babbanci tsakani",
     "wato da alhamdu da ashukuru akwai bambanci tsakani"),
    ("sai mutumya fara karanta wannan kalmomin gudabiu",
     "sai mutum ya fara karanta wannan kalmomin guda biyu"),
    ("nanikadawaniisi owa alakowa", "nanikadawaniisi owa alakowa"),
]

MIN_SIMILARITY = 0.80     # letters of suggestion vs original (spaces, case and punctuation ignored)
MAX_LENGTH_CHANGE = 0.20  # letters added or removed, as a share of the original
HAUSA_TEXT = re.compile(r"^[a-zɓɗƙƴʼ'’ ]+$")


def letters(t):
    """Letters only, hooked letters folded to plain ones: what the recogniser could have heard."""
    t = unicodedata.normalize("NFKD", t.lower().translate(str.maketrans("ɓɗƙƴ", "bdky")))
    return "".join(c for c in t if c.isalpha() and not unicodedata.combining(c))


def tidy(t):
    """Match the transcript's style: lower case, no punctuation except the apostrophe."""
    t = re.sub(r"[.,;:!?\"“”()\[\]-]", " ", t.strip().lower())
    return re.sub(r"\s+", " ", t).strip()


def check(original, suggestion):
    """Return (status, similarity): 'suggested', 'unchanged', or 'rejected: <reason>'."""
    if not suggestion or suggestion == tidy(original):
        return "unchanged", 1.0
    if not HAUSA_TEXT.match(suggestion):
        return "rejected: contains characters that are not Hausa letters", 0.0
    a, b = letters(original), letters(suggestion)
    sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    if abs(len(b) - len(a)) > MAX_LENGTH_CHANGE * max(len(a), 1):
        return "rejected: adds or removes too much", sim
    if sim < MIN_SIMILARITY:
        return "rejected: changes too much", sim
    return "suggested", sim


class LlamaCpp:
    def __init__(self, spec, threads):
        from llama_cpp import Llama
        repo, _, filename = spec.partition(":")
        self.name = spec
        self.llm = Llama.from_pretrained(repo_id=repo, filename=filename, n_ctx=2048,
                                         n_threads=threads, verbose=False)

    def chat(self, messages, max_tokens):
        r = self.llm.create_chat_completion(messages=messages, temperature=0, max_tokens=max_tokens)
        return r["choices"][0]["message"]["content"]


class Transformers:
    def __init__(self, model_id, threads):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        token = os.environ.get("HF_TOKEN")
        self.name, self.torch = model_id, torch
        self.tok = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype="auto",
                                                          device_map="auto").eval()

    def chat(self, messages, max_tokens):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def messages_for(text):
    m = [{"role": "system", "content": SYSTEM}]
    for q, a in EXAMPLES:
        m += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return m + [{"role": "user", "content": text}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", help="player/data/<id>.json")
    ap.add_argument("--out", help="default: work/<id>/suggestions.json")
    ap.add_argument("--gguf", default=DEFAULT_GGUF, help="REPO:FILE of a GGUF on Hugging Face (default %(default)s)")
    ap.add_argument("--hf", help="use these transformers weights instead, e.g. NCAIR1/N-ATLaS")
    ap.add_argument("--from", dest="t0", type=float, default=0, help="only segments starting at or after (s)")
    ap.add_argument("--to", dest="t1", type=float, default=float("inf"), help="only segments starting before (s)")
    ap.add_argument("--flagged-only", action="store_true", help="only segments flagged as uncertain")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new segments")
    a = ap.parse_args()

    doc = json.load(open(a.transcript))
    lid = doc["lecture"]
    out = a.out or os.path.join(ROOT, "work", lid, "suggestions.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    prev = json.load(open(out)) if os.path.exists(out) else {"lecture": lid, "items": []}
    done = {(round(i["start"], 2), i["original"]) for i in prev["items"]}

    todo = [s for s in doc["segments"]
            if s["type"] == "speech" and not s.get("reviewed") and a.t0 <= s["start"] < a.t1
            and (s.get("review") or not a.flagged_only) and len(letters(s.get("text", ""))) >= 4
            and (round(s["start"], 2), s["text"]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[suggest] {len(todo)} segments to check ({len(done)} done before)", flush=True)
    if not todo:
        return

    threads = os.cpu_count() or 4
    llm = Transformers(a.hf, threads) if a.hf else LlamaCpp(a.gguf, threads)
    prev["model"] = llm.name
    t_start, counts = time.time(), {}
    for n, s in enumerate(todo, 1):
        raw = llm.chat(messages_for(s["text"]), max_tokens=len(s["text"]) // 2 + 40)
        suggestion = tidy(raw.splitlines()[0] if raw.strip() else "")
        status, sim = check(s["text"], suggestion)
        counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
        prev["items"].append({"start": s["start"], "end": s["end"], "original": s["text"],
                              "suggestion": suggestion, "status": status, "similarity": round(sim, 3)})
        prev["items"].sort(key=lambda i: i["start"])
        prev["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        tmp = out + ".tmp"  # save after every segment so an interrupted run can continue
        json.dump(prev, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, out)
        mark = {"suggested": "→", "unchanged": "="}.get(status, "✗")
        print(f"[suggest] {n}/{len(todo)} {s['start']:7.1f}  {s['text'][:60]}\n"
              f"          {mark} {suggestion[:60] if status == 'suggested' else status}", flush=True)
    print(f"[suggest] {counts} in {time.time() - t_start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
