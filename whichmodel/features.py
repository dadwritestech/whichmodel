"""Interpretable stylometric features.

Every feature has a human-readable name so the CLI can say *why* it guessed.
Two groups:

* style:*  dense, hand-written measures (em-dash rate, bullet habit, ...)
* word:*   learned vocabulary (top words by document frequency at train time),
           scaled per 100 words. Because every model answered the same prompts,
           topic words are balanced across classes and what is left is habit.
"""
from __future__ import annotations

import math
import re
import statistics

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’]*")
SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]”’]*\s+(?=[A-Z0-9\"'(\[“*#-])")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅]")

HEDGES = ["might", "perhaps", "may", "generally", "typically", "often", "usually",
          "arguably", "likely", "somewhat", "potentially", "it depends"]
# Phrases folk wisdom ties to specific assistants. Counted, not assumed: the
# classifier decides whether they mean anything.
SIGNATURES = ["delve", "tapestry", "crucial", "vibrant", "testament", "navigate",
              "landscape", "realm", "embark", "foster", "leverage", "robust",
              "seamless", "nuanced", "intricate", "pivotal", "honestly", "genuinely",
              "ultimately", "essentially", "absolutely", "certainly", "fantastic",
              "important to note", "keep in mind", "in summary", "in conclusion",
              "let's break", "here's", "it's not just", "game-changer", "dive into"]
OPENERS = {
    "open_sure": r"^(sure|certainly|absolutely|of course)\b",
    "open_great": r"^(great|excellent|what a (great|wonderful|fantastic)) (question|idea|choice)",
    "open_heres": r"^here(?:'|’)?s\b|^here is\b|^here are\b",
    "open_heading": r"^#",
    "open_ah_okay": r"^(ah|oh|okay|ok|alright|so)\b",
    "open_title_bold": r"^\*\*",
}
CLOSERS = {
    "close_question": r"\?\s*$",
    "close_offer": r"(would you like|do you want|let me know|happy to help|feel free|i can also|shall i)",
    "close_hope": r"(i hope (this|that) helps|hope this helps|good luck|best of luck|you've got this|you got this)",
}


def words(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in WORD_RE.findall(text)]


def strip_md(text: str) -> str:
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"[*_`#>|]+", " ", t)
    return t


def sentences(text: str) -> list[str]:
    out = []
    for line in strip_md(text).splitlines():
        line = line.strip(" -\t")
        if not line:
            continue
        out.extend(s for s in SENT_SPLIT.split(line) if len(words(s)) >= 2)
    return out


def _count(pattern: str, text: str, flags: int = 0) -> int:
    return len(re.findall(pattern, text, flags))


def style_features(text: str) -> dict[str, float]:
    ws = words(text)
    nw = max(len(ws), 1)
    per100 = 100.0 / nw
    lines = [l for l in text.splitlines() if l.strip()]
    nl = max(len(lines), 1)
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sents = sentences(text)
    slens = [len(words(s)) for s in sents] or [nw]
    low = text.lower()
    first = low.lstrip()[:120]
    last = low.rstrip()[-250:]

    f: dict[str, float] = {
        "em_dash": text.count("—") * per100,
        "en_dash": text.count("–") * per100,
        "double_hyphen": _count(r"(?<!-)--(?!-)", text) * per100,
        "spaced_hyphen": _count(r"\w - \w", text) * per100,
        "semicolon": text.count(";") * per100,
        "colon": text.count(":") * per100,
        "exclamation": text.count("!") * per100,
        "question": text.count("?") * per100,
        "ellipsis": (text.count("...") + text.count("…")) * per100,
        "parenthesis": text.count("(") * per100,
        "curly_quotes": (text.count("’") + text.count("“")) * per100,
        "straight_apostrophe": _count(r"\w'\w", text) * per100,
        "emoji": len(EMOJI_RE.findall(text)) * per100,
        "bold": _count(r"\*\*[^*\n]+\*\*", text) * per100,
        "italic": _count(r"(?<![*\w])\*[^*\n]+\*(?![*\w])", text) * per100,
        "heading_lines": _count(r"^#{1,6} ", text, re.M) / nl,
        "h3_lines": _count(r"^### ", text, re.M) / nl,
        "bullet_lines": _count(r"^\s*[-*•] ", text, re.M) / nl,
        "numbered_lines": _count(r"^\s*\d+[.)] ", text, re.M) / nl,
        "bold_label_lines": _count(r"^\s*(?:[-*]\s+|\d+[.)]\s+)?\*\*[^*\n]+:?\*\*:?", text, re.M) / nl,
        "hr_lines": _count(r"^\s*(---|\*\*\*|___)\s*$", text, re.M) / nl,
        "table_lines": _count(r"^\s*\|.*\|\s*$", text, re.M) / nl,
        "code_blocks": text.count("```") / 2 * per100,
        "paragraphs_per_100w": len(paras) * per100,
        "log_words": math.log(nw),
        "sent_len_mean": statistics.fmean(slens),
        "sent_len_sd": statistics.pstdev(slens),
        "sent_short_frac": sum(1 for s in slens if s <= 8) / len(slens),
        "sent_long_frac": sum(1 for s in slens if s >= 30) / len(slens),
        "word_len_mean": statistics.fmean(len(w) for w in ws) if ws else 0.0,
        "type_token": len(set(ws[:200])) / max(len(ws[:200]), 1),
        "first_person_i": sum(1 for w in ws if w in ("i", "i'm", "i've", "i'd", "my")) * per100,
        "second_person": sum(1 for w in ws if w in ("you", "your", "you're", "yours")) * per100,
        "we_us": sum(1 for w in ws if w in ("we", "us", "our", "let's")) * per100,
        "contractions": sum(1 for w in ws if "'" in w) * per100,
        "hedges": sum(low.count(h) for h in HEDGES if " " in h) * per100
                  + sum(1 for w in ws if w in HEDGES) * per100,
    }
    for sig in SIGNATURES:
        key = "sig_" + re.sub(r"\W+", "_", sig).strip("_")
        f[key] = (low.count(sig) if " " in sig or "'" in sig else ws.count(sig)) * per100
    for k, pat in OPENERS.items():
        f[k] = 1.0 if re.search(pat, first) else 0.0
    for k, pat in CLOSERS.items():
        f[k] = 1.0 if re.search(pat, last) else 0.0
    return {"style:" + k: v for k, v in f.items()}


def word_counts(text: str) -> dict[str, int]:
    c: dict[str, int] = {}
    for w in words(text):
        c[w] = c.get(w, 0) + 1
    return c


def build_vocab(texts: list[str], size: int = 400, min_df: int = 5) -> list[str]:
    """Top words by document frequency, ignoring ones that are nearly universal."""
    df: dict[str, int] = {}
    for t in texts:
        for w in set(words(t)):
            df[w] = df.get(w, 0) + 1
    n = len(texts)
    keep = [w for w, d in df.items() if min_df <= d <= 0.9 * n]
    keep.sort(key=lambda w: (-df[w], w))
    return sorted(keep[:size])


def extract(text: str, vocab: list[str] | None = None) -> dict[str, float]:
    f = style_features(text)
    if vocab:
        counts = word_counts(text)
        nw = max(sum(counts.values()), 1)
        for w in vocab:
            if w in counts:
                f["word:" + w] = math.log1p(counts[w] * 100.0 / nw)
    return f


def describe(name: str) -> str:
    """Short label for CLI output."""
    kind, _, key = name.partition(":")
    if kind == "word":
        return f'uses the word "{key}"'
    if key.startswith("sig_"):
        return f'signature phrase "{key[4:].replace("_", " ")}"'
    return key.replace("_", " ")
