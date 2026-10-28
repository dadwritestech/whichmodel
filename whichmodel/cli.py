"""whichmodel FILE  -> per-family probabilities and the features behind them."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import features as F
from .classifier import Model

DEFAULT_MODEL = Path(__file__).resolve().parent / "model.json"
SHORT = 100  # words; below this the guess is mostly noise (see README)


def fmt_raw(name: str, raw: float) -> str:
    if name.startswith("word:") or "_lines" in name or name.startswith("style:open") \
            or name.startswith("style:close") or name.endswith("_frac"):
        return ""
    return f" ({raw:.2f})"


def render(r: dict, classes: list[str]) -> str:
    out = []
    probs = sorted(r["probabilities"].items(), key=lambda kv: -kv[1])
    for fam, p in probs:
        bar = "#" * round(p * 30)
        out.append(f"  {fam:<8} {p * 100:5.1f}%  {bar}")
    out.append("")
    out.append(f"Leaning {r['top']} because of:")
    for name, raw, c in r["for"]:
        out.append(f"  + {F.describe(name)}{fmt_raw(name, raw)}")
    if r["against"]:
        out.append("Pulling the other way:")
        for name, raw, c in r["against"]:
            out.append(f"  - {F.describe(name)}{fmt_raw(name, raw)}")
    out.append("")
    if r["words"] < SHORT:
        out.append(f"! Only {r['words']} words. Short texts are close to a coin flip.")
    out.append(f"Known families: {', '.join(classes)}. Text from anything else (a person, Claude, GPT...)")
    out.append("is still forced into one of these. This is a probability, not a verdict on anyone.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="whichmodel", description=__doc__)
    ap.add_argument("file", nargs="?", help="text file to check ('-' or empty for stdin)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="trained model.json")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--top", type=int, default=6, help="how many features to show")
    args = ap.parse_args(argv)

    if args.file in (None, "-"):
        text = sys.stdin.read()
    else:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not F.words(text):
        print("whichmodel: no words in input", file=sys.stderr)
        return 2
    if not Path(args.model).exists():
        print(f"whichmodel: no trained model at {args.model}; run `python evaluate.py` first",
              file=sys.stderr)
        return 2

    m = Model.load(Path(args.model))
    r = m.explain(text, top=args.top)
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(render(r, m.classes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
