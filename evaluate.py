#!/usr/bin/env python3
"""Measure whichmodel honestly, then train the shipped model on all data.

    python evaluate.py            # writes results/accuracy.{md,json} and whichmodel/model.json

Tests (all splits are by PROMPT, so no prompt is seen in both train and test):
  1. 5-fold CV, prompt-grouped, all features        -> headline accuracy
  2. same, style-only / words-only                  -> which half carries the signal
  3. leave-one-MODEL-out                            -> does it recognise a family
                                                       member it has never seen?
  4. truncated test texts (first N words)           -> how short is too short
  5. majority-class baseline
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from whichmodel import features as F  # noqa: E402
from whichmodel.classifier import train  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def truncate(text: str, n: int) -> str:
    # Keep layout (newlines, markdown) for the first n words.
    count, out = 0, []
    for tok in text.split(" "):
        out.append(tok)
        count += len(F.words(tok))
        if count >= n:
            break
    return " ".join(out)


def folds_by_prompt(rows: list[dict], k: int, seed: int = 0) -> list[set[int]]:
    pids = sorted({r["prompt_id"] for r in rows})
    random.Random(seed).shuffle(pids)
    return [set(pids[i::k]) for i in range(k)]


def score(model, rows: list[dict], cut: int | None = None) -> tuple[int, list[tuple[str, str]]]:
    pairs = []
    for r in rows:
        t = truncate(r["text"], cut) if cut else r["text"]
        p = model.predict_proba(t)
        pairs.append((r["family"], max(p, key=p.get)))
    return sum(a == b for a, b in pairs), pairs


def cv(rows, k=5, cuts=(None,), **kw):
    """Prompt-grouped CV. Returns {cut: (correct, total, pairs)}."""
    res = {c: [0, 0, []] for c in cuts}
    for test_p in folds_by_prompt(rows, k):
        tr = [r for r in rows if r["prompt_id"] not in test_p]
        te = [r for r in rows if r["prompt_id"] in test_p]
        m = train([r["text"] for r in tr], [r["family"] for r in tr], **kw)
        for c in cuts:
            ok, pairs = score(m, te, c)
            res[c][0] += ok
            res[c][1] += len(te)
            res[c][2] += pairs
    return res


def lomo(rows, **kw):
    """Hold out one whole model AND a fold of prompts; train on the rest."""
    out = {}
    models = sorted({r["model"] for r in rows})
    fam_models = defaultdict(set)
    for r in rows:
        fam_models[r["family"]].add(r["model"])
    for held in models:
        fam = next(f for f, ms in fam_models.items() if held in ms)
        if len(fam_models[fam]) < 2:
            continue  # family would vanish from training
        ok = tot = 0
        guesses = Counter()
        for test_p in folds_by_prompt(rows, 5):
            tr = [r for r in rows if r["model"] != held and r["prompt_id"] not in test_p]
            te = [r for r in rows if r["model"] == held and r["prompt_id"] in test_p]
            m = train([r["text"] for r in tr], [r["family"] for r in tr], **kw)
            c, pairs = score(m, te)
            ok += c
            tot += len(te)
            guesses.update(g for _, g in pairs)
        out[held] = {"family": fam, "correct": ok, "total": tot, "guesses": dict(guesses)}
    return out


def confusion(pairs, classes):
    m = {a: Counter() for a in classes}
    for a, b in pairs:
        m[a][b] += 1
    return m


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "samples.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--model-out", default=str(ROOT / "whichmodel" / "model.json"))
    ap.add_argument("--quick", action="store_true", help="skip the slow ablations")
    args = ap.parse_args(argv)

    rows = load(Path(args.data))
    classes = sorted({r["family"] for r in rows})
    by_model = Counter(r["model"] for r in rows)
    fam_n = Counter(r["family"] for r in rows)
    majority = max(fam_n.values()) / len(rows)
    print(f"{len(rows)} samples, {len(classes)} families: {dict(fam_n)}", flush=True)

    cuts = (None, 200, 100, 50, 25)
    full = cv(rows, cuts=cuts)
    print("full CV done", flush=True)
    abl = {}
    if not args.quick:
        abl["style only"] = cv(rows, use_words=False)[None]
        abl["words only"] = cv(rows, use_style=False)[None]
        print("ablations done", flush=True)
    lo = lomo(rows)
    print("leave-one-model-out done", flush=True)

    ok, tot, pairs = full[None]
    conf = confusion(pairs, classes)
    per_fam = {f: (conf[f][f], sum(conf[f].values())) for f in classes}

    md = ["# whichmodel v0: measured accuracy", "",
          f"Data: {len(rows)} answers to {len({r['prompt_id'] for r in rows})} prompts from "
          f"{len(by_model)} local models in {len(classes)} families. All splits are by prompt "
          f"(5 folds): the test prompts were never seen in training.", "",
          "| Test | Accuracy | n |", "|---|---|---|",
          f"| Majority-class baseline | {100 * majority:.1f}% | {len(rows)} |",
          f"| **Full model, held-out prompts** | **{pct(ok, tot)}** | {tot} |"]
    for name, (a, b, _) in abl.items():
        md.append(f"| {name.capitalize()}, held-out prompts | {pct(a, b)} | {b} |")
    lo_ok = sum(v["correct"] for v in lo.values())
    lo_tot = sum(v["total"] for v in lo.values())
    md.append(f"| **Unseen model** (leave-one-model-out, held-out prompts) | **{pct(lo_ok, lo_tot)}** | {lo_tot} |")
    md += ["", "## Per family (full model, held-out prompts)", "",
           "| True \\ Guessed | " + " | ".join(classes) + " | Recall |",
           "|---" * (len(classes) + 2) + "|"]
    for f in classes:
        md.append(f"| {f} | " + " | ".join(str(conf[f][g]) for g in classes)
                  + f" | {pct(*per_fam[f])} |")
    md += ["", "## Unseen model: can it place a model it never trained on?", "",
           "| Held-out model | True family | Accuracy | What it guessed |", "|---|---|---|---|"]
    for mname, v in lo.items():
        g = ", ".join(f"{k} {n}" for k, n in sorted(v["guesses"].items(), key=lambda kv: -kv[1]))
        md.append(f"| {mname} | {v['family']} | {pct(v['correct'], v['total'])} | {g} |")
    md += ["", "## Length: accuracy when the test text is cut to its first N words", "",
           "| Words | Accuracy |", "|---|---|"]
    for c in cuts:
        a, b, _ = full[c]
        md.append(f"| {c or 'full (median ' + str(sorted(len(F.words(r['text'])) for r in rows)[len(rows) // 2]) + ')'} | {pct(a, b)} |")

    # Final model on everything, plus its strongest features per family.
    final = train([r["text"] for r in rows], [r["family"] for r in rows])
    final.meta = {"samples": len(rows), "models": dict(by_model), "cv_accuracy": ok / tot,
                  "unseen_model_accuracy": lo_ok / lo_tot if lo_tot else None}
    final.save(Path(args.model_out))
    md += ["", "## Strongest fingerprints per family (final model weights)", ""]
    for c, fam in enumerate(final.classes):
        others = [final.W[k] for k in range(len(final.classes)) if k != c]
        rel = [(final.W[c][i] - sum(o[i] for o in others) / len(others), n)
               for i, n in enumerate(final.names)]
        rel.sort(reverse=True)
        md.append(f"- **{fam}**: " + ", ".join(F.describe(n) for _, n in rel[:8]))

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "accuracy.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "accuracy.json").write_text(json.dumps({
        "samples": len(rows), "families": dict(fam_n), "models": dict(by_model),
        "majority_baseline": majority,
        "cv": {str(c): full[c][0] / full[c][1] for c in cuts},
        "ablations": {k: a / b for k, (a, b, _) in abl.items()},
        "leave_one_model_out": lo,
        "confusion": {f: dict(conf[f]) for f in classes},
    }, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
