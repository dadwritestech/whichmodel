#!/usr/bin/env python3
"""Send one fixed prompt set to several models and store every answer.

    python build_dataset.py                       # all available models
    python build_dataset.py --models gemma-4-12b  # just one
    python build_dataset.py --samples 2           # two answers per prompt
    python build_dataset.py --add-human DIR       # import human .txt files

Output: data/samples.jsonl, one row per answer:
    {"family", "model", "prompt_id", "sample", "text"}

The run is resumable: rows already in the file are skipped.

Backends and the env vars they read (nothing is hardcoded):
    llama      local llama-server   LLAMA_BASE_URL (default http://localhost:8080), LLAMA_API_KEY
    anthropic  Claude               ANTHROPIC_API_KEY
    openai     GPT                  OPENAI_API_KEY
    gemini     Gemini               GEMINI_API_KEY
A model entry marked "optional" is skipped when its key is missing.

No system prompt is sent. The fingerprint we want is each model's default voice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def _post(url: str, body: dict, headers: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gen_llama(model: str, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
    base = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080").rstrip("/")
    key = os.environ.get("LLAMA_API_KEY")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        # gemma-4 / qwen3 otherwise spend the budget thinking; this is the only
        # switch that reliably turns it off on this server.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = _post(base + "/v1/chat/completions", body,
                 {"Authorization": f"Bearer {key}"} if key else {})
    return data["choices"][0]["message"].get("content") or ""


def gen_anthropic(model: str, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
    data = _post("https://api.anthropic.com/v1/messages",
                 {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                  "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
    return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


def gen_openai(model: str, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
    data = _post("https://api.openai.com/v1/chat/completions",
                 {"model": model, "max_completion_tokens": max_tokens, "seed": seed,
                  "messages": [{"role": "user", "content": prompt}]},
                 {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    return data["choices"][0]["message"].get("content") or ""


def gen_gemini(model: str, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    data = _post(url, {"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}},
                 {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    parts = data["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts)


BACKENDS = {
    "llama": (gen_llama, None),
    "anthropic": (gen_anthropic, "ANTHROPIC_API_KEY"),
    "openai": (gen_openai, "OPENAI_API_KEY"),
    "gemini": (gen_gemini, "GEMINI_API_KEY"),
}


def load_done(path: Path) -> set[tuple[str, int, int]]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["prompt_id"], r["sample"]))
    return done


def available(entry: dict) -> bool:
    _, env = BACKENDS[entry["backend"]]
    return env is None or bool(os.environ.get(env))


def add_human(src: Path, out: Path) -> int:
    """Import human-written texts: one .txt file per sample."""
    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for i, p in enumerate(sorted(src.glob("*.txt"))):
            text = p.read_text(encoding="utf-8").strip()
            if text:
                fh.write(json.dumps({"family": "human", "model": f"human:{src.name}",
                                     "prompt_id": -1 - i, "sample": 0, "text": text,
                                     "source": p.name}, ensure_ascii=False) + "\n")
                n += 1
    print(f"added {n} human samples from {src}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(DATA / "prompts.json"))
    ap.add_argument("--models-file", default=str(DATA / "models.json"))
    ap.add_argument("--models", nargs="*", help="only these model ids")
    ap.add_argument("--out", default=str(DATA / "samples.jsonl"))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=450)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--add-human", metavar="DIR")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.add_human:
        return add_human(Path(args.add_human), out)

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    entries = json.loads(Path(args.models_file).read_text(encoding="utf-8"))
    if args.models:
        entries = [e for e in entries if e["model"] in args.models]
    done = load_done(out)

    for e in entries:
        if not available(e):
            print(f"skip {e['model']}: {BACKENDS[e['backend']][1]} not set")
            continue
        gen = BACKENDS[e["backend"]][0]
        todo = [(pid, s) for pid in range(len(prompts)) for s in range(args.samples)
                if (e["model"], pid, s) not in done]
        print(f"{e['model']} ({e['family']}): {len(todo)} to generate", flush=True)
        t0 = time.time()
        for k, (pid, s) in enumerate(todo, 1):
            try:
                text = gen(e["model"], prompts[pid], max_tokens=args.max_tokens,
                           temperature=args.temperature, seed=1000 * s + pid)
            except (urllib.error.URLError, TimeoutError, KeyError) as err:
                print(f"  ! {e['model']} prompt {pid}: {err}", flush=True)
                continue
            if not text.strip():
                print(f"  ! {e['model']} prompt {pid}: empty answer", flush=True)
                continue
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"family": e["family"], "model": e["model"], "prompt_id": pid,
                                     "sample": s, "text": text.strip()}, ensure_ascii=False) + "\n")
            if k % 10 == 0:
                print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
