# whichmodel

Paste a text, get an estimate of *which* model family wrote it, plus the
features that drove the guess. It does not answer "AI or human". It answers
"gemma 72%, qwen 21%, mistral 7%, and here is why".

The example below is a fresh gemma-4-12b LinkedIn post, from a prompt that is not in the training set:

```
$ whichmodel post.txt
  gemma     89.7%  ###########################
  qwen       9.9%  ###
  mistral    0.4%

Leaning gemma because of:
  + uses the word "team"
  + hr lines
  + straight apostrophe (1.30)
  ...
Known families: gemma, mistral, qwen. Text from anything else (a person, Claude, GPT...)
is still forced into one of these. This is a probability, not a verdict on anyone.
```

## Honest accuracy (v0)

Read this before you believe any output. Data: 600 answers to the same 100
prompts from 6 local models in 3 families. Every split is **by prompt**, so a
test prompt was never seen in training. Full tables are in
[results/accuracy.md](results/accuracy.md).

| Test | Accuracy | Chance |
|---|---|---|
| **Family, models it was trained on** (held-out prompts) | **66.8%** | 33.3% |
| Style features only | 66.7% | 33.3% |
| Word features only | 60.5% | 33.3% |
| **Family, a model it has never seen** (leave-one-model-out) | **31.8%** | 33.3% |
| Exact model, 6-way | 49.0% | 16.7% |
| Family, first 100 words only | 63.3% | 33.3% |
| Family, first 25 words only | 51.8% | 33.3% |

What this means:

1. **It recognises models it has seen, and does so well above chance.** It does
   not yet recognise a *family*. When I hold out one whole model, such as
   gemma-4-12b, and train on its sibling (gemma-4-e4b), accuracy falls to
   chance. On this data, a fingerprint belongs to one model and its post-training.
   It is not shared by a vendor's whole family. Stronger regularisation and style-only
   variants did not help (25–31%).
2. **The style features carry the signal.** The 75 hand-written features alone
   match the full model, so a word list adds almost nothing.
3. **Short text is weak.** At 25 words (about one tweet) it is right about half the time.
4. **The probabilities are calibrated.** Raw logistic regression said "90%+" on
   71% of texts and was right on only 75% of those. The shipped model divides
   logits by T = 3.75, which was fit on out-of-fold predictions. After that,
   "80%" is right 81% of the time and "90%+" is right 88% of the time. (T is one
   parameter, fit and checked on the same out-of-fold set, so the calibration
   numbers are slightly optimistic.)

### What v0 does NOT cover (yet)

- **No Claude, GPT, or Gemini.** No API keys were available when this was built,
  so the classes are the local open-weight families: Gemma (Google), Qwen
  (Alibaba), and Mistral. The adapters exist in `build_dataset.py`; set the key
  and re-run.
- **No Llama** weights were on the machine.
- **No human class.** v0 has no human-written corpus. The CLI forces every text into a
  model family, so a human-written text *will* get a confident-looking label.
  `build_dataset.py --add-human DIR` imports one.
- **Small, quantised models.** These are 3B–35B GGUF models, not frontier APIs.
  The published work on API models
  ([Idiosyncrasies in LLMs](https://github.com/locuslab/llm-idiosyncrasies)) reports about 97%
  for five chat APIs, using fine-tuned text embeddings. That is a stronger model
  on stronger-fingerprinted data, and it does not explain its guesses.
- "mistral" includes Hearthfire-24B, a community finetune. I assume it is built on
  Mistral-Small-24B (its GGUF only says `llama` architecture). The other Mistral,
  Ministral-3-3B-Reasoning, was dropped because its reasoning used up the whole
  token budget and it returned empty answers.

### Measured habits (mean per answer)

Rates are per 100 words; `*_lines` values are fractions of non-empty lines.

| Model | em-dash | bullets | headings | bold | `---` rules | curly quotes | opens "Sure" | closes with an offer | words/sentence |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-12b | 0.23 | 0.26 | 0.16 | 2.80 | 0.04 | 0.65 | 0% | 1% | 12.4 |
| gemma-4-e4b | 0.27 | 0.21 | 0.15 | 2.98 | 0.07 | 0.25 | 0% | 0% | 12.4 |
| qwen3.6-35b-a3b | 0.16 | 0.29 | 0.15 | 3.36 | 0.01 | 1.42 | 0% | 3% | 11.8 |
| qwen3.8-27b | 0.14 | 0.31 | 0.13 | 3.23 | 0.01 | 1.60 | 0% | 1% | 12.1 |
| ministral-3-3b | 0.81 | 0.34 | 0.11 | 5.17 | 0.07 | 2.13 | 2% | 15% | 11.6 |
| hearthfire-24b | 0.50 | 0.28 | 0.09 | 1.66 | 0.01 | 0.70 | 9% | 5% | 12.5 |

The two Mistral-family models differ from each other about as much as they differ from the
other families. The leave-one-model-out result above is the same fact.

## Ethics: read before you screenshot

whichmodel produces a **probability from a small statistical model**. It is
never a verdict about a person.

- It cannot prove that anyone used AI. A person who writes in bullets with bold
  labels will look like a model, because models learned that style from people.
- On text from a family it does not know, including all human text in v0, it
  will still name one of the families it knows.
- False accusations harm real people: students, job applicants, and
  non-native writers, whose style is often more formulaic. Do not use this to
  grade, hire, fire, or publicly accuse anyone.
- Fingerprints drift with every model release. Treat a model trained in one month as
  stale by the next.

## Use

Python 3.10+, no dependencies.

```bash
pip install -e .
```

```bash
whichmodel some.txt
```

```bash
cat some.txt | whichmodel --json
```

## Rebuild the dataset and the model

```bash
python build_dataset.py
```

```bash
python evaluate.py
```

`build_dataset.py` sends `data/prompts.json` to every entry in
`data/models.json`, with no system prompt (the target is each model's default voice),
temperature 0.8, and at most 400 tokens. It resumes where it stopped. Keys come
from env vars only: `LLAMA_API_KEY` / `LLAMA_BASE_URL` (local llama-server), plus
optional `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GEMINI_API_KEY`. Gemma and Qwen
get `chat_template_kwargs.enable_thinking=false`. Without it they spend the budget reasoning.

`evaluate.py` runs every test in the table above, writes `results/`, and
trains the shipped `whichmodel/model.json` on all the data.

How it works: about 75 stylometric features (em-dash, en-dash, and semicolon
rates; bullet, numbered, heading, bold-label, and `---` habits; curly versus straight
quotes; hedges; 32 "signature" phrases such as *delve*, *tapestry*, and *it's important to
note*; openers and closers; sentence-length mean, spread, and tails) plus the 400 most
common words. A pure-Python multinomial logistic regression combines them. The
CLI ranks each feature's push toward the top guess, using its weight
relative to the probability-weighted average weight, times its value.

```bash
python -m unittest discover -s tests
```

## Competition (checked 2026-10-28)

| Repo | Stars | What it is |
|---|---|---|
| lynote-ai/ai-text-detector | 652 | AI-or-human detector, not which model |
| locuslab/llm-idiosyncrasies | 61 | Paper code: which chat API, about 97%, embedding classifier, no tool |
| litemars/LLM-Fingerprinter | 22 | Identifies a model by *querying* it (needs access to the model) |
| tehryanx/whorl | 16 | LLM fingerprinting by probing the model |
| syedamaann/llm-idiosyncrasies | 1 | Reimplementation of the paper |
| harishkotra/which-model-wrote-this | 1 | Guessing game, two models |

Nobody ships an explainable "which model wrote this pasted text" tool. The
closest is academic. Most "fingerprinting" repos need to query the model itself.

## Next

1. Add Claude, GPT, and Gemini (keys) and a human corpus. Without these, the headline use case is not tested.
2. Collect two or more models per vendor, from different sizes and generations, so the classifier can learn
   what a family shares rather than what one checkpoint does.
3. Run the leave-one-model-out test again. It is the number that matters for "which family".
