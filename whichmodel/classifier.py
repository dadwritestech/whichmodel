"""Multinomial logistic regression in pure Python.

Small on purpose: a few hundred named features, a weight per (class, feature),
nothing hidden. Features are divided by their training standard deviation (not
centred, so absent words stay zero and inputs stay sparse).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from . import features as F


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


class Model:
    def __init__(self, classes: list[str], names: list[str], scale: list[float],
                 W: list[list[float]], b: list[float], vocab: list[str], meta: dict | None = None):
        self.classes, self.names, self.scale = classes, names, scale
        self.W, self.b, self.vocab = W, b, vocab
        self.meta = meta or {}
        self.index = {n: i for i, n in enumerate(names)}

    # ------------------------------------------------------------ inference
    def vectorize(self, feats: dict[str, float]) -> list[tuple[int, float]]:
        out = []
        for k, v in feats.items():
            i = self.index.get(k)
            if i is not None and v:
                out.append((i, max(-10.0, min(10.0, v / self.scale[i]))))
        return out

    def _logits(self, x: list[tuple[int, float]]) -> list[float]:
        return [self.b[c] + sum(self.W[c][i] * v for i, v in x) for c in range(len(self.classes))]

    def predict_proba(self, text: str) -> dict[str, float]:
        p = softmax(self._logits(self.vectorize(F.extract(text, self.vocab))))
        return dict(zip(self.classes, p))

    def explain(self, text: str, top: int = 6) -> dict:
        feats = F.extract(text, self.vocab)
        x = self.vectorize(feats)
        p = softmax(self._logits(x))
        best = max(range(len(p)), key=p.__getitem__)
        # How much each feature raises log p(best): its weight for `best` minus
        # the probability-weighted average weight, times its value.
        contrib = []
        for i, v in x:
            avg = sum(p[c] * self.W[c][i] for c in range(len(p)))
            contrib.append((v * (self.W[best][i] - avg), self.names[i], feats[self.names[i]]))
        contrib.sort(reverse=True)
        return {
            "probabilities": dict(zip(self.classes, p)),
            "top": self.classes[best],
            "for": [(n, raw, round(c, 3)) for c, n, raw in contrib[:top] if c > 0],
            "against": [(n, raw, round(c, 3)) for c, n, raw in sorted(contrib)[:top // 2] if c < 0],
            "words": len(F.words(text)),
        }

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "classes": self.classes, "names": self.names, "scale": self.scale,
            "W": [[round(w, 5) for w in row] for row in self.W],
            "b": [round(v, 5) for v in self.b], "vocab": self.vocab, "meta": self.meta,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Model":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["classes"], d["names"], d["scale"], d["W"], d["b"], d["vocab"], d.get("meta"))


def train(texts: list[str], labels: list[str], *, use_style: bool = True, use_words: bool = True,
          vocab_size: int = 400, l2: float = 1e-3, epochs: int = 40, lr: float = 0.05,
          seed: int = 0) -> Model:
    classes = sorted(set(labels))
    vocab = F.build_vocab(texts, vocab_size) if use_words else []
    rows = []
    for t in texts:
        f = F.extract(t, vocab)
        if not use_style:
            f = {k: v for k, v in f.items() if not k.startswith("style:")}
        rows.append(f)
    names = sorted({k for r in rows for k in r})
    index = {n: i for i, n in enumerate(names)}

    # Root-mean-square scaling (no centring keeps the inputs sparse).
    sq = [0.0] * len(names)
    for r in rows:
        for k, v in r.items():
            sq[index[k]] += v * v
    scale = [math.sqrt(s / len(rows)) or 1.0 for s in sq]

    m = Model(classes, names, scale, [[0.0] * len(names) for _ in classes], [0.0] * len(classes), vocab)
    X = [m.vectorize(r) for r in rows]
    Y = [classes.index(y) for y in labels]

    # Adam on mini-batches.
    C, D = len(classes), len(names)
    mW = [[0.0] * D for _ in range(C)]
    vW = [[0.0] * D for _ in range(C)]
    mb, vb = [0.0] * C, [0.0] * C
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    rng = random.Random(seed)
    order = list(range(len(X)))
    for _ in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), 32):
            batch = order[s:s + 32]
            gW: list[dict[int, float]] = [{} for _ in range(C)]
            gb = [0.0] * C
            for n in batch:
                p = softmax(m._logits(X[n]))
                for c in range(C):
                    err = p[c] - (1.0 if c == Y[n] else 0.0)
                    gb[c] += err
                    g = gW[c]
                    for i, v in X[n]:
                        g[i] = g.get(i, 0.0) + err * v
            step += 1
            k = 1.0 / len(batch)
            bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
            for c in range(C):
                Wc, mc, vc, g = m.W[c], mW[c], vW[c], gW[c]
                # Lazy L2: only touch weights seen in this batch (plus bias).
                for i, gi in g.items():
                    gi = gi * k + l2 * Wc[i]
                    mc[i] = b1 * mc[i] + (1 - b1) * gi
                    vc[i] = b2 * vc[i] + (1 - b2) * gi * gi
                    Wc[i] -= lr * (mc[i] / bc1) / (math.sqrt(vc[i] / bc2) + eps)
                gbc = gb[c] * k
                mb[c] = b1 * mb[c] + (1 - b1) * gbc
                vb[c] = b2 * vb[c] + (1 - b2) * gbc * gbc
                m.b[c] -= lr * (mb[c] / bc1) / (math.sqrt(vb[c] / bc2) + eps)
    return m
