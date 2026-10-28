import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_dataset  # noqa: E402
import evaluate  # noqa: E402
from whichmodel import cli  # noqa: E402
from whichmodel import features as F  # noqa: E402
from whichmodel.classifier import Model, train  # noqa: E402


def toy_corpus(n=40, seed=0):
    """Two fake 'families' with opposite habits and shared topic words."""
    rng = random.Random(seed)
    topics = ["coffee", "sleep", "running", "taxes", "python", "gardens", "trains", "music"]
    rows = []
    for i in range(n):
        t = rng.choice(topics)
        a = (f"## About {t}\n\n- **Point:** {t} matters — truly.\n- **Also:** it is crucial — "
             f"delve into {t}.\n\nWould you like more on {t}?")
        b = (f"Sure! So {t} is fun. I think {t} is great, honestly. "
             f"You'll like {t}. Good luck!")
        rows.append({"family": "dashy", "model": f"dashy-{i % 2}", "prompt_id": i, "text": a})
        rows.append({"family": "chirpy", "model": f"chirpy-{i % 2}", "prompt_id": i, "text": b})
    return rows


class FeatureTests(unittest.TestCase):
    def test_em_dash_rate_per_100_words(self):
        f = F.style_features("one two three four — five six seven eight nine ten")
        self.assertAlmostEqual(f["style:em_dash"], 10.0)

    def test_list_and_heading_habits(self):
        text = "# Title\n\n- a thing\n- b thing\n1. first\n2. second"
        f = F.style_features(text)
        self.assertGreater(f["style:heading_lines"], 0)
        self.assertAlmostEqual(f["style:bullet_lines"], 2 / 5)
        self.assertAlmostEqual(f["style:numbered_lines"], 2 / 5)

    def test_openers_and_closers(self):
        f = F.style_features("Certainly! Here is the plan. Let me know if you want more.")
        self.assertEqual(f["style:open_sure"], 1.0)
        self.assertEqual(f["style:close_offer"], 1.0)
        self.assertEqual(f["style:open_heading"], 0.0)

    def test_signature_words_and_hedges(self):
        f = F.style_features("Let us delve into it. It might perhaps be crucial.")
        self.assertGreater(f["style:sig_delve"], 0)
        self.assertGreater(f["style:sig_crucial"], 0)
        self.assertGreater(f["style:hedges"], 0)

    def test_sentence_lengths(self):
        f = F.style_features("Short one here. " + "This is a much longer sentence " * 7 + "end.")
        self.assertGreater(f["style:sent_len_sd"], 0)
        self.assertGreater(f["style:sent_short_frac"], 0)

    def test_vocab_drops_rare_and_universal_words(self):
        texts = ["the cat sat"] * 10 + ["the dog ran"] * 10 + ["the unique zebra"]
        v = F.build_vocab(texts, size=10, min_df=5)
        self.assertNotIn("the", v)      # in every document
        self.assertNotIn("zebra", v)    # too rare
        self.assertIn("cat", v)

    def test_empty_text_does_not_crash(self):
        F.extract("", ["word"])


class ClassifierTests(unittest.TestCase):
    def test_learns_separable_styles_on_held_out_text(self):
        rows = toy_corpus()
        tr, te = rows[:60], rows[60:]
        m = train([r["text"] for r in tr], [r["family"] for r in tr], epochs=20)
        ok = sum(max(m.predict_proba(r["text"]).items(), key=lambda kv: kv[1])[0] == r["family"]
                 for r in te)
        self.assertEqual(ok, len(te))

    def test_probabilities_sum_to_one(self):
        rows = toy_corpus(10)
        m = train([r["text"] for r in rows], [r["family"] for r in rows], epochs=5)
        p = m.predict_proba("anything at all — really")
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)

    def test_explain_names_the_driving_features(self):
        rows = toy_corpus()
        m = train([r["text"] for r in rows], [r["family"] for r in rows], epochs=20)
        r = m.explain("## Heading\n\n- **x:** y — z — w.\n- **q:** crucial.")
        self.assertEqual(r["top"], "dashy")
        names = [n for n, _, _ in r["for"]]
        self.assertTrue(any(n.startswith("style:") for n in names), names)

    def test_save_load_roundtrip(self):
        rows = toy_corpus(10)
        m = train([r["text"] for r in rows], [r["family"] for r in rows], epochs=5)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            m.save(p)
            m2 = Model.load(p)
        t = rows[0]["text"]
        for k, v in m.predict_proba(t).items():
            self.assertAlmostEqual(v, m2.predict_proba(t)[k], places=3)


class EvaluateTests(unittest.TestCase):
    def test_folds_never_share_a_prompt(self):
        rows = toy_corpus()
        folds = evaluate.folds_by_prompt(rows, 5)
        seen = set()
        for f in folds:
            self.assertFalse(seen & f)
            seen |= f
        self.assertEqual(seen, {r["prompt_id"] for r in rows})

    def test_truncate_keeps_first_n_words(self):
        self.assertEqual(len(F.words(evaluate.truncate("a b c d e f g", 3))), 3)

    def test_lomo_holds_out_whole_model(self):
        rows = toy_corpus(20)
        res = evaluate.lomo(rows, epochs=5)
        self.assertEqual(set(res), {"dashy-0", "dashy-1", "chirpy-0", "chirpy-1"})
        self.assertEqual(sum(v["total"] for v in res.values()), len(rows))


class DatasetTests(unittest.TestCase):
    def test_resumable_and_skips_optional_without_key(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "p.json").write_text(json.dumps(["q1", "q2"]))
            (d / "m.json").write_text(json.dumps([
                {"family": "fam", "backend": "llama", "model": "local"},
                {"family": "claude", "backend": "anthropic", "model": "c", "optional": True}]))
            calls = []

            def fake(model, prompt, **kw):
                calls.append(prompt)
                return f"answer to {prompt}"

            args = ["--prompts", str(d / "p.json"), "--models-file", str(d / "m.json"),
                    "--out", str(d / "s.jsonl")]
            with mock.patch.dict(build_dataset.BACKENDS, {"llama": (fake, None)}), \
                    mock.patch.dict("os.environ", {}, clear=False) as env, redirect_stdout(io.StringIO()):
                env.pop("ANTHROPIC_API_KEY", None)
                build_dataset.main(args)
                build_dataset.main(args)  # second run: nothing new
            rows = [json.loads(l) for l in (d / "s.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(calls, ["q1", "q2"])
            self.assertEqual({r["family"] for r in rows}, {"fam"})

    def test_llama_request_disables_thinking_and_uses_env_key(self):
        seen = {}

        def fake_post(url, body, headers, timeout=900):
            seen.update(url=url, body=body, headers=headers)
            return {"choices": [{"message": {"content": "hi"}}]}

        with mock.patch.object(build_dataset, "_post", fake_post), \
                mock.patch.dict("os.environ", {"LLAMA_API_KEY": "k", "LLAMA_BASE_URL": "http://h:1"}):
            out = build_dataset.gen_llama("gemma-4-12b", "q", max_tokens=5, temperature=0, seed=1)
        self.assertEqual(out, "hi")
        self.assertEqual(seen["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(seen["headers"]["Authorization"], "Bearer k")
        self.assertEqual(seen["url"], "http://h:1/v1/chat/completions")


class CliTests(unittest.TestCase):
    def test_cli_prints_probabilities_and_features(self):
        rows = toy_corpus()
        m = train([r["text"] for r in rows], [r["family"] for r in rows], epochs=10)
        with tempfile.TemporaryDirectory() as d:
            mp, tp = Path(d) / "m.json", Path(d) / "t.txt"
            m.save(mp)
            tp.write_text("Sure! So coffee is fun. Honestly, good luck!", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main([str(tp), "--model", str(mp)])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("chirpy", out)
        self.assertIn("%", out)
        self.assertIn("because of", out)
        self.assertIn("not a verdict", out)

    def test_cli_json(self):
        rows = toy_corpus(10)
        m = train([r["text"] for r in rows], [r["family"] for r in rows], epochs=5)
        with tempfile.TemporaryDirectory() as d:
            mp, tp = Path(d) / "m.json", Path(d) / "t.txt"
            m.save(mp)
            tp.write_text("## About trains\n\n- **Point:** trains — yes.", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main([str(tp), "--model", str(mp), "--json"])
        r = json.loads(buf.getvalue())
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
