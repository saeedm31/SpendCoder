"""Tests that run anywhere: no network, no GPU, no PyTorch. The API is replaced by a fake."""

import importlib.util
import unittest

import numpy as np
import pandas as pd

from spendcoder import data, metrics
from spendcoder.baseline import TfidfBaseline
from spendcoder.modelcard import model_card
from spendcoder.predictor import Predictor, decide
from spendcoder.taxonomy import CATEGORIES, LABEL2ID, psc_to_category


class TaxonomyTests(unittest.TestCase):
    def test_codes_map_to_categories(self):
        cases = {"R425": "Engineering & professional services", "J059": "Equipment maintenance & repair",
                 "Z1AA": "Construction & building maintenance", "AJ11": "Research & development",
                 "DA01": "IT & telecom services", "7A21": "IT hardware & software", "7030": "IT hardware & software",
                 "6105": "Electrical & power equipment", "6640": "Lab instruments & chemicals",
                 "4320": "Vehicles & machinery", "7510": "Other goods", "V231": "Other services",
                 " f999 ": "Environmental services"}
        for code, cat in cases.items():
            self.assertEqual(psc_to_category(code), cat, code)
        self.assertIsNone(psc_to_category(None))
        self.assertIsNone(psc_to_category(""))

    def test_every_mapped_category_is_known(self):
        for code in ["R499", "M181", "S206", "F108", "5998", "6810", "2310", "9999"]:
            self.assertIn(psc_to_category(code), LABEL2ID)


class CleaningTests(unittest.TestCase):
    def test_markers_and_spaces_removed(self):
        self.assertEqual(data.clean_text("IGF::OT::IGF  ANNUAL MAINTENANCE  OF HV SWITCHGEAR."),
                         "ANNUAL MAINTENANCE OF HV SWITCHGEAR")
        self.assertEqual(data.clean_text("IGF::CL,CT::IGF engineering support"), "engineering support")
        self.assertEqual(data.clean_text(None), "")
        self.assertEqual(data.clean_text(float("nan")), "")


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, "error"

    def json(self):
        return self.payload


class FakeSession:
    """Two pages of contracts, then the end; the first call is rate limited once."""

    def __init__(self):
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append(json["page"])
        if len(self.calls) == 1:
            return FakeResponse({}, status=429)
        n = 100 if json["page"] == 1 else 30
        results = [{"Award ID": f"A{json['page']}-{i}", "generated_internal_id": f"G{json['page']}-{i}",
                    "Description": "IGF::OT::IGF LAPTOPS", "Award Amount": 1000, "Start Date": "2022-01-05",
                    "PSC": {"code": "7B21", "description": "IT end user devices"}} for i in range(n)]
        return FakeResponse({"results": results, "page_metadata": {"hasNext": json["page"] == 1}})


class FetchTests(unittest.TestCase):
    def test_pagination_retry_and_parsing(self):
        s = FakeSession()
        data.time.sleep = lambda *_: None
        rows = data.fetch_year(2022, session=s, log=lambda m: None)
        self.assertEqual(len(rows), 130)
        self.assertEqual(s.calls, [1, 1, 2])
        df = data.build_dataset(pd.DataFrame(rows))
        self.assertEqual(df["category"].unique().tolist(), ["IT hardware & software"])
        self.assertEqual(df["text"].iloc[0], "LAPTOPS")

    def test_api_error_is_clear(self):
        class Bad:
            def post(self, *a, **k):
                return FakeResponse({}, status=422)
        with self.assertRaises(RuntimeError):
            data.fetch_year(2022, session=Bad(), log=lambda m: None)


def raw_rows():
    rows = []
    for i, (text, psc, start) in enumerate([
        ("IGF::OT::IGF ENGINEERING SUPPORT SERVICES", "R425", "2021-03-01"),
        ("ENGINEERING SUPPORT SERVICES", "R425", "2024-11-01"),      # same text as train: removed from test
        ("HVAC REPAIR BLDG 12", "Z2AA", "2024-12-01"),
        ("DELL LAPTOPS", "7B21", "2023-11-15"),
        ("", "R425", "2022-01-01"),                                 # empty text: dropped
        ("NO CODE", None, "2022-01-01"),                            # no label: dropped
    ]):
        rows.append({"generated_internal_id": f"G{i}", "Award ID": f"A{i}", "Description": text,
                     "PSC": {"code": psc} if psc else None, "Start Date": start, "Award Amount": 10})
    rows.append(dict(rows[0], **{"Start Date": "2021-03-01"}))     # same contract seen in another year
    return pd.DataFrame(rows)


class DatasetTests(unittest.TestCase):
    def test_build_dedupes_and_labels(self):
        df = data.build_dataset(raw_rows())
        self.assertEqual(len(df), 4)
        self.assertTrue(df["award_id"].is_unique)
        self.assertEqual(set(df["label"]), {LABEL2ID[c] for c in df["category"]})

    def test_time_split_without_text_leakage(self):
        train, val, test = data.split(data.build_dataset(raw_rows()))
        self.assertEqual(train["text"].tolist(), ["ENGINEERING SUPPORT SERVICES"])
        self.assertEqual(val["text"].tolist(), ["DELL LAPTOPS"])
        self.assertEqual(test["text"].tolist(), ["HVAC REPAIR BLDG 12"])
        self.assertLess(train["start_date"].max(), val["start_date"].min())
        summary = data.summary(train, val, test)
        self.assertEqual(int(summary.values.sum()), 3)


def synthetic(n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    words = {"IT hardware & software": ["laptop", "server", "monitor", "software license"],
             "Equipment maintenance & repair": ["repair pump", "maintenance of switchgear", "overhaul turbine"],
             "Engineering & professional services": ["engineering support", "consulting services", "program management"]}
    rows = []
    for cat, ws in words.items():
        for _ in range(n_per):
            rows.append({"text": f"{rng.choice(ws)} {rng.integers(1, 99)}", "label": LABEL2ID[cat]})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


class ModelTests(unittest.TestCase):
    def test_baseline_learns_and_returns_all_categories(self):
        df = synthetic()
        m = TfidfBaseline().fit(df["text"], df["label"])
        p = m.predict_proba(["repair pump 7", "laptop 12"])
        self.assertEqual(p.shape, (2, len(CATEGORIES)))
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        self.assertEqual(CATEGORIES[p[0].argmax()], "Equipment maintenance & repair")
        self.assertGreater(metrics.scores(df["label"], m.predict_proba(df["text"]))["accuracy"], 0.95)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 0, 1, 1])
        self.p = np.array([[0.95, 0.05], [0.6, 0.4], [0.3, 0.7], [0.55, 0.45]])
        self.p = np.hstack([self.p, np.zeros((4, len(CATEGORIES) - 2))])

    def test_scores_and_automation(self):
        s = metrics.scores(self.y, self.p)
        self.assertAlmostEqual(s["accuracy"], 0.75)
        rows = {r["threshold"]: r for r in metrics.automation(self.y, self.p, thresholds=(0.5, 0.9))}
        self.assertEqual(rows[0.5]["automated_pct"], 100.0)
        self.assertEqual(rows[0.9]["automated_pct"], 25.0)
        self.assertEqual(rows[0.9]["accuracy_of_automated_pct"], 100.0)

    def test_threshold_reaches_target(self):
        y = np.r_[np.zeros(50, int), np.ones(50, int)]
        conf = np.linspace(0.4, 0.99, 100)
        pred_right = conf > 0.6                       # confident answers are right, unsure ones wrong
        p = np.zeros((100, len(CATEGORIES)))
        for i in range(100):
            k = y[i] if pred_right[i] else 1 - y[i]
            p[i, k], p[i, 1 - k] = conf[i], 1 - conf[i]
        t = metrics.threshold_for_accuracy(y, p, target=0.95)
        self.assertTrue(0.55 <= t <= 0.65, t)

    def test_class_weights_favour_rare_categories(self):
        from spendcoder.finetune import class_weights     # module imports torch only inside functions
        labels = [0] * 90 + [1] * 10
        w = class_weights(labels, n_classes=3)
        self.assertGreater(w[1], w[0])                     # rare category weighs more
        self.assertAlmostEqual(float(w[np.array(labels)].mean()), 1.0)   # average weight per line is 1
        self.assertAlmostEqual(w[1] / w[0], 3.0)           # square root of the 9x imbalance


class FakeEncoding:
    def __init__(self, n):
        self.ids, self.attention_mask = [0] + [5] * n + [2], [1] * (n + 2)


class FakeTokenizer:
    def encode_batch(self, texts):
        width = max(len(t.split()) for t in texts)
        out = []
        for t in texts:
            e = FakeEncoding(len(t.split()))
            pad = width + 2 - len(e.ids)
            e.ids, e.attention_mask = e.ids + [1] * pad, e.attention_mask + [0] * pad
            out.append(e)
        return out


class FakeSessionONNX:
    class Input:
        def __init__(self, name):
            self.name = name

    def get_inputs(self):
        return [self.Input("input_ids"), self.Input("attention_mask")]

    def run(self, outputs, feeds):
        n_words = feeds["attention_mask"].sum(axis=1) - 2
        logits = np.zeros((len(n_words), len(CATEGORIES)))
        logits[np.arange(len(n_words)), np.minimum(n_words, len(CATEGORIES) - 1)] = 6.0   # longer text, other class
        logits[n_words >= 5, :] = 0.0                                                    # long text: unsure
        return [logits]


class PredictorTests(unittest.TestCase):
    def test_decide(self):
        p = np.zeros(len(CATEGORIES))
        p[[2, 5, 7]] = [0.6, 0.3, 0.1]
        d = decide("x", p, CATEGORIES, threshold=0.8)
        self.assertEqual(d["category"], CATEGORIES[2])
        self.assertTrue(d["needs_review"])
        self.assertEqual([s["category"] for s in d["suggestions"]], [CATEGORIES[i] for i in (2, 5, 7)])

    def test_classify_with_onnx_style_session(self):
        pred = Predictor(FakeTokenizer(), FakeSessionONNX(), CATEGORIES, threshold=0.9)
        out = pred.classify(["IGF::OT::IGF ONE", "TWO WORDS", "a b c d e f"])
        self.assertEqual([o["category"] for o in out[:2]], [CATEGORIES[1], CATEGORIES[2]])
        self.assertEqual([o["needs_review"] for o in out], [False, False, True])
        self.assertEqual(pred.probs([]).shape, (0, len(CATEGORIES)))
        texts = ["ONE", "TWO WORDS", "a b c d e f"] * 3
        np.testing.assert_allclose(pred.probs(texts, batch_size=2), pred.probs(texts, batch_size=100))   # batching


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "fastapi not installed")
class ApiTests(unittest.TestCase):
    def test_classify_endpoint(self):
        from fastapi.testclient import TestClient

        from serve import app as service
        service.predictor.cache_clear()
        service.predictor = lambda: Predictor(FakeTokenizer(), FakeSessionONNX(), CATEGORIES, threshold=0.9)
        c = TestClient(service.app)
        r = c.post("/classify", json={"lines": ["ONE", "a b c d e f"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["summary"], {"lines": 2, "needs_review": 1})
        self.assertEqual(c.post("/classify", json={"lines": []}).status_code, 422)
        self.assertEqual(c.post("/classify", json={"lines": ["x" * 501]}).status_code, 400)

    def test_page_and_health(self):
        from fastapi.testclient import TestClient

        from serve import app as service
        service.predictor = lambda: Predictor(FakeTokenizer(), FakeSessionONNX(), CATEGORIES, threshold=0.9)
        c = TestClient(service.app)
        r = c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("SpendCoder", r.text)
        h = c.get("/health").json()
        self.assertEqual((h["status"], h["categories"], h["review_threshold"]), ("ready", len(CATEGORIES), 0.9))
        self.assertEqual(c.get("/health").json()["categories"], len(CATEGORIES))


class ModelCardTests(unittest.TestCase):
    def test_card_renders(self):
        m = {"accuracy": 0.8, "macro_f1": 0.7, "weighted_f1": 0.79}
        res = {"rows": {"train": 1000, "val": 200, "test": 300}, "baseline_test": m, "finetuned_test": m,
               "threshold": 0.9, "automation_test": [{"threshold": 0.9, "automated_pct": 70.0,
                                                      "accuracy_of_automated_pct": 96.0, "to_review_pct": 30.0}]}
        card = model_card(res)
        self.assertIn("pipeline_tag: text-classification", card)
        self.assertIn("70.0% of test lines", card)
        for c in CATEGORIES:
            self.assertIn(c, card)


if __name__ == "__main__":
    unittest.main()
