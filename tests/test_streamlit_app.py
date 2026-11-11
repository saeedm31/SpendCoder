"""Smoke test for the control panel: every page renders without errors (no data, no model, no AWS).
Skipped when Streamlit is not installed."""

import importlib.util
import os
import unittest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "streamlit_app.py")
PAGES = ["Overview", "1 · Data", "2 · Train", "3 · Evaluate", "4 · Classify", "5 · Review queue", "6 · Deploy"]


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "streamlit not installed")
class StreamlitAppTests(unittest.TestCase):
    def test_every_page_renders(self):
        from streamlit.testing.v1 import AppTest

        os.environ["MODEL_REPO"] = "nobody/does-not-exist-spendcoder"
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertFalse(at.exception, at.exception)
        for page in PAGES:
            at.sidebar.radio[0].set_value(page).run()
            self.assertFalse(at.exception, f"{page}: {at.exception}")


if __name__ == "__main__":
    unittest.main()
