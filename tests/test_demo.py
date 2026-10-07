from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from trend_timing.backtest import ExecutionConfig
from trend_timing.cli import run_demo


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.active = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "iframe"} or (tag == "link" and "href" in attrs):
            self.active.append(tag)
        if tag == "a":
            self.links.append(attrs["href"])


class DemoTests(unittest.TestCase):
    def test_demo_is_deterministic_self_contained_and_labeled(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            manifest = run_demo(first, ExecutionConfig())
            run_demo(second, ExecutionConfig())
            expected = {"report.html", "manifest.json", "metrics.csv", "daily_returns.csv",
                        "walk_forward_folds.csv", "stress.csv", "parameter_surface.csv"}
            self.assertEqual({p.name for p in first.iterdir()}, expected)
            for name in expected:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            self.assertTrue(manifest["synthetic_data_only"])
            self.assertEqual(manifest["execution"]["fill"], "open")
            self.assertEqual(manifest["walk_forward"]["folds"], 5)
            self.assertEqual(json.loads((first / "manifest.json").read_text()), manifest)
            html = (first / "report.html").read_text(encoding="utf-8")
            self.assertIn("Synthetic data only", html)
            parser = ReportParser(); parser.feed(html)
            self.assertFalse(parser.active)
            for link in parser.links:
                self.assertNotIn("://", link)
                self.assertTrue((first / link).is_file())
            metrics = pd.read_csv(first / "metrics.csv")
            self.assertEqual(metrics["sessions"].nunique(), 1)
            self.assertEqual(len(metrics), 3)
            for name in ("stress.csv", "parameter_surface.csv", "walk_forward_folds.csv"):
                self.assertEqual(set(pd.read_csv(first / name)["fill"]), {"open"})


if __name__ == "__main__":
    unittest.main()
