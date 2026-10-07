"""Build the synthetic report and its downloads for GitHub Pages."""
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trend_timing.backtest import ExecutionConfig
from trend_timing.cli import run_demo


if __name__ == "__main__":
    destination = ROOT / "docs"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary)
        run_demo(generated, ExecutionConfig())
        for source in generated.iterdir():
            target_name = "index.html" if source.name == "report.html" else source.name
            shutil.copyfile(source, destination / target_name)
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    print("Built docs/index.html and synthetic result downloads.")
