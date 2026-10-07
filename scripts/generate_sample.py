"""Explicitly regenerate the bundled synthetic CSV; never downloads data."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trend_timing.data import SAMPLE, generate_sample

if __name__ == "__main__":
    generate_sample().to_csv(SAMPLE, float_format="%.12g")
    print(f"Generated synthetic input: {SAMPLE.name}")
