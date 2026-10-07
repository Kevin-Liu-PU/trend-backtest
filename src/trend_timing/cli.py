"""Run the local synthetic demonstration and write reviewable artifacts."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from . import __version__
from .backtest import ExecutionConfig, run_backtest
from .data import SAMPLE, SEED, load_sample
from .metrics import summary
from .report import build_html
from .signals import buy_hold_weights, sma_weights
from .validation import evaluate_windows, stress_test, walk_forward


def source_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode()); h.update(b"\0"); h.update(path.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def run_demo(output: Path, config: ExecutionConfig):
    panel = load_sample()
    fixed = run_backtest(panel, sma_weights(panel, 200), config=config)
    benchmark = run_backtest(panel, buy_hold_weights(panel), config=config)
    wf = walk_forward(panel, config=config)
    returns = pd.DataFrame({"Fixed SMA200": fixed.returns, "Buy and hold": benchmark.returns,
                            "Walk-forward": wf.result.returns}).loc[wf.evaluation_start:]
    metrics = pd.DataFrame([{"strategy": name, **summary(returns[name], panel.cash_ret.loc[returns.index])}
                            for name in returns])
    stress = stress_test(panel, config=config)
    surface = evaluate_windows(panel, (150, 200, 250), config=config)
    manifest = {"project": "trend-backtest", "version": __version__,
                "synthetic_data_only": True, "execution": asdict(config),
                "dataset": {"file": "packaged sample.csv", "seed": SEED, "sessions": len(panel),
                            "sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
                            "generator": "row-wise Gaussian draws; fixed manufactured regime schedule"},
                "evaluation": {"start": str(returns.index[0].date()), "end": str(returns.index[-1].date()),
                               "initialization": "slice of continuous simulations; WF starts in cash"},
                "walk_forward": {"windows": [150, 200, 250], "initial_train_bars": 756,
                                 "test_bars": 252, "metric": "training excess Sharpe", "folds": len(wf.folds)},
                "source_sha256": source_hash(),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
                "limits": ["synthetic mechanics only", "costs deducted from daily return as approximation",
                           "one-hot unleveraged holdings", "no broker, tax, real data or predictive claims"]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame, index in [("metrics.csv", metrics, False), ("daily_returns.csv", returns, True),
                               ("walk_forward_folds.csv", wf.folds, False), ("stress.csv", stress, False),
                               ("parameter_surface.csv", surface, False)]:
        frame.to_csv(output / name, index=index, float_format="%.12g")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.html").write_text(build_html(metrics, returns, wf.folds, stress, surface, manifest), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline synthetic trend-timing demonstration")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--fill", choices=("open", "close"), default="open", help="close is the legacy return approximation")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="one-way cost; a full switch has two legs")
    parser.add_argument("--lag", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        manifest = run_demo(args.output, ExecutionConfig(args.fill, args.cost_bps, args.lag))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print("SYNTHETIC DATA ONLY — no investment-performance evidence.")
    print(f"Executed fixed SMA200, benchmark, {manifest['walk_forward']['folds']} walk-forward folds, 6 stress cases and 3 parameter cases.")
    print(f"Fill={args.fill}; cost={args.cost_bps:g} bp per side; lag={args.lag}.")
    print(f"Report: {(args.output / 'report.html').resolve()}")
    return 0
