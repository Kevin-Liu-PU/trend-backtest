"""Small, explicit-execution validation pipeline adapted from quant.

Every engine call requires the same execution configuration. Walk-forward builds
one causal target stream and executes once, preserving costs at fold boundaries.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from .backtest import BacktestResult, ExecutionConfig, run_backtest
from .metrics import summary
from .signals import sma_weights


def _windows(windows):
    values = tuple(windows)
    if not values or len(set(values)) != len(values) or any(isinstance(x, bool) or not isinstance(x, int) or x < 2 for x in values):
        raise ValueError("windows must be distinct integers >= 2")
    return values


def evaluate_windows(panel, windows, *, config: ExecutionConfig) -> pd.DataFrame:
    """Diagnostic surface on the same full sample, not a selection endorsement."""
    rows = []
    for window in _windows(windows):
        result = run_backtest(panel, sma_weights(panel, window), config=config)
        rows.append({"window": window, **asdict(config), **summary(result.returns, panel.cash_ret)})
    return pd.DataFrame(rows)


def stress_test(panel, window=200, *, config: ExecutionConfig,
                costs=(0.0, 10.0, 25.0), lags=(1, 2)) -> pd.DataFrame:
    weights = sma_weights(panel, window)
    rows = []
    for cost in costs:
        for lag in lags:
            stressed = replace(config, cost_bps=cost, lag=lag)
            result = run_backtest(panel, weights, config=stressed)
            rows.append({"window": window, **asdict(stressed), **summary(result.returns, panel.cash_ret)})
    return pd.DataFrame(rows)


@dataclass
class WalkForwardResult:
    result: BacktestResult
    targets: pd.DataFrame
    folds: pd.DataFrame
    evaluation_start: pd.Timestamp


def walk_forward(panel, windows=(150, 200, 250), *, config: ExecutionConfig,
                 train_bars=756, test_bars=252) -> WalkForwardResult:
    """Expanding training; fixed grid and deterministic first-in-grid tie break.

    A fold is selected from returns strictly before its decision_start. New targets
    begin at that day's close, so their first possible fill is lag bars later.
    Training metrics exclude a common max-window + lag warmup. All pre-training
    targets are cash. Test blocks include the prior holding until a new fill occurs.
    No per-fold liquidation, retroactive target assignment or stitched NAV resets.
    """
    windows = _windows(windows)
    warmup = max(windows) + config.lag
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (train_bars, test_bars)):
        raise ValueError("block sizes must be integers")
    if train_bars < warmup + 2 or test_bars < 1 or train_bars >= len(panel):
        raise ValueError("insufficient training history or invalid test block length")
    weights = {w: sma_weights(panel, w) for w in windows}
    # Full causal series can be computed once; selection slices strictly at cutoff.
    candidates = {w: run_backtest(panel, weights[w], config=config) for w in windows}
    targets = pd.DataFrame({"cash": 1.0, "equity": 0.0}, index=panel.index)
    rows = []
    for start in range(train_bars, len(panel), test_bars):
        stop = min(start + test_bars, len(panel))
        scores = []
        for window in windows:
            r = candidates[window].returns.iloc[warmup:start]
            score = summary(r, panel.cash_ret.iloc[warmup:start])["excess_sharpe"]
            scores.append(score if score is not None and np.isfinite(score) else -np.inf)
        chosen_index = int(np.argmax(scores))
        chosen = windows[chosen_index]
        targets.iloc[start:stop] = weights[chosen].iloc[start:stop].to_numpy()
        first_fill = start + config.lag
        rows.append({"fold": len(rows) + 1, "train_start": panel.index[warmup],
                     "train_end": panel.index[start - 1], "decision_start": panel.index[start],
                     "decision_end": panel.index[stop - 1],
                     "first_possible_fill": panel.index[first_fill] if first_fill < len(panel) else pd.NaT,
                     "window": chosen, "training_score": scores[chosen_index] if np.isfinite(scores[chosen_index]) else None,
                     "training_sessions": start - warmup, "test_sessions": stop - start,
                     **asdict(config)})
    result = run_backtest(panel, targets, config=config)
    return WalkForwardResult(result, targets, pd.DataFrame(rows), panel.index[train_bars])
