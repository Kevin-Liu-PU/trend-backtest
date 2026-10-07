"""Vectorized daily accounting adapted from quant/qr/backtest.py.

The open convention assigns the overnight interval to the previous holding and
the intraday interval to the new holding. 'close' preserves the legacy whole-day
return approximation solely for comparison; it is not a realistic next-open fill.
Costs preserve the source's approximation: turnover*bps deducted from daily return.
"""
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from .signals import ASSETS


@dataclass(frozen=True)
class ExecutionConfig:
    fill: str = "open"
    cost_bps: float = 10.0
    lag: int = 1

    def __post_init__(self):
        if self.fill not in {"open", "close"}:
            raise ValueError("fill must be open or close")
        if isinstance(self.lag, bool) or not isinstance(self.lag, Integral) or self.lag < 1:
            raise ValueError("lag must be a positive integer")
        if not np.isfinite(self.cost_bps) or self.cost_bps < 0:
            raise ValueError("cost_bps must be finite and nonnegative")


@dataclass
class BacktestResult:
    returns: pd.Series
    equity: pd.Series
    weights_held: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    config: ExecutionConfig

    @property
    def metadata(self):
        return asdict(self.config)


def run_backtest(panel: pd.DataFrame, targets: pd.DataFrame, *, config: ExecutionConfig) -> BacktestResult:
    """Run one continuous portfolio from flat; explicit config is mandatory.

    Target weights must be one-hot cash/equity, sum to one, and match panel dates.
    The first lag rows are flat. Initial entry costs one side; switching costs two.
    A result slice is a slice of a continuously simulated portfolio, not a restart.
    """
    if not isinstance(config, ExecutionConfig):
        raise TypeError("config must be ExecutionConfig")
    if len(panel) < 2 or not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("panel must have at least two unique sorted rows")
    if list(targets.columns) != ASSETS or not targets.index.equals(panel.index):
        raise ValueError("targets must match panel dates and cash/equity column order")
    w = targets.to_numpy(dtype=float)
    if not np.isfinite(w).all() or not np.isin(w, [0.0, 1.0]).all() or not (w.sum(axis=1) == 1).all():
        raise ValueError("only one-hot, fully allocated cash/equity targets are supported")
    columns = [f"{a}_{part}" for part in ["ret", "on", "in"] for a in ASSETS]
    if not set(columns).issubset(panel.columns):
        raise ValueError("panel is missing return components")
    r = panel[[f"{a}_ret" for a in ASSETS]].to_numpy(dtype=float)
    on = panel[[f"{a}_on" for a in ASSETS]].to_numpy(dtype=float)
    intra = panel[[f"{a}_in" for a in ASSETS]].to_numpy(dtype=float)
    if not all(np.isfinite(x).all() and (x > -1).all() for x in [r, on, intra]):
        raise ValueError("returns must be finite and greater than -1")
    if not np.allclose((1 + on) * (1 + intra), 1 + r, rtol=0, atol=1e-12):
        raise ValueError("overnight/intraday components do not reconcile")
    held = np.zeros_like(w)
    if config.lag < len(w):
        held[config.lag:] = w[:-config.lag]
    previous = np.zeros_like(held)
    previous[1:] = held[:-1]
    turnover = np.abs(held - previous).sum(axis=1)
    costs = turnover * (config.cost_bps / 10000)
    if config.fill == "open":
        gross = (1 + (previous * on).sum(axis=1)) * (1 + (held * intra).sum(axis=1)) - 1
    else:
        gross = (held * r).sum(axis=1)
    net = gross - costs
    if (net <= -1).any() or not np.isfinite(net).all():
        raise ValueError("costs/returns imply invalid or exhausted portfolio wealth")
    index = panel.index
    return BacktestResult(
        pd.Series(net, index=index, name="return"),
        pd.Series(np.cumprod(1 + net), index=index, name="equity"),
        pd.DataFrame(held, index=index, columns=ASSETS),
        pd.Series(turnover, index=index, name="turnover"),
        pd.Series(costs, index=index, name="cost"), config)
