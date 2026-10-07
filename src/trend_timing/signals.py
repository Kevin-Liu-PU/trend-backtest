"""Causal, one-asset SMA gate adapted from quant/qr/signals.py and strategy.py."""
import numpy as np
import pandas as pd

ASSETS = ["cash", "equity"]


def sma_weights(panel: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """At each close: above SMA -> equity; below -> cash; equality holds state.

    Undefined warmup is cash. No signal lag here: the accounting engine owns lag.
    Only full positions are supported, excluding the legacy fractional-drift issue.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    price = panel["close"].to_numpy(dtype=float)
    average = panel["close"].rolling(window, min_periods=window).mean().to_numpy()
    equity = np.zeros(len(panel))
    state = 0.0
    for i in range(len(panel)):
        if price[i] > average[i]:
            state = 1.0
        elif price[i] < average[i]:
            state = 0.0
        equity[i] = state
    return pd.DataFrame({"cash": 1 - equity, "equity": equity}, index=panel.index)


def buy_hold_weights(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"cash": 0.0, "equity": 1.0}, index=panel.index)
