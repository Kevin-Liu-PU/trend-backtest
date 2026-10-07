"""Synthetic OHLC input and explicit validation; no downloads or pickle caches."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261007
SAMPLE = Path(__file__).with_name("sample.csv")


def generate_sample(sessions: int = 2016, seed: int = SEED) -> pd.DataFrame:
    """Row-wise random draws make every generated prefix stable.

    Alternating manufactured regimes exercise switching; they are not fitted to
    prices or selected for favorable results. Dates are a fictional weekday clock.
    """
    if not isinstance(sessions, int) or sessions < 2:
        raise ValueError("sessions must be an integer >= 2")
    rng = np.random.default_rng(seed)
    rows = []
    last_close = 100.0
    for day in range(sessions):
        gap_noise, intraday_noise = rng.normal(size=2)
        drift = (0.00035, -0.00045, 0.00010, 0.00050)[(day // 252) % 4]
        gap = drift * 0.35 + 0.005 * gap_noise
        intraday = drift * 0.65 + 0.008 * intraday_noise
        opening = last_close * np.exp(gap) if day else 100.0
        closing = opening * np.exp(intraday) if day else 100.0
        rows.append((opening, closing, 0.00005))
        last_close = closing
    return pd.DataFrame(rows, index=pd.bdate_range("2030-01-01", periods=sessions,
                        name="date"), columns=["open", "close", "cash_return"])


def prepare_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a validated CSV-shaped frame to the engine's return components."""
    if len(raw) < 2 or not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("need at least two dated rows")
    if raw.index.tz is not None or raw.index.hasnans or not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("dates must be unique, sorted, non-null and timezone-naive")
    if not {"open", "close", "cash_return"}.issubset(raw.columns):
        raise ValueError("required columns: open, close, cash_return")
    p = raw[["open", "close", "cash_return"]].astype(float).copy()
    if not np.isfinite(p.to_numpy()).all() or (p[["open", "close"]] <= 0).any().any():
        raise ValueError("prices must be positive; every value must be finite")
    if (p["cash_return"] <= -1).any():
        raise ValueError("cash return must be greater than -1")
    p["equity_ret"] = p["close"].pct_change(fill_method=None).fillna(0.0)
    p["equity_on"] = (p["open"] / p["close"].shift(1) - 1).fillna(0.0)
    p["equity_in"] = (1 + p["equity_ret"]) / (1 + p["equity_on"]) - 1
    p["cash_ret"] = p["cash_return"]
    p["cash_on"] = 0.0
    p["cash_in"] = p["cash_ret"]
    return p


def load_sample(path: Path = SAMPLE) -> pd.DataFrame:
    # No auto-generation or silent fallback when the input is absent or invalid.
    raw = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return prepare_panel(raw)
