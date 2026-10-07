"""Small metrics subset; results describe the artificial sample only."""
import numpy as np
import pandas as pd


def summary(returns: pd.Series, cash_returns: pd.Series) -> dict:
    if len(returns) < 2 or not returns.index.equals(cash_returns.index):
        raise ValueError("metrics need >=2 matching return/cash rows")
    r = returns.to_numpy(dtype=float)
    rf = cash_returns.to_numpy(dtype=float)
    if not np.isfinite(r).all() or not np.isfinite(rf).all() or (r <= -1).any():
        raise ValueError("metrics require finite returns and positive wealth")
    equity = np.r_[1.0, np.cumprod(1 + r)]
    dd = equity / np.maximum.accumulate(equity) - 1
    excess = r - rf
    vol = np.std(excess, ddof=1)
    return {"sessions": len(r), "total_return": float(equity[-1] - 1),
            "annualized_return": float(equity[-1] ** (252 / len(r)) - 1),
            "max_drawdown": float(dd.min()),
            "excess_sharpe": float(np.mean(excess) / vol * np.sqrt(252)) if vol > 1e-14 else None}
