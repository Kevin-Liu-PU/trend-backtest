import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from trend_timing.backtest import ExecutionConfig, run_backtest
from trend_timing.data import generate_sample, prepare_panel
from trend_timing.metrics import summary
from trend_timing.signals import sma_weights


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range("2030-01-01", periods=4)
        # Day 1: +10% overnight then -5% intraday. Day 2: +20% overnight,
        # +10% intraday. The strategy enters day 1 and exits day 2.
        raw = pd.DataFrame({"open": [100, 110, 125.4, 137.94],
                            "close": [100, 104.5, 137.94, 137.94],
                            "cash_return": [0, 0, 0, 0]}, index=self.index)
        self.panel = prepare_panel(raw)
        self.targets = pd.DataFrame({"cash": [0., 1., 1., 1.], "equity": [1., 0., 0., 0.]}, index=self.index)

    def test_open_gap_ownership_and_costs_by_hand(self):
        result = run_backtest(self.panel, self.targets, config=ExecutionConfig("open", 10, 1))
        # Starts flat: cannot own day-1 gap. On exit day the old equity owns +20%.
        np.testing.assert_allclose(result.returns, [0, -0.051, 0.198, 0], atol=1e-12)
        np.testing.assert_allclose(result.turnover, [0, 1, 2, 0])
        self.assertAlmostEqual(result.equity.iloc[-1], 0.949 * 1.198)

    def test_legacy_close_approximation_is_distinct(self):
        result = run_backtest(self.panel, self.targets, config=ExecutionConfig("close", 10, 1))
        np.testing.assert_allclose(result.returns, [0, 0.044, -0.002, 0], atol=1e-12)

    def test_open_compounds_intervals_and_no_extra_fee_without_switch(self):
        targets = self.targets.copy()
        targets.loc[:, "cash"] = 0.
        targets.loc[:, "equity"] = 1.
        result = run_backtest(self.panel, targets, config=ExecutionConfig("open", 0, 1))
        self.assertAlmostEqual(result.returns.iloc[2], 1.2 * 1.1 - 1)
        self.assertEqual(result.turnover.iloc[2], 0)

    def test_lag_and_prefix_stability(self):
        result = run_backtest(self.panel, self.targets, config=ExecutionConfig("open", 10, 2))
        np.testing.assert_allclose(result.weights_held.iloc[:2], 0)
        self.assertEqual(result.weights_held.iloc[2].equity, 1)
        shorter = run_backtest(self.panel.iloc[:3], self.targets.iloc[:3], config=result.config)
        assert_series_equal(shorter.returns, result.returns.iloc[:3])

    def test_invalid_inputs_fail_closed(self):
        for kwargs in ({"fill": "bad"}, {"lag": 0}, {"lag": 1.5}, {"lag": True}, {"cost_bps": float("nan")}, {"cost_bps": -1}):
            with self.assertRaises(ValueError):
                ExecutionConfig(**kwargs)
        bad = self.targets.copy()
        bad.iloc[0] = [0.5, 0.5]
        with self.assertRaises(ValueError):
            run_backtest(self.panel, bad, config=ExecutionConfig())
        broken = self.panel.copy()
        broken.iloc[1, broken.columns.get_loc("equity_on")] = 0.9
        with self.assertRaises(ValueError):
            run_backtest(broken, self.targets, config=ExecutionConfig())

    def test_initial_loss_is_in_drawdown(self):
        r = pd.Series([-0.1, 0.0], index=self.index[:2])
        self.assertAlmostEqual(summary(r, r * 0)["max_drawdown"], -0.1)


class SignalDataTests(unittest.TestCase):
    def test_generator_is_deterministic_and_prefix_stable(self):
        assert_frame_equal(generate_sample(60), generate_sample(100).iloc[:60])
        assert_frame_equal(generate_sample(60), generate_sample(60))

    def test_signal_warmup_equality_and_next_open_causality(self):
        index = pd.bdate_range("2030-01-01", periods=5)
        raw = pd.DataFrame({"open": [10]*5, "close": [10, 10, 12, 12, 8], "cash_return": [0]*5}, index=index)
        panel = prepare_panel(raw)
        targets = sma_weights(panel, 2)
        np.testing.assert_array_equal(targets.equity, [0, 0, 1, 1, 0])
        result = run_backtest(panel, targets, config=ExecutionConfig())
        # Today's close cannot change today's next-open holding.
        self.assertEqual(result.weights_held.equity.iloc[2], 0)
        self.assertEqual(result.weights_held.equity.iloc[3], 1)
        changed = panel.copy()
        changed.iloc[-1, changed.columns.get_loc("close")] = 100
        assert_frame_equal(sma_weights(changed, 2).iloc[:-1], targets.iloc[:-1])

    def test_data_validation_and_return_identity(self):
        panel = prepare_panel(generate_sample(30))
        np.testing.assert_allclose((1 + panel.equity_on) * (1 + panel.equity_in), 1 + panel.equity_ret)
        raw = generate_sample(30)
        for bad in [raw.iloc[::-1], pd.concat([raw, raw.iloc[-1:]]), raw.assign(close=np.nan), raw.assign(open=0)]:
            with self.assertRaises(ValueError):
                prepare_panel(bad)


if __name__ == "__main__":
    unittest.main()
