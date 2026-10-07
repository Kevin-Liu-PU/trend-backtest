"""Regression coverage for validation execution settings and causal selection.

Small manufactured paths exercise gaps and switching without historical data.
The direct engine supplies execution expectations; truncated inputs independently
establish what information was available to each walk-forward selection.
"""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from trend_timing.backtest import ExecutionConfig, run_backtest
from trend_timing.data import generate_sample, prepare_panel
from trend_timing.metrics import summary
from trend_timing.signals import sma_weights
from trend_timing.validation import evaluate_windows, stress_test, walk_forward


def gap_prices(sessions=72):
    """Repeated reversals, with overnight and intraday moves that often oppose."""
    step = np.arange(sessions)
    cycle = np.array([100, 108, 112, 105, 96, 91, 98, 107, 115, 109, 99, 93], dtype=float)
    close = cycle[step % len(cycle)] * np.power(1.002, step)
    gaps = np.array([0.08, -0.06, 0.10, -0.04, -0.09, 0.07, -0.05], dtype=float)
    opening = np.r_[close[0], close[:-1]] * (1 + gaps[step % len(gaps)])
    opening[0] = close[0]
    return pd.DataFrame(
        {"open": opening, "close": close, "cash_return": np.full(sessions, 0.00003)},
        index=pd.bdate_range("2030-01-01", periods=sessions, name="date"),
    )


class ExecutionPropagationTests(unittest.TestCase):
    def setUp(self):
        self.panel = prepare_panel(gap_prices())

    def assert_summary_matches(self, row, result):
        # Wealth and drawdown are derived directly from the portfolio path,
        # avoiding an oracle that simply calls the wrapper's summary function.
        wealth = np.r_[1.0, result.equity.to_numpy()]
        drawdown = wealth / np.maximum.accumulate(wealth) - 1
        self.assertEqual(int(row["sessions"]), len(result.returns))
        self.assertAlmostEqual(float(row["total_return"]), wealth[-1] - 1, places=12)
        self.assertAlmostEqual(float(row["max_drawdown"]), drawdown.min(), places=12)
        self.assertEqual(row["fill"], result.config.fill)
        self.assertEqual(float(row["cost_bps"]), result.config.cost_bps)
        self.assertEqual(int(row["lag"]), result.config.lag)

    def test_surface_preserves_both_fill_modes_and_nondefault_config(self):
        surfaces = {}
        for fill in ("open", "close"):
            config = ExecutionConfig(fill=fill, cost_bps=17.0, lag=2)
            table = evaluate_windows(self.panel, (2, 3, 5), config=config)
            self.assertEqual(table["window"].tolist(), [2, 3, 5])
            surfaces[fill] = table
            for _, row in table.iterrows():
                with self.subTest(fill=fill, window=int(row["window"])):
                    direct = run_backtest(
                        self.panel, sma_weights(self.panel, int(row["window"])), config=config
                    )
                    self.assert_summary_matches(row, direct)
        # Ensure this fixture could actually catch a silently substituted fill.
        self.assertFalse(np.allclose(surfaces["open"].total_return, surfaces["close"].total_return))

    def test_stress_preserves_fill_while_overriding_cost_and_lag(self):
        tables = {}
        for fill in ("open", "close"):
            config = ExecutionConfig(fill=fill, cost_bps=37.0, lag=2)
            table = stress_test(self.panel, window=3, config=config,
                                costs=(3.0, 17.0), lags=(1, 3))
            tables[fill] = table
            self.assertEqual(len(table), 4)
            self.assertEqual(
                set(zip(table.cost_bps, table.lag)), {(3.0, 1), (3.0, 3), (17.0, 1), (17.0, 3)}
            )
            for _, row in table.iterrows():
                with self.subTest(fill=fill, cost=row["cost_bps"], lag=row["lag"]):
                    stressed = replace(config, cost_bps=float(row["cost_bps"]), lag=int(row["lag"]))
                    direct = run_backtest(self.panel, sma_weights(self.panel, 3), config=stressed)
                    self.assert_summary_matches(row, direct)
            self.assertEqual(config, ExecutionConfig(fill=fill, cost_bps=37.0, lag=2))
        self.assertFalse(np.allclose(tables["open"].total_return, tables["close"].total_return))

    def test_walk_forward_executes_selected_targets_with_explicit_fill(self):
        outcomes = {}
        for fill in ("open", "close"):
            config = ExecutionConfig(fill=fill, cost_bps=13.0, lag=2)
            outcome = walk_forward(self.panel, (2, 3, 5), config=config,
                                   train_bars=16, test_bars=9)
            outcomes[fill] = outcome
            direct = run_backtest(self.panel, outcome.targets, config=config)
            self.assertEqual(outcome.result.config, config)
            self.assertTrue((outcome.folds["fill"] == fill).all())
            self.assertTrue((outcome.folds["cost_bps"] == config.cost_bps).all())
            self.assertTrue((outcome.folds["lag"] == config.lag).all())
            for field in ("returns", "equity", "turnover", "costs"):
                with self.subTest(fill=fill, field=field):
                    pd.testing.assert_series_equal(getattr(outcome.result, field), getattr(direct, field))
            pd.testing.assert_frame_equal(outcome.result.weights_held, direct.weights_held)
        self.assertFalse(np.allclose(outcomes["open"].result.returns, outcomes["close"].result.returns))


class WalkForwardCausalityTests(unittest.TestCase):
    windows = (2, 3, 5)
    train_bars = 16
    test_bars = 9

    def setUp(self):
        self.raw = gap_prices()
        self.panel = prepare_panel(self.raw)

    def run_walk_forward(self, panel, config):
        return walk_forward(panel, self.windows, config=config,
                            train_bars=self.train_bars, test_bars=self.test_bars)

    def test_each_selection_and_score_are_reproducible_from_training_only(self):
        for fill in ("open", "close"):
            config = ExecutionConfig(fill=fill, cost_bps=11.0, lag=2)
            outcome = self.run_walk_forward(self.panel, config)
            for _, fold in outcome.folds.iterrows():
                with self.subTest(fill=fill, fold=int(fold["fold"])):
                    # The oracle never receives any row from the decision/test block.
                    training = self.panel.loc[:fold["train_end"]].copy()
                    self.assertLess(training.index[-1], fold["decision_start"])
                    scores = []
                    for window in self.windows:
                        direct = run_backtest(training, sma_weights(training, window), config=config)
                        scored = direct.returns.iloc[max(self.windows) + config.lag:]
                        score = summary(scored, training.loc[scored.index, "cash_ret"])["excess_sharpe"]
                        scores.append(score if score is not None and np.isfinite(score) else -np.inf)
                    # Python max retains the first candidate on a tie, as specified.
                    best = max(range(len(self.windows)), key=lambda i: scores[i])
                    self.assertEqual(int(fold["window"]), self.windows[best])
                    self.assertTrue(np.isfinite(scores[best]), "fixture must have a meaningful training score")
                    self.assertAlmostEqual(float(fold["training_score"]), scores[best], places=12)

                    # A one-row test suffix is enough to reproduce the decision;
                    # later returns and the eventual block length are unavailable.
                    prefix = self.panel.loc[:fold["decision_start"]]
                    truncated = self.run_walk_forward(prefix, config).folds.iloc[-1]
                    self.assertEqual(int(truncated["window"]), int(fold["window"]))
                    self.assertAlmostEqual(float(truncated["training_score"]), float(fold["training_score"]), places=12)

    def test_future_price_mutation_cannot_change_a_fold_selection(self):
        config = ExecutionConfig(fill="open", cost_bps=7.0, lag=2)
        baseline = self.run_walk_forward(self.panel, config)
        for _, fold in baseline.folds.iterrows():
            with self.subTest(fold=int(fold["fold"])):
                start = self.raw.index.get_loc(fold["decision_start"])
                changed = self.raw.copy()
                future = np.arange(len(changed) - start, dtype=float)
                anchor = changed["close"].iloc[start - 1]
                closes = anchor * np.exp(1.3 + 0.03 * future + 0.4 * np.sin(future))
                changed.loc[changed.index[start:], "close"] = closes
                changed.loc[changed.index[start:], "open"] = closes * (0.7 + 0.1 * np.cos(future))
                mutated = self.run_walk_forward(prepare_panel(changed), config)
                actual = mutated.folds.loc[mutated.folds["decision_start"] == fold["decision_start"]].iloc[0]
                self.assertEqual(int(actual["window"]), int(fold["window"]))
                self.assertAlmostEqual(float(actual["training_score"]), float(fold["training_score"]), places=12)
                pd.testing.assert_series_equal(mutated.result.returns.iloc[:start], baseline.result.returns.iloc[:start])
                # Future closes cannot retroactively alter orders already decided.
                pd.testing.assert_frame_equal(
                    mutated.result.weights_held.iloc[start:start + config.lag],
                    baseline.result.weights_held.iloc[start:start + config.lag],
                )

    def test_appending_future_bars_preserves_signals_portfolio_and_selections(self):
        config = ExecutionConfig(fill="open", cost_bps=13.0, lag=2)
        prefix_length = self.train_bars + 2 * self.test_bars + 4
        prefix = self.panel.iloc[:prefix_length].copy()
        for window in self.windows:
            pd.testing.assert_frame_equal(sma_weights(prefix, window), sma_weights(self.panel, window).iloc[:prefix_length])
        short = self.run_walk_forward(prefix, config)
        full = self.run_walk_forward(self.panel, config)
        pd.testing.assert_frame_equal(short.targets, full.targets.iloc[:prefix_length])
        pd.testing.assert_frame_equal(short.result.weights_held, full.result.weights_held.iloc[:prefix_length])
        for field in ("returns", "equity", "turnover", "costs"):
            with self.subTest(field=field):
                pd.testing.assert_series_equal(getattr(short.result, field), getattr(full.result, field).iloc[:prefix_length])
        # The final partial fold's end/length can grow; its selection cannot.
        selection_columns = ["train_start", "train_end", "decision_start", "window", "training_score"]
        pd.testing.assert_frame_equal(short.folds[selection_columns], full.folds.iloc[:len(short.folds)][selection_columns])

    def test_fold_boundaries_preserve_holdings_and_do_not_add_costs(self):
        sessions = 48
        close = 100 * np.power(1.005, np.arange(sessions))
        opening = np.r_[close[0], close[:-1]] * 1.002
        opening[0] = close[0]
        raw = pd.DataFrame(
            {"open": opening, "close": close, "cash_return": np.zeros(sessions)},
            index=pd.bdate_range("2030-01-01", periods=sessions, name="date"),
        )
        panel = prepare_panel(raw)
        config = ExecutionConfig(fill="open", cost_bps=19.0, lag=2)
        train_bars = 12
        outcome = walk_forward(panel, self.windows, config=config, train_bars=train_bars, test_bars=7)
        # Every SMA is bullish before the first decision and remains bullish.
        # The first-in-grid tie break is deterministic; no fold requires a trade.
        self.assertTrue((outcome.folds["window"] == self.windows[0]).all())
        expected_targets = pd.DataFrame({"cash": 1.0, "equity": 0.0}, index=panel.index)
        expected_targets.iloc[train_bars:] = [0.0, 1.0]
        pd.testing.assert_frame_equal(outcome.targets, expected_targets)
        direct = run_backtest(panel, expected_targets, config=config)
        pd.testing.assert_series_equal(outcome.result.equity, direct.equity)
        pd.testing.assert_series_equal(outcome.result.returns, direct.returns)
        first_equity_fill = train_bars + config.lag
        self.assertTrue((outcome.result.weights_held["equity"].iloc[first_equity_fill:] == 1).all())
        self.assertEqual(
            outcome.result.turnover.index[outcome.result.turnover > 0].tolist(),
            [panel.index[config.lag], panel.index[first_equity_fill]],
        )
        # The documented approximation costs initial cash allocation one side
        # and the single cash/equity switch two; later fold boundaries cost zero.
        self.assertAlmostEqual(outcome.result.costs.iloc[config.lag], 19 / 10000)
        self.assertAlmostEqual(outcome.result.costs.iloc[first_equity_fill], 2 * 19 / 10000)
        self.assertTrue((outcome.result.costs.iloc[first_equity_fill + 1:] == 0).all())
        for _, fold in outcome.folds.iterrows():
            start = panel.index.get_loc(fold["decision_start"])
            if start + config.lag < len(panel):
                self.assertEqual(fold["first_possible_fill"], panel.index[start + config.lag])
            else:
                self.assertTrue(pd.isna(fold["first_possible_fill"]))

    def test_synthetic_generator_is_prefix_stable(self):
        pd.testing.assert_frame_equal(generate_sample(40), generate_sample(90).iloc[:40])


if __name__ == "__main__":
    unittest.main()
