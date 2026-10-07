# trend-backtest

A small Python backtester for checking how signal timing and trading costs affect a moving-average strategy. It compares a fixed SMA200 rule, buy and hold, and an expanding walk-forward selection of SMA150 / 200 / 250.

[Open the static HTML report](https://kevin-liu-pu.github.io/trend-backtest/)

**All prices and results in this repository are synthetic.** They demonstrate the software's behavior, not real trading performance.

## Run the demo

Use Python 3.11 or newer. The checked report was generated with Python 3.12.10, NumPy 2.4.2, and pandas 3.0.0.

```sh
python -m venv .venv
# macOS / Linux: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-demo.txt
python -m pip install --no-deps -e .
python -m trend_timing --output outputs/demo
python -m unittest discover -s tests -v
```

Open `outputs/demo/report.html` in a browser. The report uses inline CSS and SVG; it needs no server, JavaScript, CDN, or network connection. The same run writes CSV results and a manifest with the data hash, source hash, settings, and library versions.

To rebuild the published report:

```sh
python scripts/build_pages.py
```

GitHub Pages serves only `docs/`: the generated report, its downloadable results, and a `.nojekyll` marker. The repository's Python code is not part of that site.

## How it works

- A trailing SMA produces a cash-or-equity target after each close. Equality keeps the previous target; warmup stays in cash.
- The default one-session lag applies in the accounting engine. The old holding receives the overnight move; the new holding receives the intraday move.
- Initial entry costs one traded side. Switching between cash and equity costs two. The default cost is 10 basis points per side.
- Walk-forward selection uses only the training history available before each block. Holdings and costs remain continuous across folds.
- An explicit immutable execution configuration passes through the engine, parameter checks, stress checks, and walk-forward calculation.

The main chart compares the same 1,260 post-training sessions for all three strategies. Cost/lag and window diagnostics use the full 2,016-session sample, including warmup; they have a different scope from the main chart.

## Data and limits

`src/trend_timing/sample.csv` contains generated prices, using seed `20261007` and a fictional weekday calendar beginning in 2030. Rebuild it explicitly with `python scripts/generate_sample.py`; missing input never triggers a download or silent regeneration.

The model has one synthetic equity and one tradable cash sleeve, fully allocated without leverage. Costs are deducted from daily returns as an approximation; this is not a share-level ledger. There are no broker connections, order endpoints, market-data downloads, taxes, partial fills, or real account data.

`--fill close` retains a whole-day-return approximation for comparison. It is not realistic next-open execution because it can assign an overnight move to a newly entered holding. Synthetic held-out periods demonstrate the selection mechanics only; they do not establish predictive performance.

## Tests

The 18 standard-library `unittest` cases cover hand-calculated gap ownership and costs, signal lag, invalid input, initial-loss drawdown, deterministic sample generation, execution-setting propagation, causal walk-forward selection, fold continuity, and a deterministic self-contained report with working local downloads.

The code is under `src/trend_timing/`, regressions under `tests/`, and regeneration commands under `scripts/`.
