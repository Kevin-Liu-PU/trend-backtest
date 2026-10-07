"""Self-contained static HTML; no JavaScript, fonts, CDNs or external assets."""
from html import escape

import numpy as np


NAMES = {"Fixed SMA200": "Fixed SMA200", "Buy and hold": "Buy and hold", "Walk-forward": "Walk-forward"}
COLORS = ["#15786e", "#71849e", "#c28149"]


def _table(frame, columns=None):
    f = frame if columns is None else frame[columns]
    return f.to_html(index=False, border=0, classes="data", escape=True,
                     float_format=lambda x: f"{x:.4f}", na_rep="—")


def _percent(value):
    return "—" if value is None or not np.isfinite(value) else f"{100 * value:.2f}%"


def _number(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.2f}"


def _display_metrics(frame):
    visible = frame.copy()
    for key in ["total_return", "annualized_return", "max_drawdown"]:
        if key in visible:
            visible[key] = visible[key].map(_percent)
    if "excess_sharpe" in visible:
        visible["excess_sharpe"] = visible["excess_sharpe"].map(_number)
    if "strategy" in visible:
        visible["strategy"] = visible["strategy"].map(lambda name: NAMES.get(name, name))
    return visible.rename(columns={
        "strategy": "Strategy", "total_return": "Total return",
        "annualized_return": "Annualized return", "max_drawdown": "Max drawdown",
        "excess_sharpe": "Excess Sharpe", "sessions": "Sessions",
        "cost_bps": "Cost per side (bp)", "lag": "Lag (sessions)", "window": "SMA window",
    })


def equity_svg(returns):
    curves = {name: np.r_[100.0, 100 * np.cumprod(1 + r.to_numpy())] for name, r in returns.items()}
    lo = min(float(x.min()) for x in curves.values()) * 0.95
    hi = max(float(x.max()) for x in curves.values()) * 1.05
    first_date = str(returns.index[0].date())
    last_date = str(returns.index[-1].date())

    def render_chart(width, height, left, top, inner_w, inner_h, css_class):
        right = left + inner_w
        svg = [f'<svg class="{css_class}" viewBox="0 0 {width} {height}" role="img" aria-label="Equity curves for three simulations, each starting at 100">',
               '<title>Equity comparison using synthetic data</title>']
        for value in np.linspace(lo, hi, 5):
            y = top + inner_h * (hi - value) / (hi - lo)
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5ebe9"/>')
            svg.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>')
        for (name, values), color in zip(curves.items(), COLORS):
            points = " ".join(f'{left + inner_w*i/(len(values)-1):.1f},{top + inner_h*(hi-v)/(hi-lo):.1f}' for i, v in enumerate(values))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"><title>{escape(NAMES.get(name, name))}</title></polyline>')
        svg.append(f'<text x="{left}" y="{height - 10}">{escape(first_date)}</text><text x="{right}" y="{height - 10}" text-anchor="end">{escape(last_date)}</text></svg>')
        return "".join(svg)

    legend = "".join(f'<span><i style="background:{color}"></i>{escape(NAMES.get(name, name))}</span>' for name, color in zip(curves, COLORS))
    desktop = render_chart(940, 320, 54, 24, 858, 250, "chart-desktop")
    mobile = render_chart(360, 240, 34, 20, 316, 180, "chart-mobile")
    return f'<div class="legend">{legend}</div><div class="chart-scroll">{desktop}{mobile}</div>'


def build_html(metrics, returns, folds, stress, surface, manifest):
    execution = manifest["execution"]
    fill_label = "Open: split-interval returns" if execution["fill"] == "open" else "Close: whole-day approximation"
    visible = _display_metrics(metrics.drop(columns=["sessions"]))
    best = metrics.loc[metrics["total_return"].idxmax()]
    negative_note = " All three simulations have negative total returns." if (metrics["total_return"] < 0).all() else ""
    conclusion = f"{NAMES.get(best['strategy'], best['strategy'])} has the highest total return in this demonstration.{negative_note}"
    cards = []
    for index, row in enumerate(metrics.to_dict("records")):
        cards.append(f'''<article class="result-card" style="--series:{COLORS[index % len(COLORS)]}">
<div class="strategy"><span class="series-dot"></span>{escape(NAMES.get(row['strategy'], row['strategy']))}</div>
<div class="metric-label">Total return</div><div class="metric-value">{_percent(row['total_return'])}</div>
<div class="card-bottom"><span>Max drawdown</span><strong>{_percent(row['max_drawdown'])}</strong></div></article>''')
    fold_view = folds[["fold", "train_end", "decision_start", "first_possible_fill", "window", "training_score"]].copy()
    fold_view["training_score"] = fold_view["training_score"].map(_number)
    fold_view = fold_view.rename(columns={
        "fold": "Fold", "train_end": "Training ends", "decision_start": "Decisions begin",
        "first_possible_fill": "First possible fill", "window": "Selected SMA", "training_score": "Training excess Sharpe",
    })
    stress_view = _display_metrics(stress[["cost_bps", "lag", "total_return", "max_drawdown", "excess_sharpe"]])
    stress_view["Cost per side (bp)"] = stress_view["Cost per side (bp)"].map(lambda v: f"{v:g}")
    surface_view = _display_metrics(surface[["window", "total_return", "max_drawdown", "excess_sharpe"]])
    sessions = len(returns)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>trend-backtest</title>
<style>
:root{{--ink:#20312e;--muted:#63736e;--line:#dfe7e3;--paper:#fff;--teal:#15786e}}
*{{box-sizing:border-box}}body{{margin:0;background:#f1f4f1;color:var(--ink);font:15px/1.7 "Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}}
main{{max-width:1160px;margin:0 auto;padding:44px 44px 32px}}
.masthead{{display:flex;align-items:center;justify-content:space-between;gap:20px;padding-bottom:26px;border-bottom:1px solid var(--line)}}
.brand{{font-size:17px;font-weight:750;letter-spacing:.02em;display:flex;align-items:center;gap:10px}}
.brand-mark{{width:22px;height:22px;border:6px solid var(--teal);border-radius:50%}}
.edition{{font-size:12px;letter-spacing:.1em;color:var(--muted)}}
.hero{{padding:44px 0 30px}}.eyebrow{{font-size:12px;letter-spacing:.12em;color:var(--teal);font-weight:700;margin:0 0 10px}}
h1{{font-size:42px;line-height:1.35;font-weight:650;letter-spacing:-.035em;margin:0 0 14px}}
.intro{{font-size:17px;color:var(--muted);max-width:790px;margin:0;line-height:1.85}}
.run-settings{{display:grid;grid-template-columns:1fr 1fr 1.5fr;gap:26px;padding:22px 0 25px;border-bottom:1px solid var(--line)}}
.setting-label{{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}}.setting-value{{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}}
section{{margin-top:40px}}.section-top{{display:flex;align-items:baseline;justify-content:space-between;gap:18px;margin-bottom:17px}}
h2{{font-size:24px;line-height:1.4;letter-spacing:-.025em;margin:0;font-weight:650}}
.section-index{{font-size:12px;letter-spacing:.08em;color:var(--teal);display:inline-block;margin-right:12px;vertical-align:middle}}
.synthetic-note{{font-size:13px;color:#91652e;margin:0;white-space:nowrap}}
.conclusion{{margin:0 0 20px;font-size:17px;line-height:1.8}}
.result-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
.result-card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:22px 24px 18px}}
.strategy{{font-weight:650;font-size:15px;display:flex;align-items:center;gap:8px}}
.series-dot{{width:8px;height:8px;background:var(--series);border-radius:50%;flex:none}}
.metric-label{{color:var(--muted);font-size:12px;margin-top:22px}}
.metric-value{{font-size:42px;font-weight:600;letter-spacing:-.055em;line-height:1.35;font-variant-numeric:tabular-nums}}
.card-bottom{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px 10px;border-top:1px solid #edf0ee;margin-top:18px;padding-top:12px;font-size:13px;color:var(--muted)}}
.card-bottom strong{{font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}}
.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:26px 28px}}
.chart-header{{display:flex;align-items:baseline;justify-content:space-between;gap:20px;margin-bottom:18px}}
h3{{font-size:17px;line-height:1.5;margin:0;font-weight:650}}.scope{{font-size:13px;color:var(--muted);margin:0}}
.legend{{display:flex;gap:24px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-bottom:6px}}
.legend span{{display:inline-flex;align-items:center;gap:8px}}.legend i{{display:inline-block;width:18px;height:3px;border-radius:2px}}
svg{{display:block;width:100%;height:auto}}.chart-mobile{{display:none}}svg text{{font:12px "Segoe UI",sans-serif;fill:#73817d}}
.chart-scroll{{overflow-x:auto}}.table-scroll{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:14px 13px;border-bottom:1px solid #e9eeeb;white-space:nowrap;text-align:right!important}}
th{{color:var(--muted);font-size:12px;font-weight:600;background:#f8faf8}}
th:first-child,td:first-child{{text-align:left!important}}th:first-child{{border-radius:6px 0 0 6px}}th:last-child{{border-radius:0 6px 6px 0}}
tbody tr:last-child td{{border-bottom:0}}tbody tr:hover{{background:#fafcfb}}
.chart-table{{margin-top:20px}}.detail-block{{margin-top:26px}}.detail-heading{{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:12px}}
.method-sentence{{margin:8px 0 18px;color:var(--muted);font-size:14px}}
details{{margin-top:18px;background:var(--paper);border:1px solid var(--line);border-radius:14px}}
summary{{padding:20px 26px;font-size:16px;font-weight:600;cursor:pointer;list-style-position:outside;margin-left:20px}}
summary::marker{{color:var(--teal)}}summary:focus-visible,a:focus-visible{{outline:2px solid var(--teal);outline-offset:4px}}
.detail-content{{padding:0 26px 24px}}.detail-content .scope{{margin:0 0 14px}}
footer{{display:grid;grid-template-columns:1fr 1.25fr;gap:60px;border-top:1px solid var(--line);margin-top:40px;padding-top:25px;color:var(--muted);font-size:12px}}
.footer-title{{font-size:13px;color:var(--ink);font-weight:650;margin:0 0 10px}}
.downloads{{display:flex;gap:8px 18px;flex-wrap:wrap;max-width:380px}}
a{{color:var(--teal);text-decoration:none}}a:hover{{text-decoration:underline}}
.footer-notes{{text-align:left;max-width:550px;justify-self:end}}.footer-notes p{{margin:0 0 9px;line-height:1.85}}
.fingerprints{{margin-top:18px}}.fingerprints p{{margin:5px 0}}code{{font-family:Consolas,monospace;font-size:12px;overflow-wrap:anywhere}}
@media(max-width:760px){{
main{{padding:24px 20px}}.masthead{{padding-bottom:20px}}.edition{{font-size:10px}}
.hero{{padding:32px 0 22px}}h1{{font-size:32px}}.intro{{font-size:15px}}
.run-settings{{grid-template-columns:1fr 1fr;gap:18px}}.run-settings>div:last-child{{grid-column:1/-1}}
.section-top{{align-items:flex-start;flex-direction:column;gap:8px}}section{{margin-top:32px}}h2{{font-size:22px}}
.conclusion{{font-size:15px}}.result-grid{{gap:10px}}.result-card{{padding:18px 14px 14px}}
.strategy{{font-size:13px}}.metric-value{{font-size:29px}}.card-bottom{{flex-direction:column;align-items:flex-start;gap:3px}}
.panel{{padding:20px 16px}}.chart-header,.detail-heading{{align-items:flex-start;flex-direction:column;gap:5px}}
.chart-desktop{{display:none}}.chart-mobile{{display:block}}.legend{{gap:12px 18px}}th,td{{padding:12px 11px}}
footer{{grid-template-columns:1fr;gap:26px}}.footer-notes{{justify-self:stretch;max-width:none}}
summary{{padding:18px 18px;font-size:15px}}.detail-content{{padding:0 16px 20px}}
}}
@media(max-width:480px){{
.result-grid{{grid-template-columns:1fr}}.result-card{{display:grid;grid-template-columns:1fr auto;align-items:center;gap:0 18px;padding:20px}}
.strategy{{grid-column:1;grid-row:1;font-size:15px}}.metric-label{{grid-column:2;grid-row:1;margin:0;text-align:right}}
.metric-value{{grid-column:2;grid-row:2;font-size:36px}}.card-bottom{{grid-column:1;grid-row:2;border:0;margin:0;padding-top:8px;font-size:12px}}
.synthetic-note{{font-size:12px}}.section-index{{margin-right:8px}}.legend{{font-size:12px}}.edition{{letter-spacing:0}}
}}
@media print{{
body{{background:#fff}}main{{max-width:none;padding:10mm}}.result-card,.panel,details{{break-inside:avoid}}
section{{margin-top:25px}}.hero{{padding-top:25px}}h1{{font-size:32px}}.metric-value{{font-size:32px}}
summary{{display:none}}details>.detail-content{{display:block}}footer{{gap:25px}}a{{color:var(--ink)}}
}}
</style></head><body><main>
<header class="masthead"><div class="brand"><span class="brand-mark" aria-hidden="true"></span>trend-backtest</div><div class="edition">SYNTHETIC DATA</div></header>
<div class="hero"><p class="eyebrow">OFFLINE TREND-TIMING COMPARISON</p><h1>Trend timing vs. buy and hold</h1>
<p class="intro">Compare fixed SMA200, buy and hold, and walk-forward selection on the same dataset. Each simulation accounts for signals, trading costs, and execution delays, with downloadable results for checking the calculations.</p></div>
<div class="run-settings" aria-label="Run settings">
<div><span class="setting-label">Execution model</span><span class="setting-value">{fill_label}</span></div>
<div><span class="setting-label">Cost per side / signal lag</span><span class="setting-value">{execution['cost_bps']:g} bp / {execution['lag']} session(s)</span></div>
<div><span class="setting-label">Common evaluation period · {sessions:,} sessions</span><span class="setting-value">{manifest['evaluation']['start']} — {manifest['evaluation']['end']}</span></div></div>
<section aria-labelledby="results-title">
<div class="section-top"><h2 id="results-title"><span class="section-index">01</span>Results</h2><p class="synthetic-note">Synthetic data, not real trading results</p></div>
<p class="conclusion">{escape(conclusion)}</p><div class="result-grid">{"".join(cards)}</div></section>
<section aria-labelledby="chart-title">
<div class="section-top"><h2 id="chart-title"><span class="section-index">02</span>Equity and risk</h2></div>
<div class="panel"><div class="chart-header"><h3>Three curves over the same period</h3><p class="scope">Continuous portfolio slices · rebased to 100</p></div>
{equity_svg(returns)}
<div class="chart-table table-scroll" role="region" aria-label="Metrics for the common evaluation period" tabindex="0">{_table(visible)}</div></div></section>
<section aria-labelledby="details-title">
<div class="section-top"><h2 id="details-title"><span class="section-index">03</span>Validation details</h2><p class="scope">{len(folds)} walk-forward folds · {len(stress)} execution checks · {len(surface)} candidate windows</p></div>
<div class="panel"><div class="detail-heading"><h3>Walk-forward selections</h3><p class="scope">Training data available at each decision only</p></div>
<p class="method-sentence">Each fold selects SMA150, 200, or 250 by training excess Sharpe. Holdings and costs remain continuous across folds.</p>
<div class="table-scroll" role="region" aria-label="Walk-forward selection records" tabindex="0">{_table(fold_view)}</div></div>
<details><summary>Cost and lag sensitivity</summary><div class="detail-content">
<p class="scope">Fixed SMA200 · all {manifest['dataset']['sessions']:,} synthetic sessions, including warmup · {fill_label}</p>
<div class="table-scroll" role="region" aria-label="Execution sensitivity metrics" tabindex="0">{_table(stress_view)}</div></div></details>
<details><summary>Candidate window comparison</summary><div class="detail-content">
<p class="scope">All {manifest['dataset']['sessions']:,} synthetic sessions, including warmup · {fill_label} · {execution['cost_bps']:g} bp per side / {execution['lag']}-session lag</p>
<div class="table-scroll" role="region" aria-label="Candidate window metrics" tabindex="0">{_table(surface_view)}</div></div></details></section>
<footer>
<div><p class="footer-title">Download this run</p><div class="downloads"><a href="metrics.csv">Metrics</a><a href="daily_returns.csv">Daily returns</a><a href="walk_forward_folds.csv">Fold selections</a><a href="stress.csv">Execution checks</a><a href="parameter_surface.csv">Window comparison</a><a href="manifest.json">Settings and hashes</a></div>
<div class="fingerprints"><p>Data SHA-256<br><code title="{manifest['dataset']['sha256']}">{manifest['dataset']['sha256'][:12]}…{manifest['dataset']['sha256'][-8:]}</code></p><p>Source SHA-256<br><code title="{manifest['source_sha256']}">{manifest['source_sha256'][:12]}…{manifest['source_sha256'][-8:]}</code></p></div></div>
<aside class="footer-notes" aria-label="Model assumptions and scope"><p class="footer-title">Model assumptions and scope</p>
<p>The evaluation period is a slice of continuous simulations. Fixed SMA200 and buy and hold retain their earlier positions; walk-forward starts in cash until its first decision fills after the configured lag. Annualization uses 252 sessions, and excess Sharpe subtracts the cash return.</p>
<p>Signals are formed after the close. The open model assigns overnight and intraday returns to the appropriate holdings. Positions are unleveraged equity or cash; switching costs two traded sides, with costs approximately deducted from daily returns. The close model is a whole-day-return approximation, not realistic next-open execution.</p>
<p>Sensitivity and window comparisons use the full sample as diagnostics. They are not directly comparable with the main evaluation period and are not used to choose the fixed SMA200 rule after seeing the results. Synthetic prices and weekday dates demonstrate the software only. There are no broker connections, orders, taxes, real market data, partial fills, or claims of statistical significance. Synthetic data only.</p>
</aside></footer></main></body></html>
'''
