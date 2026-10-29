import numpy as np
import pandas as pd

from trader import backtest, consensus, data, live, report, risk, strategies
from trader.indicators import atr, rsi


def test_indicators_are_sane(bars):
    df = bars()
    a = atr(df).dropna()
    r = rsi(df["close"]).dropna()
    assert (a > 0).all()
    assert r.between(0, 100).all()


def test_consensus_needs_agreement_and_no_opposition():
    v = pd.DataFrame({"trend": [1, 1, 1, 1, 0], "breakout": [0, 1, 1, 1, 0], "pullback": [0, 0, 1, -1, 0]})
    out = consensus.combine(v, 2, {3: 1.0, 2: 0.5})
    assert list(out["direction"]) == [0, 1, 1, 0, 0]
    assert list(out["fire"]) == [False, True, True, False, False]  # bar 2 upgrades to full size
    assert list(out["size"]) == [0.0, 0.5, 1.0, 0.0, 0.0]
    assert out.at[2, "agreeing"] == "trend,breakout,pullback"


def test_no_lookahead(bars, cfg):
    """Signals on past bars must not change when future bars arrive."""
    df = bars(1500, drift=0.0003)
    full = backtest.signals(df, cfg)
    part = backtest.signals(df.iloc[:1000], cfg)
    cols = ["direction", "agree", "fire"]
    pd.testing.assert_frame_equal(full.iloc[:1000][cols], part[cols])


def test_uptrend_makes_money_without_costs(bars, cfg):
    df = bars(3000, drift=0.0008, vol=0.003)
    inst = {"name": "T", "spread_bps": 0, "leverage": 20}
    trades = backtest.run_instrument(df, inst, cfg)
    assert trades
    assert sum(t.pnl for t in trades) > 0
    assert sum(t.direction == 1 for t in trades) > sum(t.direction == -1 for t in trades)


def test_costs_reduce_pnl(bars, cfg):
    df = bars(3000, drift=0.0003)
    cheap = backtest.run_instrument(df, {"name": "T", "spread_bps": 0, "leverage": 20}, cfg)
    dear = backtest.run_instrument(df, {"name": "T", "spread_bps": 30, "swap_annual": 0.2, "leverage": 20}, cfg)
    assert sum(t.pnl for t in dear) < sum(t.pnl for t in cheap)


def test_trades_never_overlap_and_risk_is_capped(bars, cfg):
    df = bars(3000)
    trades = backtest.run_instrument(df, {"name": "T", "spread_bps": 5, "leverage": 20}, cfg)
    for a, b in zip(trades, trades[1:]):
        assert b.entry_time > a.exit_time or (b.entry_time == a.exit_time and a.reason == "reverse")
    max_risk = cfg["account"]["capital"] * cfg["account"]["risk_per_trade"]
    assert all(t.risk <= max_risk + 1e-6 for t in trades)
    # Losses can exceed 1R only through gaps and costs, never by much on smooth data.
    assert min(t.r_multiple for t in trades) > -1.5


def test_position_size_respects_leverage():
    units, at_risk = risk.position_size(10_000, 0.01, 1.0, entry=100, stop=99.99, leverage=2)
    assert units * 100 <= 20_000 + 1e-6
    assert at_risk < 100


def test_portfolio_limits_open_positions(bars, cfg):
    trades = []
    for seed in range(8):
        trades += backtest.run_instrument(bars(2000, seed=seed), {"name": f"I{seed}", "leverage": 20}, cfg)
    accepted = backtest.portfolio(trades, cfg)
    events = sorted([(t.entry_time, 1) for t in accepted] + [(t.exit_time, -1) for t in accepted],
                    key=lambda e: (e[0], e[1]))
    open_now = peak = 0
    for _, step in events:
        open_now += step
        peak = max(peak, open_now)
    assert peak <= cfg["account"]["max_open_positions"]


def test_report_runs_on_frames(bars, cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = cfg | {"instruments": [{"name": "A", "source": "csv", "leverage": 20, "spread_bps": 2},
                                 {"name": "B", "source": "csv", "leverage": 20, "spread_bps": 2}]}
    summary = report.run(cfg, frames={"A": bars(3000, seed=1), "B": bars(3000, seed=2)})
    assert set(summary["instruments"]) == {"A", "B"}
    assert "pass" in summary["verdict"]
    assert (tmp_path / "reports" / "summary.json").is_file()
    assert "Verdict" in report.table(summary)


def test_closed_bars_drops_forming_bar(bars):
    df = bars(10)
    now = df.index[-1] + pd.Timedelta(minutes=30)
    assert len(live.closed_bars(df, "1h", now)) == 9
    assert len(live.closed_bars(df, "1h", now + pd.Timedelta(hours=1))) == 10


def test_evaluate_matches_backtest_signal(bars, cfg):
    df = bars(3000, drift=0.0005)
    sig = backtest.signals(df, cfg)
    i = int(np.flatnonzero(sig["fire"].to_numpy())[-1])
    s = live.evaluate(df.iloc[: i + 1], {"name": "T", "leverage": 20}, cfg)
    assert s is not None and s["direction"] == sig["direction"].iloc[i]
    assert (s["stop"] < s["entry"]) == (s["direction"] == 1)


def test_read_mt5_csv(tmp_path):
    p = tmp_path / "GOLD_H1.csv"
    p.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
                 "2025.01.02\t00:00:00\t2620.1\t2625.0\t2618.3\t2623.4\t1200\t0\t35\n"
                 "2025.01.02\t01:00:00\t2623.4\t2627.0\t2621.0\t2626.2\t1100\t0\t35\n")
    df = data.read_csv(str(p))
    assert list(df.columns) == data.COLUMNS
    assert len(df) == 2 and df["close"].iloc[-1] == 2626.2
