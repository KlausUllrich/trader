"""Trading the jumpy moments. Run from the repo root:
    python research/jumpy_daily.py is            # design period 2010-2017
    python research/jumpy_daily.py oos B1,F1     # check period 2018-2023, run once for the picks

Daily bars from Yahoo; everything from 2024-06 on is cut off (already used).
Signal on the close of day D, entry at the open of day D+1.

Jumpy regime: ATR(14) > 1.5 x ATR(100) on the signal day.
Costs: trades entered in a jumpy regime pay 3x the normal spread (spread plus slippage).

Pre-registered candidates (fixed before running):
  B0  20-day breakout (strategies.breakout) on every day, normal exits from the config
  B1  the same breakout, but only while jumpy: risk 0.5% per trade, at most 2 open,
      skipped if the jumpy spread is more than 10% of the stop distance
  F1  oil only: fade a spike. Spike day = range of at least 2x ATR(14) and a close at a
      20-day high (low). If within 5 days a close comes back below the spike day's open
      (above, for a down spike), go the other way. Stop 0.5 ATR beyond the spike's extreme,
      target 3 ATR, time stop 20 days. Risk 0.5% per trade.
"""
import json
import sys

import numpy as np
import pandas as pd

from trader import backtest, data, strategies
from trader.config import load_config
from trader.indicators import atr

MODE = sys.argv[1] if len(sys.argv) > 1 else "is"
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
PERIODS = {"is": ("2010-01-01", "2018-01-01"), "oos": ("2018-01-01", "2024-01-01")}
start, end = (pd.Timestamp(x, tz="UTC") for x in PERIODS[MODE])

base = load_config("config.example.yaml")
base["timeframe"] = "1d"
base["signal"].pop("timeframe", None)
NAMES = ["GOLD", "SILVER", "BRENT", "NATGAS", "USA100", "NVDA", "MSFT", "AAPL"]
insts = [i for i in base["instruments"] if i["name"] in NAMES]

frames = {}
for inst in insts:
    df = data.load(inst, "1d")
    frames[inst["name"]] = df[(df.index >= "2008-01-01") & (df.index < end)]


def jumpy(df):
    return (atr(df, 14) > 1.5 * atr(df, 100)).fillna(False)


def frame(df, direction, atr_v):
    d = pd.Series(direction, index=df.index).fillna(0).astype(int)
    fire = (d != 0) & (d != d.shift(fill_value=0))
    return pd.DataFrame({"direction": d, "agree": 1, "size": 1.0, "fire": fire, "agreeing": "", "atr": atr_v},
                        index=df.index)


def run(df, inst, cfg, sig):
    orig = backtest.signals
    backtest.signals = lambda *_a, **_k: sig
    try:
        return [t for t in backtest.run_instrument(df, inst, cfg) if t.entry_time >= start]
    finally:
        backtest.signals = orig


def breakout_trades(df, inst, only_jumpy):
    j = jumpy(df).to_numpy()
    b = strategies.breakout(df, **base["strategies"]["breakout"])
    sig = frame(df, b, atr(df, base["exits"]["atr_period"]))
    trades = []
    for regime in (False, True):
        if only_jumpy and not regime:
            continue
        s = sig.copy()
        s["fire"] &= j if regime else ~j
        cfg = json.loads(json.dumps(base))
        i = dict(inst)
        if regime:
            i["spread_bps"] = inst.get("spread_bps", 0) * 3
            if only_jumpy:
                cfg["account"]["risk_per_trade"] = 0.005
                stop_dist = cfg["exits"]["stop_atr"] * s["atr"] / df["close"]
                s["fire"] &= (i["spread_bps"] / 10_000) <= 0.1 * stop_dist
        trades += run(df, i, cfg, s)
    return trades


def spike_fade_trades(df, inst):
    a = atr(df, 14)
    rng = df["high"] - df["low"]
    hi20 = df["close"].rolling(20).max().shift()
    lo20 = df["close"].rolling(20).min().shift()
    up = (rng >= 2 * a) & (df["close"] > hi20)
    down = (rng >= 2 * a) & (df["close"] < lo20)
    direction = np.zeros(len(df), dtype=int)
    extreme = np.full(len(df), np.nan)
    o, c, h, l = (df[k].to_numpy() for k in ("open", "close", "high", "low"))
    for k in np.flatnonzero(up.to_numpy() | down.to_numpy()):
        side = -1 if up.iloc[k] else 1  # fade: short an up spike, buy a down spike
        for m in range(k + 1, min(k + 6, len(df))):
            if (side == -1 and c[m] < o[k]) or (side == 1 and c[m] > o[k]):
                direction[m] = side
                extreme[m] = h[k] if side == -1 else l[k]
                break
    s = frame(df, direction, a)
    s["fire"] = s["direction"] != 0  # one-day events, not a state
    # Stop 0.5 ATR beyond the spike extreme, expressed as an ATR multiple for risk.levels.
    entry_ref = df["close"]
    stop_atr = (np.abs(entry_ref - pd.Series(extreme, index=df.index)) / a + 0.5)
    cfg = json.loads(json.dumps(base))
    cfg["account"]["risk_per_trade"] = 0.005
    cfg["exits"].update({"target_atr": 3.0, "max_days": 20})
    trades = []
    for k in np.flatnonzero(s["fire"].to_numpy()):
        one = s.copy()
        one["fire"] = False
        one.iloc[k, one.columns.get_loc("fire")] = True
        cfg["exits"]["stop_atr"] = float(stop_atr.iloc[k])
        i = dict(inst, spread_bps=inst.get("spread_bps", 0) * 3)
        trades += run(df, i, cfg, one)
    return trades


CANDIDATES = {
    "B0": lambda df, inst: breakout_trades(df, inst, only_jumpy=False),
    "B1": lambda df, inst: breakout_trades(df, inst, only_jumpy=True),
    "F1": lambda df, inst: spike_fade_trades(df, inst) if inst["name"] == "BRENT" else [],
}

for name, fn in CANDIDATES.items():
    if ONLY and name not in ONLY:
        continue
    cfg = json.loads(json.dumps(base))
    if name in ("B1", "F1"):
        cfg["account"]["max_open_positions"] = 2
    trades = []
    print(f"\n=== {name} {MODE} {start.date()}..{end.date()}")
    for inst in insts:
        tr = fn(frames[inst["name"]], inst)
        trades += tr
        m = backtest.metrics(tr, 10_000)
        if m["trades"]:
            print(f"  {inst['name']:7s} n={m['trades']:4d} win={m['win_rate']*100:5.1f} PF={m['profit_factor']:5.2f} "
                  f"pnl={m['total_pnl']:7.0f} DD={m['max_drawdown']:6.0f}")
    pm = backtest.metrics(backtest.portfolio(trades, cfg), 10_000)
    if pm["trades"]:
        print(f"  PORTFOLIO trades={pm['trades']} PF={pm['profit_factor']:.2f} exp=${pm['expectancy']:.2f} "
              f"pnl=${pm['total_pnl']:.0f} DD=${pm['max_drawdown']:.0f} sharpe={pm['sharpe']:.2f}")
    else:
        print("  PORTFOLIO no trades")
