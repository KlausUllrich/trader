"""Chart-pattern ideas as fixed rules. Run from the repo root:
    python research/chart_daily.py is          # design period 2010-2017
    python research/chart_daily.py oos E1      # check period 2018-2023, run once for a pick

Daily bars from Yahoo, 8 markets; everything from 2024-06 on is cut off (already used).
Signal on the close of day D, entry at the open of day D+1, normal costs.

Pre-registered candidates (fixed before running):
  E1  Elliott wave 3, objective proxy. Swings come from a zigzag on closes that turns
      after a 3 x ATR(14) move against the last extreme (a swing is only known once that
      move has happened, so no hindsight). Setup: wave 1 = low L0 -> high H1, wave 2 =
      pullback to low L2 that retraces 38.2-78.6% of wave 1 and stays above L0. Entry:
      first close above H1 while the swing from L2 is still running. Stop 0.25 ATR below
      L2, target L2 + 1.618 x wave 1, time stop 40 days. Mirror image for shorts.
  L1  Leading candle, simple form: today's candle is the opposite colour of yesterday's
      and its body is larger than yesterday's whole range. Trade in today's direction.
      Exits from the config (2 ATR stop, 3 ATR target, 15 days).
  M1  Multi-timeframe support/resistance. Weekly swing high/low = a week whose high (low)
      beats the 2 weeks on each side, known 2 weeks later. Long after two daily closes in
      a row above the latest weekly swing high, while the weekly close is above its
      20-week EMA. Mirror for shorts. Exits from the config.
"""
import json
import sys

import numpy as np
import pandas as pd

from trader import backtest, data
from trader.config import load_config
from trader.indicators import atr, ema

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


def run_one(df, inst, k, direction, a, exits):
    """One trade fired on bar k, with the engine's entry, costs and exit handling."""
    sig = pd.DataFrame({"direction": 0, "agree": 1, "size": 1.0, "fire": False, "agreeing": "", "atr": a},
                       index=df.index)
    sig.iloc[k, sig.columns.get_loc("direction")] = direction
    sig.iloc[k, sig.columns.get_loc("fire")] = True
    cfg = json.loads(json.dumps(base))
    cfg["exits"].update(exits)
    orig = backtest.signals
    backtest.signals = lambda *_a, **_k: sig
    try:
        return backtest.run_instrument(df, inst, cfg)
    finally:
        backtest.signals = orig


def run_state(df, inst, direction, a):
    """Fire on each new non-zero state with the config exits."""
    d = pd.Series(direction, index=df.index).fillna(0).astype(int)
    sig = pd.DataFrame({"direction": d, "agree": 1, "size": 1.0, "fire": (d != 0) & (d != d.shift(fill_value=0)),
                        "agreeing": "", "atr": a}, index=df.index)
    orig = backtest.signals
    backtest.signals = lambda *_a, **_k: sig
    try:
        return backtest.run_instrument(df, inst, base)
    finally:
        backtest.signals = orig


def zigzag(close, a, mult=3.0):
    """Swing pivots as (pivot index, confirmed index, price, +1 high / -1 low)."""
    c, pivots = close.to_numpy(), []
    trend, ext_i, hi_i, lo_i = 0, 0, 0, 0
    for t in range(len(c)):
        if not np.isfinite(a[t]):
            ext_i = hi_i = lo_i = t
            continue
        move = mult * a[t]
        if trend == 0:
            hi_i = t if c[t] > c[hi_i] else hi_i
            lo_i = t if c[t] < c[lo_i] else lo_i
            if c[hi_i] - c[t] > move:
                pivots.append((hi_i, t, c[hi_i], 1)); trend, ext_i = -1, t
            elif c[t] - c[lo_i] > move:
                pivots.append((lo_i, t, c[lo_i], -1)); trend, ext_i = 1, t
        elif trend == 1:
            if c[t] >= c[ext_i]:
                ext_i = t
            elif c[ext_i] - c[t] > move:
                pivots.append((ext_i, t, c[ext_i], 1)); trend, ext_i = -1, t
        else:
            if c[t] <= c[ext_i]:
                ext_i = t
            elif c[t] - c[ext_i] > move:
                pivots.append((ext_i, t, c[ext_i], -1)); trend, ext_i = 1, t
    return pivots


def elliott(df, inst):
    a = atr(df, 14)
    av, c = a.to_numpy(), df["close"].to_numpy()
    piv = zigzag(df["close"], av)
    trades, used = [], set()
    for n in range(2, len(piv)):
        (i0, _, p0, s0), (i1, _, p1, s1), (i2, conf2, p2, s2) = piv[n - 2], piv[n - 1], piv[n]
        nxt = piv[n + 1][1] if n + 1 < len(piv) else len(c)
        if s2 == -1 and s1 == 1:   # L0 -> H1 -> L2, look for a break above H1
            d, w1 = 1, p1 - p0
            ok = w1 > 0 and p2 > p0 and 0.382 <= (p1 - p2) / w1 <= 0.786
        elif s2 == 1 and s1 == -1:  # H0 -> L1 -> H2, look for a break below L1
            d, w1 = -1, p0 - p1
            ok = w1 > 0 and p2 < p0 and 0.382 <= (p2 - p1) / w1 <= 0.786
        else:
            continue
        if not ok:
            continue
        for k in range(conf2, nxt):
            if (d == 1 and c[k] > p1) or (d == -1 and c[k] < p1):
                stop = p2 - d * 0.25 * av[k]
                target = p2 + d * 1.618 * w1
                if d * (target - c[k]) <= 0 or d * (c[k] - stop) <= 0:
                    break
                exits = {"stop_atr": abs(c[k] - stop) / av[k], "target_atr": abs(target - c[k]) / av[k],
                         "max_days": 40, "trail_atr": 0}
                if k not in used:
                    used.add(k)
                    trades += run_one(df, inst, k, d, a, exits)
                break
    return trades


def leading_candle(df, inst):
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    body, prev_body, prev_range = c - o, (c - o).shift(), (h - l).shift()
    opposite = np.sign(body) == -np.sign(prev_body)
    bigger = body.abs() > prev_range
    direction = np.where(opposite & bigger, np.sign(body), 0)
    s = pd.Series(direction, index=df.index)
    # Each leading candle is its own event: fire on that day even if the last one was the same side.
    trades, a = [], atr(df, 14)
    for k in np.flatnonzero(s.to_numpy()):
        trades += run_one(df, inst, k, int(s.iloc[k]), a, {})
    return _no_overlap(trades)


def mtf_levels(df, inst):
    w = data.resample(df, "W-FRI")
    hi, lo = w["high"], w["low"]
    swing_hi = (hi > hi.shift(1)) & (hi > hi.shift(2)) & (hi > hi.shift(-1)) & (hi > hi.shift(-2))
    swing_lo = (lo < lo.shift(1)) & (lo < lo.shift(2)) & (lo < lo.shift(-1)) & (lo < lo.shift(-2))
    # A swing week is only known 2 weeks later; weekly values are used from the next week on.
    known_hi = hi.where(swing_hi).shift(2).ffill()
    known_lo = lo.where(swing_lo).shift(2).ffill()
    up_trend = w["close"] > ema(w["close"], 20)
    weekly = pd.DataFrame({"res": known_hi, "sup": known_lo, "up": up_trend.astype(float)})
    weekly.index = weekly.index + pd.Timedelta(days=1)  # Friday's bar is usable from Saturday
    m = pd.merge_asof(pd.DataFrame({"t": df.index}), weekly.rename_axis("t").reset_index(), on="t")
    m.index = df.index
    c = df["close"]
    above = (c > m["res"]) & (c.shift() > m["res"])
    below = (c < m["sup"]) & (c.shift() < m["sup"])
    direction = np.where(above & (m["up"] == 1), 1, np.where(below & (m["up"] == 0), -1, 0))
    return run_state(df, inst, direction, atr(df, 14))


def _no_overlap(trades):
    out, busy_until = [], None
    for t in sorted(trades, key=lambda t: t.entry_time):
        if busy_until is None or t.entry_time > busy_until:
            out.append(t)
            busy_until = t.exit_time
    return out


CANDIDATES = {"E1": elliott, "L1": leading_candle, "M1": mtf_levels}

for name, fn in CANDIDATES.items():
    if ONLY and name not in ONLY:
        continue
    trades = []
    print(f"\n=== {name} {MODE} {start.date()}..{end.date()}")
    for inst in insts:
        tr = [t for t in fn(frames[inst["name"]], inst) if t.entry_time >= start]
        trades += tr
        m = backtest.metrics(tr, 10_000)
        if m["trades"]:
            print(f"  {inst['name']:7s} n={m['trades']:4d} win={m['win_rate']*100:5.1f} PF={m['profit_factor']:5.2f} "
                  f"pnl={m['total_pnl']:7.0f} DD={m['max_drawdown']:6.0f}")
    pm = backtest.metrics(backtest.portfolio(trades, base), 10_000)
    if pm["trades"]:
        print(f"  PORTFOLIO trades={pm['trades']} PF={pm['profit_factor']:.2f} exp=${pm['expectancy']:.2f} "
              f"pnl=${pm['total_pnl']:.0f} DD=${pm['max_drawdown']:.0f} sharpe={pm['sharpe']:.2f}")
    else:
        print("  PORTFOLIO no trades")
