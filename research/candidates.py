"""Run from the repo root: python research/candidates.py is   (or: oos C1)

In-sample-only research. The held-out last 30% of each series is cut off BEFORE any
strategy sees the data, so nothing here can leak out-of-sample information.

Pre-registered candidates (fixed before running):
  C0 baseline: current 1h three-strategy consensus
  C1 daily consensus: same three ideas computed on daily bars, exits in daily ATR
  C2 daily trend following: 55-day Donchian breakout, 3 ATR trailing stop, long/short
  C3 like C2 but long-only on equities (USA100, NVDA, MSFT, AAPL)
  C4 1h pullback only in the direction of the daily trend
All are executed on 1h bars (next 1h open after the signal is known) with full costs.
"""
import sys, json
import numpy as np, pandas as pd

from trader import backtest, consensus, data, strategies
from trader.config import load_config
from trader.indicators import atr, donchian, ema, rsi

EQUITIES = {"USA100", "NVDA", "MSFT", "AAPL"}
MODE = sys.argv[1] if len(sys.argv) > 1 else "is"   # "is" or "oos" (oos only run once at the end)
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None


def daily(df):
    return data.resample(df, "1D")


def to_hourly(dsig, df):
    """Daily values become usable from the first hour of the NEXT day."""
    d = dsig.copy()
    d.index = d.index + pd.Timedelta(days=1)
    key = pd.DataFrame({"t": df.index, "day": df.index.floor("D")})
    m = pd.merge_asof(key, d.reset_index(names="day").sort_values("day"), on="day", direction="backward")
    m.index = df.index
    return m.drop(columns=["t", "day"])


def finish(h, size_map=None):
    h = h.copy()
    h["direction"] = h["direction"].fillna(0).astype(int)
    h["agree"] = h.get("agree", pd.Series(1, index=h.index)).fillna(0).astype(int)
    if "size" not in h:
        h["size"] = np.where(h["direction"] != 0, 1.0, 0.0)
    h["size"] = h["size"].fillna(0.0)
    prev_d, prev_a = h["direction"].shift(fill_value=0), h["agree"].shift(fill_value=0)
    h["fire"] = (h["direction"] != 0) & ((h["direction"] != prev_d) | (h["agree"] > prev_a)) & (h["size"] > 0)
    if "agreeing" not in h:
        h["agreeing"] = ""
    h["agreeing"] = h["agreeing"].fillna("")
    return h


def c1(df, inst, cfg):
    d = daily(df)
    v = strategies.votes(d, cfg["strategies"])
    s = consensus.combine(v, 2, {3: 1.0, 2: 0.5})
    s["atr"] = atr(d, 14)
    return finish(to_hourly(s[["direction", "agree", "size", "agreeing", "atr"]], df))


def donchian_state(d, n=55):
    up, lo = donchian(d, n)
    ev = np.where(d["close"] > up, 1, np.where(d["close"] < lo, -1, np.nan))
    return pd.Series(ev, index=d.index).ffill().fillna(0)


def c2(df, inst, cfg, long_only=False):
    d = daily(df)
    s = pd.DataFrame({"direction": donchian_state(d), "atr": atr(d, 14)})
    if long_only:
        s["direction"] = s["direction"].clip(lower=0)
    s["agree"] = 1
    return finish(to_hourly(s, df))


def c3(df, inst, cfg):
    return c2(df, inst, cfg, long_only=inst["name"] in EQUITIES)


def c4(df, inst, cfg):
    d = daily(df)
    dt = strategies.trend(d, 50, 200)
    h = to_hourly(pd.DataFrame({"dtrend": dt}), df)["dtrend"].fillna(0)
    pb = strategies.pullback(df, 14, 40, 60, 3)
    direction = np.where((pb == h) & (h != 0), h, 0)
    out = pd.DataFrame({"direction": direction, "atr": atr(df, 14)}, index=df.index)
    out["agree"] = 1
    return finish(out)


CANDIDATES = {
    "C0": (None, None),
    "C1": (c1, {"stop_atr": 2.0, "target_atr": 3.0, "trail_atr": 0.0, "max_days": 15}),
    "C2": (c2, {"stop_atr": 3.0, "target_atr": 0.0, "trail_atr": 3.0, "max_days": 365}),
    "C3": (c3, {"stop_atr": 3.0, "target_atr": 0.0, "trail_atr": 3.0, "max_days": 365}),
    "C4": (c4, {"stop_atr": 2.0, "target_atr": 3.0, "trail_atr": 0.0, "max_days": 2}),
}

cfg = load_config("config.example.yaml")
orig_signals = backtest.signals
frames = {}
for inst in cfg["instruments"]:
    df = data.load(inst, "1h")
    cut = backtest.split(df, cfg["backtest"]["out_of_sample"])
    frames[inst["name"]] = (df[df.index < cut] if MODE == "is" else df, cut)

results = {}
for name, (fn, ex) in CANDIDATES.items():
    if ONLY and name not in ONLY:
        continue
    per, all_tr = {}, []
    for inst in cfg["instruments"]:
        df, cut = frames[inst["name"]]
        c = json.loads(json.dumps(cfg))
        if fn:
            bpd = df.groupby(df.index.floor("D")).size().median()
            c["exits"] = {"atr_period": 14, "stop_atr": ex["stop_atr"], "target_atr": ex["target_atr"],
                          "trail_atr": ex["trail_atr"], "max_bars": int(ex["max_days"] * bpd)}
            backtest.signals = lambda d_, c_, _i=inst, _f=fn: _f(d_, _i, c_)
        else:
            backtest.signals = orig_signals
        tr = backtest.run_instrument(df, inst, c)
        if MODE == "oos":
            tr = [t for t in tr if t.entry_time >= cut]
        m = backtest.metrics(tr, 10000)
        cost_r = np.mean([t.costs / t.risk for t in tr]) if tr else 0
        per[inst["name"]] = (m, cost_r)
        all_tr += tr
    port = backtest.portfolio(all_tr, cfg)
    pm = backtest.metrics(port, 10000)
    results[name] = pm
    print(f"\n=== {name}  portfolio: trades={pm.get('trades')} PF={pm.get('profit_factor', 0):.2f} "
          f"exp=${pm.get('expectancy', 0):.2f} pnl=${pm.get('total_pnl', 0):.0f} DD=${pm.get('max_drawdown', 0):.0f} "
          f"sharpe={pm.get('sharpe', 0):.2f}")
    for k, (m, cr) in per.items():
        if m.get("trades"):
            print(f"  {k:7s} n={m['trades']:4d} win={m['win_rate']*100:5.1f} PF={m['profit_factor']:5.2f} "
                  f"pnl={m['total_pnl']:7.0f} DD={m['max_drawdown']:6.0f} costR={cr:.2f}")
        else:
            print(f"  {k:7s} no trades")
