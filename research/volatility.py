"""Volatility as the lead trigger. Run from the repo root: python research/volatility.py

Fixed before running.

Markets: daily Yahoo bars for gold, silver, Brent, natural gas, Nasdaq, NVDA, MSFT, AAPL.
Periods: fit on 2010-2017; test on 2000-2009 (never used in any earlier test) and on
2018-2023 (used before, but only for direction, not volatility).

Part 1, can big-move days be predicted?
  Target: average true range of the next 10 days, divided by today's close (in %).
  Inputs (all known at today's close): ATR(5), ATR(14), ATR(100) as % of price, the
  largest 1-day move of the last 5 days in ATR(100) units, the VIX (for stocks and Nasdaq)
  or the oil VIX (for Brent and gas, from 2007) as a level and its 5-day change.
  Model: ordinary least squares on the logs, one model for all markets, fitted on 2010-2017.
  Score: rank correlation of forecast vs actual, and the lift for "big move expected" days =
  forecast in the market's top 20% (thresholds from the fit period): how often the actual
  next 10 days are also in that market's top 20%. Random = 20%.

Part 2, does a small-stop breakout make money on those days?
  Signal day D. On day D+1 a buy stop at close + 0.5 ATR(14) and a sell stop at close -
  0.5 ATR(14); the first one hit is the trade, the other is cancelled. Stop 0.5 ATR beyond
  the entry, target 2 ATR, time stop 5 days. A gap through an order fills at the open.
  Daily bars cannot show the order within a day, so two bounds are reported:
    worst: if both orders could fill on D+1, it is the wrong one; if entry and stop are
           both inside one bar, the stop hits first
    best:  the right order fills, and an entry bar never also hits its own stop
  Costs: 3x the normal spread, risk 0.5% of $10,000 per trade.
  Compared: big-move-expected days only vs every day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader import data
from trader.config import load_config
from trader.indicators import atr

NAMES = ["GOLD", "SILVER", "BRENT", "NATGAS", "USA100", "NVDA", "MSFT", "AAPL"]
STOCKS = {"USA100", "NVDA", "MSFT", "AAPL"}
ENERGY = {"BRENT", "NATGAS"}
FIT = ("2010-01-01", "2018-01-01")
TESTS = {"2000-2009 (untouched)": ("2000-01-01", "2010-01-01"), "2018-2023": ("2018-01-01", "2024-01-01")}
END = "2024-01-01"

cfg = load_config("config.example.yaml")
insts = {i["name"]: i for i in cfg["instruments"] if i["name"] in NAMES}


def series(sym, name):
    df = data.load({"name": f"X_{name}", "source": "yahoo", "symbol": sym}, "1d")
    s = df["close"]
    s.index = s.index.floor("D")
    return s[~s.index.duplicated(keep="last")]


vix, ovx = series("^VIX", "vix"), series("^OVX", "ovx")


def features(name, df):
    c = df["close"]
    a5, a14, a100 = atr(df, 5), atr(df, 14), atr(df, 100)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()],
                   axis=1).max(axis=1)
    fwd = tr[::-1].rolling(10).mean()[::-1].shift(-1) / c * 100  # next 10 days' average true range, % of close
    shock = (c.diff().abs() / a100).rolling(5).max()
    fear = vix if name in STOCKS else ovx if name in ENERGY else None
    f = pd.DataFrame({"a5": a5 / c * 100, "a14": a14 / c * 100, "a100": a100 / c * 100, "shock": shock}, index=df.index)
    if fear is not None:
        lvl = fear.reindex(df.index.floor("D"), method="ffill").to_numpy()
        f["fear"] = lvl
        f["fear_chg"] = np.log(pd.Series(lvl, index=df.index)).diff(5)
    else:
        f["fear"], f["fear_chg"] = np.nan, np.nan
    f["target"] = fwd
    return f


frames, feats = {}, {}
for n in NAMES:
    df = data.load(insts[n], "1d")
    df = df[df.index < END]
    frames[n], feats[n] = df, features(n, df)

COLS = ["a5", "a14", "a100", "shock"]


def design(f, with_fear):
    x = np.log(f[COLS].clip(lower=1e-6))
    if with_fear:
        x["fear"] = np.log(f["fear"])
        x["fear_chg"] = f["fear_chg"]
    x.insert(0, "const", 1.0)
    return x


def period(f, p):
    return f[(f.index >= p[0]) & (f.index < p[1])]


# ---- Part 1: forecast ----
models = {}
for with_fear in (False, True):
    rows = [period(feats[n], FIT) for n in NAMES if (not with_fear or n in STOCKS | ENERGY)]
    fit = pd.concat(rows)
    x = design(fit, with_fear)
    ok = x.notna().all(axis=1) & fit["target"].notna()
    beta, *_ = np.linalg.lstsq(x[ok].to_numpy(), np.log(fit.loc[ok, "target"]).to_numpy(), rcond=None)
    models[with_fear] = beta


def forecast(n, f):
    use_fear = n in STOCKS | ENERGY and f["fear"].notna().any()
    x = design(f, use_fear)
    pred = np.exp(x.to_numpy() @ models[use_fear])
    pred = pd.Series(pred, index=f.index)
    if use_fear:  # fall back to the model without the fear index where it is missing (e.g. OVX before 2007)
        x0 = design(f, False)
        pred = pred.fillna(pd.Series(np.exp(x0.to_numpy() @ models[False]), index=f.index))
    return pred


print("Part 1: forecasting the next 10 days' volatility")
cut = {}
for n in NAMES:
    f = feats[n]
    fit = period(f, FIT)
    pf = forecast(n, fit)
    cut[n] = (pf.quantile(0.8), fit["target"].quantile(0.8))
for label, p in [("fit 2010-2017", FIT), *TESTS.items()]:
    rc, rc_naive, lift, n_big = [], [], [], 0
    for n in NAMES:
        f = period(feats[n], p)
        pred = forecast(n, f)
        ok = pred.notna() & f["target"].notna()
        if ok.sum() < 200:
            continue
        rc.append(pred[ok].rank().corr(f.loc[ok, "target"].rank()))
        rc_naive.append(f.loc[ok, "a14"].rank().corr(f.loc[ok, "target"].rank()))
        big = pred[ok] >= cut[n][0]
        n_big += int(big.sum())
        lift.append(float((f.loc[ok, "target"][big] >= cut[n][1]).mean()))
    print(f"  {label:22s} rank corr {np.mean(rc):.2f} (ATR14 alone {np.mean(rc_naive):.2f}); "
          f"'big move expected' days: {n_big}, of which really big: {np.mean(lift):.0%} (random 20%)")


# ---- Part 2: small-stop breakout ----
def breakout(n, p, only_big, bound):
    df, f = frames[n], feats[n]
    pred = forecast(n, f)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df, 14).to_numpy()
    spread = 3 * insts[n].get("spread_bps", 0) / 1e4
    commission = insts[n].get("commission", 0.0)
    risk = 50.0  # 0.5% of $10,000
    idx = np.flatnonzero((df.index >= p[0]) & (df.index < p[1]))
    pnl, i = [], 0
    for i in idx:
        if i + 6 >= len(df) or not np.isfinite(a[i]):
            continue
        if pnl and i <= pnl[-1][0]:
            continue  # one trade at a time
        if only_big and not pred.iloc[i] >= cut[n][0]:
            continue
        up, dn = c[i] + 0.5 * a[i], c[i] - 0.5 * a[i]
        j = i + 1
        hit_up, hit_dn = h[j] >= up, l[j] <= dn
        if not (hit_up or hit_dn):
            continue
        if hit_up and hit_dn:
            # worst: the order that loses; best: the one that wins (decided by the close)
            wins_up = c[j] > c[i]
            d = (1 if not wins_up else -1) if bound == "worst" else (1 if wins_up else -1)
        else:
            d = 1 if hit_up else -1
        entry = (max(o[j], up) if d == 1 else min(o[j], dn))
        stop, target = entry - d * 0.5 * a[i], entry + d * 2.0 * a[i]
        units = risk / (0.5 * a[i])
        exit_px, k = None, j
        for k in range(j, min(j + 5, len(df))):
            stop_hit = l[k] <= stop if d == 1 else h[k] >= stop
            tgt_hit = h[k] >= target if d == 1 else l[k] <= target
            if k == j and bound == "best":
                stop_hit = stop_hit and (c[k] - stop) * d <= 0  # only if the day closed beyond the stop
            if stop_hit:
                exit_px = (min(o[k], stop) if d == 1 else max(o[k], stop)) if k > j else stop
                break
            if tgt_hit:
                exit_px = target
                break
        if exit_px is None:
            exit_px = c[k]
        gross = d * (exit_px - entry) * units
        costs = units * entry * (spread + 2 * commission)
        pnl.append((k, gross - costs))
    return [x for _, x in pnl]


def summary(p_list):
    p = np.array(p_list)
    if not len(p):
        return "no trades"
    pf = p[p > 0].sum() / -p[p < 0].sum() if (p < 0).any() else float("inf")
    return f"{len(p):5d} trades, win {np.mean(p > 0):4.0%}, PF {pf:4.2f}, avg ${p.mean():6.2f}, total ${p.sum():7.0f}"


print("\nPart 2: buy/sell stop 0.5 ATR away, stop 0.5 ATR, target 2 ATR, 5 days, 3x spread")
for label, p in [("fit 2010-2017", FIT), *TESTS.items()]:
    for only_big in (True, False):
        for bound in ("worst", "best"):
            tr = [x for n in NAMES for x in breakout(n, p, only_big, bound)]
            days = "big-move days" if only_big else "every day    "
            print(f"  {label:22s} {days} {bound:5s}: {summary(tr)}")
