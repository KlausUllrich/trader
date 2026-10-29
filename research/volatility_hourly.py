"""The small-stop breakout from volatility.py, replayed on 1h bars so the order of events
inside a day is known. Run from the repo root: python research/volatility_hourly.py

Same rules as Part 2 of volatility.py, fixed before running. Daily bars are built from the
1h bars (UTC days). "Big move expected" = ATR(14) as % of price in the top 20% of its own
previous 250 days (ATR alone forecast almost as well as the full model). Orders work on
the 1h bars of day D+1; the trade then runs on 1h bars for up to 5 days. Only inside a
single hour is the order unknown, and there the stop is assumed to hit first.
Data: the cached 1h history (futures and stocks from 2024, crypto from 2022). These periods
were used before for direction tests, not for this rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader import data
from trader.config import load_config
from trader.indicators import atr

cfg = load_config("config.example.yaml")
risk = 50.0


def run(inst, only_big):
    h1 = data.load(inst, "1h")
    d = data.resample(h1, "1D")
    a = atr(d, 14)
    pct = a / d["close"]
    big = pct >= pct.rolling(250, min_periods=120).quantile(0.8).shift()
    spread = 3 * inst.get("spread_bps", 0) / 1e4
    commission = inst.get("commission", 0.0)
    ho, hh, hl, hc = (h1[k].to_numpy() for k in ("open", "high", "low", "close"))
    day_of = h1.index.floor("D")
    trades, busy_until = [], None
    days = d.index
    for i in range(len(days) - 6):
        if not np.isfinite(a.iloc[i]) or (only_big and not big.iloc[i]):
            continue
        nxt = days[i + 1]
        if busy_until is not None and nxt <= busy_until:
            continue
        c0, a0 = d["close"].iloc[i], a.iloc[i]
        up, dn = c0 + 0.5 * a0, c0 - 0.5 * a0
        bars = np.flatnonzero(day_of == nxt)
        if not len(bars):
            continue
        entry_bar, side = None, 0
        for j in bars:
            u, w = hh[j] >= up, hl[j] <= dn
            if u and w:  # both inside one hour: take the losing side (worst case)
                side = 1 if hc[j] < c0 else -1
            elif u or w:
                side = 1 if u else -1
            if side:
                entry_bar = j
                break
        if entry_bar is None:
            continue
        j = entry_bar
        entry = max(ho[j], up) if side == 1 else min(ho[j], dn)
        stop, target = entry - side * 0.5 * a0, entry + side * 2.0 * a0
        units = risk / (0.5 * a0)
        horizon_end = days[min(i + 6, len(days) - 1)]
        exit_px, k = None, j
        for k in range(j, len(h1)):
            if h1.index[k] >= horizon_end:
                exit_px = ho[k]
                break
            if (hl[k] <= stop) if side == 1 else (hh[k] >= stop):
                exit_px = stop if k == j else (min(ho[k], stop) if side == 1 else max(ho[k], stop))
                break
            if (hh[k] >= target) if side == 1 else (hl[k] <= target):
                exit_px = target
                break
        if exit_px is None:
            exit_px = hc[k]
        pnl = side * (exit_px - entry) * units - units * entry * (spread + 2 * commission)
        trades.append(pnl)
        busy_until = h1.index[k].floor("D")
    return trades


def summary(p):
    p = np.array(p)
    if not len(p):
        return "no trades"
    pf = p[p > 0].sum() / -p[p < 0].sum()
    return f"{len(p):5d} trades, win {np.mean(p > 0):4.0%}, PF {pf:4.2f}, avg ${p.mean():6.2f}, total ${p.sum():7.0f}"


for only_big in (True, False):
    allp = []
    for inst in cfg["instruments"]:
        t = run(inst, only_big)
        allp += t
        print(f"  {inst['name']:7s} {'big' if only_big else 'all'}: {summary(t)}")
    print(f"ALL {'big-move days' if only_big else 'every day'}: {summary(allp)}\n")
