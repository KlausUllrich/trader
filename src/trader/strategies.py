"""Three independent strategies. Each returns a vote per bar: +1 long, -1 short, 0 neutral.

They come from different families so they don't all fail in the same market:
- trend:    where is the market heading (EMA alignment)
- breakout: is there fresh momentum (recent range breakout + rate of change)
- pullback: is a short-term dip/rally just ending (RSI turning back)

A vote only uses data up to and including the bar it is computed on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import donchian, ema, rsi


def trend(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    f, s = ema(df["close"], fast), ema(df["close"], slow)
    close = df["close"]
    vote = np.where((f > s) & (close > f), 1, np.where((f < s) & (close < f), -1, 0))
    return pd.Series(vote, index=df.index).where(s.notna(), 0).astype(int)


def breakout(df: pd.DataFrame, lookback: int = 20, recent: int = 5, roc_period: int = 24) -> pd.Series:
    upper, lower = donchian(df, lookback)
    close = df["close"]
    broke_up = (close > upper).astype(int).rolling(recent, min_periods=1).max().astype(bool)
    broke_down = (close < lower).astype(int).rolling(recent, min_periods=1).max().astype(bool)
    roc = close.pct_change(roc_period)
    long = broke_up & (roc > 0) & ~broke_down
    short = broke_down & (roc < 0) & ~broke_up
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index).astype(int)


def pullback(df: pd.DataFrame, rsi_period: int = 14, low: float = 40, high: float = 60, recent: int = 3) -> pd.Series:
    r = rsi(df["close"], rsi_period)
    prev = r.shift()
    turned_up = ((prev < low) & (r >= low)).astype(int).rolling(recent, min_periods=1).max().astype(bool)
    turned_down = ((prev > high) & (r <= high)).astype(int).rolling(recent, min_periods=1).max().astype(bool)
    long = turned_up & ~turned_down
    short = turned_down & ~turned_up
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index).astype(int)


REGISTRY = {"trend": trend, "breakout": breakout, "pullback": pullback}


def votes(df: pd.DataFrame, params: dict, inst: dict | None = None) -> pd.DataFrame:
    """One column per strategy with its vote on every bar.

    If `params` has a `cot` entry and the instrument has a `cot` contract, the large
    traders' positioning (see cot.py) votes as a fourth strategy.
    """
    out = pd.DataFrame({name: fn(df, **params.get(name, {})) for name, fn in REGISTRY.items()})
    if "cot" in params and inst and inst.get("cot"):
        from . import cot

        out["cot"] = cot.strategy(df, inst, **params["cot"])
    return out
