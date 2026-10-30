"""How big the next days' moves will be, from daily bars.

research/volatility.py found that the size of moves is predictable (rank correlation of
about 0.75 out of sample) even though their direction is not. This is the live version of
that model, without the VIX inputs so it works for every market:

  target:  average true range of the next 10 days, as % of today's close
  inputs:  ATR(5), ATR(14), ATR(100) as % of close, and the largest 1-day move of the
           last 5 days in ATR(100) units, all known at today's close
  model:   least squares on the logs, fitted on the market's own history

The forecast is then ranked against the market's past forecasts: top 20% means a big move
is expected ("jumpy"), bottom 20% means small moves ("calm").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr

HORIZON = 10
MIN_FIT = 250  # daily bars with a known outcome needed to fit; fewer falls back to ATR(14)
FEATURES = ["a5", "a14", "a100", "shock"]


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
                     axis=1).max(axis=1)


def features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    a100 = atr(df, 100)
    out = pd.DataFrame({"a5": atr(df, 5) / c * 100, "a14": atr(df, 14) / c * 100, "a100": a100 / c * 100,
                        "shock": (c.diff().abs() / a100).rolling(5).max()}, index=df.index)
    out["target"] = true_range(df)[::-1].rolling(HORIZON).mean()[::-1].shift(-1) / c * 100
    return out


def _design(f: pd.DataFrame) -> np.ndarray:
    x = np.log(f[FEATURES].clip(lower=1e-6)).to_numpy()
    return np.column_stack([np.ones(len(x)), x])


def forecast(daily: pd.DataFrame) -> pd.DataFrame:
    """Per daily bar: `forecast` (expected daily range in % of close, next 10 days) and `pct`
    (where it ranks among all forecasts so far, 0..1). Earlier bars without enough history are NaN."""
    f = features(daily)
    ready = f[FEATURES].notna().all(axis=1)
    fit = ready & f["target"].notna()
    if fit.sum() >= MIN_FIT:
        beta, *_ = np.linalg.lstsq(_design(f[fit]), np.log(f.loc[fit, "target"]).to_numpy(), rcond=None)
        pred = pd.Series(np.nan, index=f.index)
        pred[ready] = np.exp(_design(f[ready]) @ beta)
    else:
        pred = f["a14"]
    # Rank against forecasts up to that day only, so the history shows what was known then.
    pct = pred.expanding(min_periods=60).rank(pct=True)
    return pd.DataFrame({"forecast": pred, "pct": pct, "actual": f["target"]})


def regime(pct: float | None) -> str:
    if pct is None or not np.isfinite(pct):
        return "unknown"
    return "jumpy" if pct >= 0.8 else "calm" if pct <= 0.2 else "normal"


def latest(daily: pd.DataFrame) -> dict:
    fc = forecast(daily).dropna(subset=["forecast"])
    if fc.empty:
        return {"forecast_pct": None, "rank": None, "regime": "unknown", "move": None}
    last = fc.iloc[-1]
    close = float(daily["close"].iloc[-1])
    rank = float(last["pct"]) if np.isfinite(last["pct"]) else None
    return {"forecast_pct": float(last["forecast"]), "rank": rank, "regime": regime(rank),
            "move": float(last["forecast"]) / 100 * close}
