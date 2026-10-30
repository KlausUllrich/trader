"""Support and resistance from daily swing highs and lows.

A swing high is a day whose high beats the `wing` days on each side (so it is only known
`wing` days later); swing lows likewise. Swings closer than `merge_atr` x ATR(14) are merged
into one level, and each level counts how many swings formed it. Only levels that price has
not closed through since they formed are kept as "intact"; the nearest ones are returned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr


def swings(df: pd.DataFrame, wing: int = 5) -> list[tuple[pd.Timestamp, float, int]]:
    """(time, price, +1 high / -1 low) for each swing point."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    out = []
    for i in range(wing, len(df) - wing):
        win_h, win_l = h[i - wing:i + wing + 1], l[i - wing:i + wing + 1]
        if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
            out.append((df.index[i], float(h[i]), 1))
        if l[i] == win_l.min() and (win_l == l[i]).sum() == 1:
            out.append((df.index[i], float(l[i]), -1))
    return out


def find(daily: pd.DataFrame, wing: int = 5, merge_atr: float = 0.5, lookback: int = 500,
         nearest: int = 3) -> list[dict]:
    """Up to `nearest` intact levels above and below the last close, nearest first."""
    df = daily.iloc[-lookback:]
    if len(df) < 2 * wing + 2:
        return []
    a = atr(df, 14).iloc[-1]
    a = float(a) if np.isfinite(a) else float((df["high"] - df["low"]).mean())
    close = float(df["close"].iloc[-1])
    closes = df["close"]

    clusters: list[dict] = []
    for t, price, kind in sorted(swings(df, wing), key=lambda s: s[1]):
        if clusters and price - clusters[-1]["prices"][-1] <= merge_atr * a:
            c = clusters[-1]
        else:
            c = {"prices": [], "times": []}
            clusters.append(c)
        c["prices"].append(price)
        c["times"].append(t)

    levels = []
    for c in clusters:
        price = float(np.mean(c["prices"]))
        last_touch = max(c["times"])
        later = closes[closes.index > last_touch + pd.Timedelta(days=wing)]
        # Broken = a later close on the far side of the level from where price is now.
        side = 1 if price > close else -1
        if side == 1 and (later > price + 0.25 * a).any():
            continue
        if side == -1 and (later < price - 0.25 * a).any():
            continue
        levels.append({"price": price, "kind": "resistance" if side == 1 else "support",
                       "touches": len(c["prices"]), "last": last_touch,
                       "distance_atr": abs(price - close) / a if a else None})
    above = sorted((x for x in levels if x["kind"] == "resistance"), key=lambda x: x["price"])[:nearest]
    below = sorted((x for x in levels if x["kind"] == "support"), key=lambda x: -x["price"])[:nearest]
    return above + below
