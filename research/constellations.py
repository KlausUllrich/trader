"""Search for rare constellations of conditions with a high hit rate. Run from the repo root:
    python research/constellations.py is              # mine 2010-2017, with shuffled-data baseline
    python research/constellations.py oos             # check the survivors saved by the "is" run, once

Everything below was fixed before the first run.

Data: daily bars (Yahoo) for 8 markets plus Bitcoin, cut at 2024-06 (later data already used).
Each day D gets ~30 yes/no conditions computed from data up to D's close (COT only from 7
days after its report date). A constellation is 1 to 3 conditions that are all true, for
a long or a short.

Outcome per event: enter at the open of D+1. A hit = price reaches 2 ATR(14) in the trade's
direction before 2 ATR against it, within 10 days (both in one day counts as a miss). With
no drift the hit rate is about 50%. R per event: +1 hit, -1 miss, else the 10-day move in
units of 2 ATR, minus the spread in R.

Events are de-clustered: after an event, the same constellation on the same market is
ignored for 10 days, so one move is never counted twice.

A constellation survives only if, on 2010-2017:
- at least 50 events, on at least 3 markets
- hit rate at least 60% and at least 10 points above the base rate of those markets
- average R after spread of at least +0.2
- one-sided binomial p-value below 0.05 / (number of constellations tested)  (Bonferroni)
Luck check: the same mining runs on 10 shuffled copies, where each market's outcomes are
shifted in time against its conditions by a random 1-5 years (keeps each series' patterns,
breaks any link between them). If shuffled copies produce survivors too, the real ones
mean little.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trader import cot, data
from trader.config import load_config
from trader.indicators import atr, ema, rsi

MODE = sys.argv[1] if len(sys.argv) > 1 else "is"
PERIODS = {"is": ("2010-01-01", "2018-01-01"), "oos": ("2018-01-01", "2024-01-01")}
HORIZON, BARRIER = 10, 2.0
SURVIVORS = Path("research/constellations_survivors.json")

cfg = load_config("config.example.yaml")
insts = [i for i in cfg["instruments"] if i["source"] == "yahoo"]
insts.append({"name": "BTC", "source": "yahoo", "symbol": "BTC-USD", "spread_bps": 10})


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), exact, in log space."""
    if k <= 0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k, n + 1)]
    top = max(terms)
    return math.exp(top) * sum(math.exp(t - top) for t in terms)


# ---------- conditions ----------

def volume_profile(df, window=120, bins=24, top=3):
    """Per day: is the close inside one of the `top` busiest price bins of the last `window` days,
    and is it above the busiest bin (point of control)?"""
    c, v = df["close"].to_numpy(), df["volume"].to_numpy()
    at_hvn, above_poc = np.zeros(len(df), bool), np.zeros(len(df), bool)
    if np.nansum(v) == 0:
        return at_hvn, above_poc, False
    for t in range(window, len(df)):
        pc, pv = c[t - window + 1:t + 1], v[t - window + 1:t + 1]
        hist, edges = np.histogram(pc, bins=bins, weights=pv)
        k = min(np.searchsorted(edges, c[t], side="right") - 1, bins - 1)
        busiest = np.argsort(hist)[-top:]
        at_hvn[t] = k in busiest
        poc = np.argmax(hist)
        above_poc[t] = c[t] > edges[poc + 1]
    return at_hvn, above_poc, True


def conditions(df, inst):
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    a14, a100 = atr(df, 14), atr(df, 100)
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi(c, 14)
    rp = r.shift()
    hi20, lo20 = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    low20c, high20c = c.rolling(20).min(), c.rolling(20).max()
    recent = lambda s, n=3: s.astype(int).rolling(n, min_periods=1).max().astype(bool)
    f = {
        "above_ema200": c > e200, "below_ema200": c < e200,
        "ema50_over_200": e50 > e200, "ema50_under_200": e50 < e200,
        "rsi_below_30": r < 30, "rsi_above_70": r > 70,
        "rsi_back_above_30": recent((rp < 30) & (r >= 30)), "rsi_back_below_70": recent((rp > 70) & (r <= 70)),
        "rsi_cross_up_50": (rp < 50) & (r >= 50), "rsi_cross_down_50": (rp > 50) & (r <= 50),
        "bull_divergence": recent((c <= low20c) & (r > r.rolling(20).min().shift() + 5)),
        "bear_divergence": recent((c >= high20c) & (r < r.rolling(20).max().shift() - 5)),
        "pullback_to_ema50_up": (e50 > e200) & ((c - e50).abs() < a14),
        "pullback_to_ema50_down": (e50 < e200) & ((c - e50).abs() < a14),
        "stretched_up": c > e20 + 2 * a14, "stretched_down": c < e20 - 2 * a14,
        "jumpy": a14 > 1.5 * a100, "calm": a14 < 0.75 * a100,
        "breakout_20d_high": c > hi20, "breakout_20d_low": c < lo20,
        "bull_candle_engulf": (c > o) & ((c - o) > (h - l).shift()) & (c.shift() < o.shift()),
        "bear_candle_engulf": (c < o) & ((o - c) > (h - l).shift()) & (c.shift() > o.shift()),
        "up_3_days": (c.diff() > 0).rolling(3).sum() == 3, "down_3_days": (c.diff() < 0).rolling(3).sum() == 3,
    }
    # Weekly levels: latest weekly swing high/low, known 2 weeks after the swing week.
    w = data.resample(df, "W-FRI")
    wh, wl = w["high"], w["low"]
    sh = (wh > wh.shift(1)) & (wh > wh.shift(2)) & (wh > wh.shift(-1)) & (wh > wh.shift(-2))
    sl = (wl < wl.shift(1)) & (wl < wl.shift(2)) & (wl < wl.shift(-1)) & (wl < wl.shift(-2))
    lv = pd.DataFrame({"res": wh.where(sh).shift(2).ffill(), "sup": wl.where(sl).shift(2).ffill()})
    lv.index = lv.index + pd.Timedelta(days=1)
    m = pd.merge_asof(pd.DataFrame({"t": df.index}), lv.rename_axis("t").reset_index(), on="t").set_index("t")
    m.index = df.index
    f["above_weekly_resistance"] = c > m["res"]
    f["below_weekly_support"] = c < m["sup"]
    f["near_weekly_support"] = ((c - m["sup"]).abs() < a14) & (c >= m["sup"])
    f["near_weekly_resistance"] = ((m["res"] - c).abs() < a14) & (c <= m["res"])
    at_hvn, above_poc, has_vol = volume_profile(df)
    if has_vol:
        f["at_high_volume_zone"] = pd.Series(at_hvn, index=df.index)
        f["above_volume_poc"] = pd.Series(above_poc, index=df.index)
    if inst.get("cot"):
        rep = cot.load(str(inst["cot"]["code"]), inst["cot"].get("report", "disagg"))
        follow = cot.align(cot.weekly_vote(rep, "follow"), df.index)
        hedgers = cot.align(cot.weekly_vote(rep, "hedgers"), df.index)
        f["cot_specs_buying"], f["cot_specs_selling"] = follow == 1, follow == -1
        f["cot_hedgers_bullish"], f["cot_hedgers_bearish"] = hedgers == 1, hedgers == -1
    out = pd.DataFrame(f).fillna(False).astype(bool)
    out[a14.isna() | a100.isna() | e200.isna()] = False
    return out


def outcomes(df, inst):
    """Hit (+1), miss (-1) or 10-day result in R, for a long and a short from each day."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df, 14).to_numpy()
    n = len(df)
    res = {}
    for d in (1, -1):
        r = np.full(n, np.nan)
        for i in range(n - HORIZON - 1):
            if not np.isfinite(a[i]) or a[i] <= 0:
                continue
            entry = o[i + 1]
            tgt, stp = entry + d * BARRIER * a[i], entry - d * BARRIER * a[i]
            val = None
            for j in range(i + 1, i + 1 + HORIZON):
                hit_s = l[j] <= stp if d == 1 else h[j] >= stp
                hit_t = h[j] >= tgt if d == 1 else l[j] <= tgt
                if hit_s:
                    val = -1.0
                    break
                if hit_t:
                    val = 1.0
                    break
            if val is None:
                val = d * (c[i + HORIZON] - entry) / (BARRIER * a[i])
            r[i] = val
        res[d] = r
    # Spread in R, charged on every event.
    res["cost"] = inst.get("spread_bps", 0) / 1e4 * o / (BARRIER * np.where(a > 0, a, np.nan))
    return res


# ---------- mining ----------

def load_market(inst, start, end):
    df = data.load(inst, "1d")
    df = df[(df.index >= "2008-01-01") & (df.index < end)]
    cond = conditions(df, inst)
    out = outcomes(df, inst)
    keep = np.asarray(df.index >= start)
    return cond[keep], {d: v[keep] for d, v in out.items()}  # keys 1, -1 (gross R) and "cost"


def declustered(idx, gap=HORIZON):
    taken, last = [], -10**9
    for i in idx:
        if i - last > gap:
            taken.append(i)
            last = i
    return taken


def mine(markets, combos, n_tested):
    rows = []
    for d in (1, -1):
        base = {m: np.nanmean(o[d][np.isfinite(o[d])] == 1.0) for m, (_, o) in markets.items()}
        for combo in combos:
            hits, rs, per_market, expected = 0, [], 0, 0.0
            for m, (cond, out) in markets.items():
                if not all(k in cond for k in combo):
                    continue
                mask = np.logical_and.reduce([cond[k].to_numpy() for k in combo])
                idx = declustered(np.flatnonzero(mask & np.isfinite(out[d])))
                if not idx:
                    continue
                v = out[d][idx]
                per_market += 1
                hits += int((v == 1.0).sum())
                rs.extend((v - np.nan_to_num(out["cost"][idx])).tolist())
                expected += base[m] * len(idx)
            n = len(rs)
            if n < 50 or per_market < 3:
                continue
            rate, p0 = hits / n, expected / n
            p = binom_sf(hits, n, p0)
            rows.append({"side": "long" if d == 1 else "short", "combo": combo, "events": n, "markets": per_market,
                         "hit_rate": rate, "base_rate": p0, "avg_r": float(np.mean(rs)), "p": p})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    survive = df[(df.hit_rate >= 0.6) & (df.hit_rate - df.base_rate >= 0.10) & (df.avg_r >= 0.2)
                 & (df.p < 0.05 / n_tested)]
    return df, survive.sort_values("p")


def all_combos(names):
    out = []
    for k in (1, 2, 3):
        out += list(itertools.combinations(names, k))
    return out


start, end = PERIODS[MODE]
markets = {i["name"]: load_market(i, start, end) for i in insts}
names = sorted(set().union(*[set(c.columns) for c, _ in markets.values()]))

if MODE == "is":
    combos = all_combos(names)
    n_tested = 2 * len(combos)
    print(f"{len(names)} conditions, {len(combos)} combinations x 2 sides = {n_tested} tested")
    table, survivors = mine(markets, combos, n_tested)
    print(f"{len(table)} had enough events; {len(survivors)} survive")
    soft = table[(table.hit_rate >= 0.6) & (table.hit_rate - table.base_rate >= 0.10) & (table.avg_r >= 0.2)]
    print(f"{len(soft)} pass the hit-rate and R filters without the significance test")
    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    print(table.sort_values("p").head(15).to_string(index=False))

    rng = np.random.default_rng(7)
    null_survivors, null_soft = [], []
    for k in range(10):
        shuffled = {}
        for m, (cond, out) in markets.items():
            shift = int(rng.integers(250, 1250)) % len(cond)
            shuffled[m] = (cond, {d: np.roll(v, shift) for d, v in out.items()})  # cost moves with its outcome
        t, s = mine(shuffled, combos, n_tested)
        null_survivors.append(len(s))
        null_soft.append(int(((t.hit_rate >= 0.6) & (t.hit_rate - t.base_rate >= 0.10) & (t.avg_r >= 0.2)).sum())
                         if len(t) else 0)
    print(f"shuffled copies: survivors {null_survivors}, soft passes {null_soft}")
    SURVIVORS.write_text(json.dumps([{"side": r.side, "combo": list(r.combo)} for r in survivors.itertuples()],
                                    indent=2))
else:
    picks = json.loads(SURVIVORS.read_text())
    if not picks:
        print("No survivors to check.")
    for p in picks:
        combo, d = tuple(p["combo"]), 1 if p["side"] == "long" else -1
        t, _ = mine(markets, [combo], 1)
        t = t[t.side == p["side"]] if len(t) else t
        print(p, t.to_dict("records") if len(t) else "fewer than 50 events / 3 markets")
