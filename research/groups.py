"""Constellations per market group, plus cross-market conditions. Run from the repo root:
    python research/groups.py is     # mine 2010-2017 with shuffled baselines
    python research/groups.py oos    # check the saved picks once on 2018-2023

Same outcome, de-clustering and data cut as constellations.py. Fixed before running:

Groups: energy (BRENT, NATGAS), metals (GOLD, SILVER), tech (USA100, NVDA, MSFT, AAPL),
crypto (BTC). Minimum 30 events per constellation, on any number of the group's markets.

Cross-market inputs: Nasdaq (tech), Bitcoin, oil (Brent), gold, silver, 10-year Treasury
note futures (bonds) and the US dollar index. For each, on day D:
  <x>_5d_surge / <x>_5d_slump: 5-day return above +1.5 / below -1.5 standard deviations
  <x>_1d_jump / <x>_1d_drop:   1-day return above +2 / below -2 standard deviations
(standard deviations from the previous 250 days). Bitcoin's day closes at 00:00 UTC, after
the US close, and every entry is at the next open, so nothing comes from the future.

Steps per group:
  A  own conditions, every combination of up to 3
  B  cross-market conditions, every combination of up to 2
  C  the 10 best of A (by p-value) x the 10 best of B: does strong + strong get stronger?
Pass bar per step: hit rate >= 60%, >= 10 points above base, average R >= +0.2, and
p < 0.05 / (tests in that step). The full A-B-C pipeline also runs on 10 shuffled copies.
Lead-lag table: correlation of each input's 5-day return with each group's next 10-day
return, sampled every 10 days so the samples don't overlap.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import constellations as cs  # noqa: E402
from trader import data  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "is"
start, end = cs.PERIODS[MODE]
PICKS = Path("research/groups_picks.json")
GROUPS = {"energy": ["BRENT", "NATGAS"], "metals": ["GOLD", "SILVER"],
          "tech": ["USA100", "NVDA", "MSFT", "AAPL"], "crypto": ["BTC"]}
CROSS = {"nasdaq": "NQ=F", "bitcoin": "BTC-USD", "oil": "BZ=F", "gold": "GC=F", "silver": "SI=F",
         "bonds": "ZN=F", "dollar": "DX-Y.NYB"}
MIN_EVENTS = 30


def cross_series():
    out = {}
    for name, sym in CROSS.items():
        df = data.load({"name": f"X_{name}", "source": "yahoo", "symbol": sym}, "1d")
        out[name] = df["close"][(df.index >= "2008-01-01") & (df.index < end)]
    return out


def cross_conditions(index, closes):
    f = {}
    for name, c in closes.items():
        r5, r1 = np.log(c).diff(5), np.log(c).diff()
        z5 = r5 / r5.rolling(250).std().shift()
        z1 = r1 / r1.rolling(250).std().shift()
        tbl = pd.DataFrame({"s": z5 > 1.5, "l": z5 < -1.5, "j": z1 > 2, "d": z1 < -2}).astype(float)
        tbl.index = tbl.index.floor("D")
        tbl = tbl[~tbl.index.duplicated(keep="last")]
        m = pd.merge_asof(pd.DataFrame({"t": index.floor("D")}), tbl.rename_axis("t").reset_index(), on="t")
        for key, label in (("s", "5d_surge"), ("l", "5d_slump"), ("j", "1d_jump"), ("d", "1d_drop")):
            f[f"{name}_{label}"] = m[key].fillna(0).to_numpy() == 1
    return pd.DataFrame(f, index=index)


def load_all():
    closes = cross_series()
    markets = {}
    for inst in cs.insts:
        df = data.load(inst, "1d")
        df = df[(df.index >= "2008-01-01") & (df.index < end)]
        own = cs.conditions(df, inst)
        out = cs.outcomes(df, inst)
        keep = np.asarray(df.index >= start)
        markets[inst["name"]] = (own[keep], cross_conditions(df.index, closes)[keep],
                                 {k: v[keep] for k, v in out.items()}, df[keep])
    return markets


def passing(t, n_tests):
    if t.empty:
        return t
    return t[(t.hit_rate >= 0.6) & (t.hit_rate - t.base_rate >= 0.10) & (t.avg_r >= 0.2)
             & (t.p < 0.05 / max(n_tests, 1))]


def pipeline(markets, names_own, names_cross, verbose=False):
    """A, B, C for one group. Returns (pass counts, best rows of each step)."""
    both = {m: (pd.concat([own, cross], axis=1), out) for m, (own, cross, out, _) in markets.items()}
    combos_a = cs.all_combos(names_own)
    a, _ = cs.mine(both, combos_a, 1, MIN_EVENTS, 1)
    combos_b = [c for k in (1, 2) for c in itertools.combinations(names_cross, k)]
    b, _ = cs.mine(both, combos_b, 1, MIN_EVENTS, 1)
    res = {"A": (a, 2 * len(combos_a)), "B": (b, 2 * len(combos_b))}
    rows_c = []
    if len(a) and len(b):
        for side in ("long", "short"):
            top_a = a[a.side == side].nsmallest(10, "p")
            top_b = b[b.side == side].nsmallest(10, "p")
            combos_c = [tuple(x.combo) + tuple(y.combo) for x in top_a.itertuples() for y in top_b.itertuples()]
            c, _ = cs.mine(both, combos_c, 1, MIN_EVENTS, 1)
            if len(c):
                rows_c.append(c[c.side == side])
    c = pd.concat(rows_c) if rows_c else pd.DataFrame()
    res["C"] = (c, len(c) if len(c) else 1)
    return {k: len(passing(t, n)) for k, (t, n) in res.items()}, res


def lead_lag(markets, closes):
    rows = []
    for g, names in GROUPS.items():
        for name in names:
            _, _, _, df = markets[name]
            fwd = np.log(df["close"].shift(-10) / df["open"].shift(-1))
            for x, c in closes.items():
                r5 = np.log(c).diff(5)
                r5.index = r5.index.floor("D")
                r5 = r5[~r5.index.duplicated(keep="last")]
                lag = pd.merge_asof(pd.DataFrame({"t": df.index.floor("D")}),
                                    r5.rename("r").rename_axis("t").reset_index(), on="t")["r"].to_numpy()
                ok = np.isfinite(lag) & np.isfinite(fwd.to_numpy())
                idx = np.flatnonzero(ok)[::10]
                rows.append({"group": g, "market": name, "input": x, "n": len(idx),
                             "corr": float(np.corrcoef(lag[idx], fwd.to_numpy()[idx])[0, 1])})
    t = pd.DataFrame(rows)
    t = t[t["market"] != t["input"]]
    g = t.groupby(["group", "input"]).agg(corr=("corr", "mean"), n=("n", "sum")).reset_index()
    g["t_stat"] = g["corr"] * np.sqrt(g["n"] - 2) / np.sqrt(1 - g["corr"] ** 2)
    return g.pivot(index="input", columns="group", values="corr").round(3), g


pd.set_option("display.width", 220, "display.max_colwidth", 100)
markets = load_all()

if MODE == "is":
    closes = cross_series()
    corr, g = lead_lag(markets, {k: v[v.index >= start] for k, v in closes.items()})
    print("Lead-lag: corr(input's last 5 days, group's next 10 days), 2010-2017")
    print(corr.to_string())
    print("strongest:", g.reindex(g.t_stat.abs().sort_values(ascending=False).index).head(5).to_string(index=False))

    picks = {}
    for grp, names in GROUPS.items():
        sub = {m: markets[m] for m in names}
        own_names = sorted(set().union(*[set(sub[m][0].columns) for m in names]))
        cross_names = list(sub[names[0]][1].columns)
        counts, res = pipeline(sub, own_names, cross_names)
        print(f"\n##### {grp}: passes per step {counts}")
        for step in ("A", "B", "C"):
            t, n = res[step]
            if len(t):
                print(f"  step {step} ({n} tests), best 3:")
                print(t.nsmallest(3, "p")[["side", "combo", "events", "hit_rate", "base_rate", "avg_r", "p"]]
                      .to_string(index=False))
        rng = np.random.default_rng(11)
        null = []
        for _ in range(10):
            shuffled = {}
            for m, (own, cross, out, df) in sub.items():
                k = int(rng.integers(250, 1250)) % len(own)
                shuffled[m] = (own, cross, {d: np.roll(v, k) for d, v in out.items()}, df)
            null.append(pipeline(shuffled, own_names, cross_names)[0])
        print(f"  shuffled copies, passes per step: {null}")
        best = {}
        for step in ("A", "B", "C"):
            t, _ = res[step]
            if len(t):
                r = t.nsmallest(1, "p").iloc[0]
                best[step] = {"side": r.side, "combo": list(r.combo)}
        picks[grp] = best
    PICKS.write_text(json.dumps(picks, indent=2))
else:
    picks = json.loads(PICKS.read_text())
    for grp, best in picks.items():
        sub = {m: markets[m] for m in GROUPS[grp]}
        both = {m: (pd.concat([own, cross], axis=1), out) for m, (own, cross, out, _) in sub.items()}
        for step, p in best.items():
            t, _ = cs.mine(both, [tuple(p["combo"])], 1, 1, 1)
            t = t[t.side == p["side"]]
            print(grp, step, p, t[["events", "hit_rate", "base_rate", "avg_r", "p"]].to_dict("records"))
