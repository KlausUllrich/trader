"""Positioning of large traders from the CFTC Commitments of Traders (COT) report.

The CFTC publishes, every Friday, what each trader group held on the Tuesday before:
- disaggregated report (commodities): managed money = hedge funds and CTAs,
  producers/merchants and swap dealers = commercial hedgers
- financial futures report (index futures): leveraged funds = hedge funds,
  dealers = the banks on the other side

Two votes, chosen before any backtest:
- `follow`: copy the speculators. +1 when their net position grew by more than
  `min_change` of open interest over `weeks`, -1 when it shrank by as much.
- `hedgers`: the classic COT index. +1 when the commercial hedgers' net position
  is in the top `extreme` of its range over `lookback` weeks, -1 when in the bottom.

A report only counts from `available_after_days` after its Tuesday date (default 7,
Tuesday 00:00 UTC a week later), which also covers holiday-delayed releases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://publicreporting.cftc.gov/resource/{dataset}.json"
DATASETS = {"disagg": "72hh-3qpy", "tff": "gpe5-46if"}
CACHE_DIR = Path("data/cache")

# Columns per report: (speculators long, short), (hedgers long, short) lists summed.
FIELDS = {
    "disagg": {
        "spec": (["m_money_positions_long_all"], ["m_money_positions_short_all"]),
        "hedge": (["prod_merc_positions_long", "swap_positions_long_all"],
                  ["prod_merc_positions_short", "swap__positions_short_all"]),
    },
    "tff": {
        "spec": (["lev_money_positions_long"], ["lev_money_positions_short"]),
        "hedge": (["dealer_positions_long_all"], ["dealer_positions_short_all"]),
    },
}


def fetch(code: str, report: str = "disagg") -> pd.DataFrame:
    """Weekly net positions as a fraction of open interest, indexed by report date (UTC)."""
    rows, offset = [], 0
    while True:
        resp = requests.get(API.format(dataset=DATASETS[report]), timeout=60, params={
            "cftc_contract_market_code": code, "$order": "report_date_as_yyyy_mm_dd",
            "$limit": 50_000, "$offset": offset,
        })
        resp.raise_for_status()
        batch = resp.json()
        rows += batch
        if len(batch) < 50_000:
            break
        offset += len(batch)
    if not rows:
        raise RuntimeError(f"CFTC returned no COT data for {code}")
    raw = pd.DataFrame(rows)
    num = lambda cols: sum(pd.to_numeric(raw[c], errors="coerce").fillna(0) for c in cols)
    oi = pd.to_numeric(raw["open_interest_all"], errors="coerce")
    out = pd.DataFrame({
        "spec_net": (num(FIELDS[report]["spec"][0]) - num(FIELDS[report]["spec"][1])) / oi,
        "hedge_net": (num(FIELDS[report]["hedge"][0]) - num(FIELDS[report]["hedge"][1])) / oi,
    })
    out.index = pd.to_datetime(raw["report_date_as_yyyy_mm_dd"], utc=True)
    # A contract can be listed under several names over the years; keep one row per date.
    return out[~out.index.duplicated(keep="last")].sort_index().dropna()


def load(code: str, report: str = "disagg", refresh: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / f"COT_{report}_{code}.csv"
    if cache.is_file() and not refresh:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    df = fetch(code, report)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def weekly_vote(cot: pd.DataFrame, rule: str = "follow", weeks: int = 4, min_change: float = 0.02,
                lookback: int = 156, extreme: float = 0.2) -> pd.Series:
    """One vote per report, indexed by report date."""
    if rule == "follow":
        change = cot["spec_net"].diff(weeks)
        vote = np.where(change > min_change, 1, np.where(change < -min_change, -1, 0))
    elif rule == "hedgers":
        net = cot["hedge_net"]
        lo = net.rolling(lookback, min_periods=lookback // 2).min()
        hi = net.rolling(lookback, min_periods=lookback // 2).max()
        index = (net - lo) / (hi - lo).replace(0, np.nan)
        vote = np.where(index >= 1 - extreme, 1, np.where(index <= extreme, -1, 0))
    else:
        raise ValueError(f"Unknown COT rule {rule!r}")
    return pd.Series(vote, index=cot.index).astype(int)


def align(vote: pd.Series, index: pd.DatetimeIndex, available_after_days: int = 7) -> pd.Series:
    """Map weekly votes onto price bars, using each report only once it was public."""
    v = vote.copy()
    v.index = v.index + pd.Timedelta(days=available_after_days)
    left = pd.DataFrame({"t": index})
    m = pd.merge_asof(left, v.rename("vote").rename_axis("t").reset_index(), on="t")
    return pd.Series(m["vote"].fillna(0).astype(int).to_numpy(), index=index)


def strategy(df: pd.DataFrame, inst: dict, rule: str = "follow", available_after_days: int = 7,
             **params) -> pd.Series:
    """COT vote for one instrument (0 everywhere if it has no `cot` entry)."""
    spec = inst.get("cot") if inst else None
    if not spec:
        return pd.Series(0, index=df.index, dtype=int)
    cot = load(str(spec["code"]), spec.get("report", "disagg"))
    return align(weekly_vote(cot, rule, **params), df.index, available_after_days)
