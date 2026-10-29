"""Price data providers. All return an OHLCV DataFrame indexed by UTC timestamps.

- yahoo: commodities futures, indices, US stocks (1h history is limited to ~730 days)
- ccxt:  crypto exchanges, years of 1h history
- csv:   any export, e.g. MT5 history from the ActivTrades demo account
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]
CACHE_DIR = Path("data/cache")

_CCXT_TF = {"1h": "1h", "4h": "4h", "1d": "1d"}
_YAHOO_TF = {"1h": "60m", "4h": "60m", "1d": "1d"}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[COLUMNS].astype(float)
    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def fetch_yahoo(symbol: str, timeframe: str = "1h", days: int = 729) -> pd.DataFrame:
    import yfinance as yf

    interval = _YAHOO_TF[timeframe]
    period = f"{min(days, 729)}d" if interval == "60m" else "max"
    raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = _normalise(raw)
    if timeframe == "4h":
        df = resample(df, "4h")
    return df


def fetch_ccxt(exchange: str, symbol: str, timeframe: str = "1h", days: int = 1500) -> pd.DataFrame:
    import ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    tf = _CCXT_TF[timeframe]
    step_ms = ex.parse_timeframe(tf) * 1000
    since = ex.milliseconds() - days * 86_400_000
    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + step_ms
        if len(batch) < 2 or since >= ex.milliseconds():
            break
        time.sleep(ex.rateLimit / 1000)
    if not rows:
        raise RuntimeError(f"{exchange} returned no data for {symbol}")
    df = pd.DataFrame(rows, columns=["ts", *COLUMNS]).set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return _normalise(df)


def read_csv(path: str) -> pd.DataFrame:
    """Reads a generic OHLCV CSV, including MT5 exports (<DATE> <TIME> <OPEN> ...)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        first = next(c for c in df.columns if c in ("datetime", "timestamp", "time", "date"))
        ts = pd.to_datetime(df[first])
    df.index = ts
    if "volume" not in df.columns:
        df["volume"] = df["tickvol"] if "tickvol" in df.columns else 0.0
    return _normalise(df)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["close"])


def load(inst: dict, timeframe: str = "1h", use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """Load bars for one instrument config entry, with a simple CSV cache."""
    source = inst["source"]
    if source == "csv":
        return read_csv(inst["path"])

    cache = CACHE_DIR / f"{inst['name']}_{timeframe}.csv"
    if use_cache and cache.is_file() and not refresh:
        cached = pd.read_csv(cache, index_col=0)
        cached.index = pd.to_datetime(cached.index, utc=True)
        return _normalise(cached)

    if source == "yahoo":
        df = fetch_yahoo(inst["symbol"], timeframe)
    elif source == "ccxt":
        df = fetch_ccxt(inst.get("exchange", "binance"), inst["symbol"], timeframe)
    else:
        raise ValueError(f"Unknown data source {source!r}")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df
