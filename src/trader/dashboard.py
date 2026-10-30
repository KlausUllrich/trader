"""What the web app shows: market overview, chart data and the pre-trade check.

Everything here is plain data (dicts of numbers and strings), so app.py only has to turn it
into JSON. Prices come through a `Market` object, which caches bars in memory; `DemoMarket`
makes synthetic bars so the app can be tried without any network access.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, data, levels, risk, volatility, whales
from .config import instrument as find_instrument
from .indicators import ema
from .live import SUMMARY_FILE, closed_bars

VOTERS = ("trend", "breakout", "pullback", "cot")
NO_EDGE = ("No direction rule has beaten costs in testing so far. Treat the direction as your call; "
           "the reliable inputs here are the size of the expected move and the position size.")


def group(inst: dict) -> str:
    if inst.get("group"):
        return inst["group"]
    if inst.get("source") == "ccxt" or inst.get("whales"):
        return "Crypto"
    return "Markets"


class Market:
    """Loads bars through data.load and keeps them in memory until refresh()."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._bars: dict[tuple[str, str], pd.DataFrame] = {}
        self._refresh = False

    def bars(self, inst: dict, timeframe: str) -> pd.DataFrame:
        key = (inst["name"], timeframe)
        if key not in self._bars:
            self._bars[key] = self._load(inst, timeframe)
        return self._bars[key]

    def _load(self, inst: dict, timeframe: str) -> pd.DataFrame:
        if inst["source"] == "csv" and timeframe == "1d":
            return data.resample(data.read_csv(inst["path"]), "1D")
        return data.load(inst, timeframe, refresh=self._refresh)

    def whales(self) -> dict[str, dict]:
        """Latest recorded stance per coin from data/whales.csv (written by `trader scan`)."""
        if not whales.LOG_FILE.is_file():
            return {}
        log = pd.read_csv(whales.LOG_FILE)
        return {r["coin"]: r for r in log.groupby("coin").tail(1).to_dict("records")}

    def refresh(self) -> None:
        self._bars.clear()
        self._refresh = True
        if self.cfg.get("whales"):
            try:
                whales.record(self.cfg)
            except Exception as exc:  # whale data is optional
                print(f"[whales] error: {exc}")


class DemoMarket(Market):
    """Synthetic random-walk bars with changing volatility, one fixed series per instrument."""

    def _load(self, inst: dict, timeframe: str) -> pd.DataFrame:
        seed = sum(map(ord, inst["name"]))
        rng = np.random.default_rng(seed)
        n = 24 * 900
        regime = np.repeat(rng.choice([0.5, 1.0, 2.2], size=n // 240 + 1, p=[0.3, 0.5, 0.2]), 240)[:n]
        vol = 0.0035 * pd.Series(regime).rolling(48, min_periods=1).mean().to_numpy()
        trend = np.repeat(rng.normal(0, 0.0004, size=n // 500 + 1), 500)[:n]
        close = 100 * (1 + seed % 7) * np.exp(np.cumsum(trend + vol * rng.standard_normal(n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(rng.standard_normal(n)) * vol * close
        end = pd.Timestamp.now(tz="UTC").floor("h")
        idx = pd.date_range(end=end, periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick,
                           "low": np.minimum(open_, close) - wick, "close": close, "volume": 1.0}, index=idx)
        return data.resample(df, "1D") if timeframe == "1d" else df

    def whales(self) -> dict[str, dict]:
        out = {}
        for i, coin in enumerate(["BTC", "ETH", "SOL"]):
            n_long, n_short = [(31, 9), (14, 18), (12, 20)][i]
            vote = 1 if n_long / (n_long + n_short) >= 0.6 else -1 if n_short / (n_long + n_short) >= 0.6 else 0
            out[coin] = {"time": pd.Timestamp.now(tz="UTC").floor("h").isoformat(), "coin": coin, "wallets": 50,
                         "n_long": n_long, "n_short": n_short, "long_usd": n_long * 2.1e6,
                         "short_usd": n_short * 1.7e6, "vote": vote}
        return out

    def refresh(self) -> None:
        self._bars.clear()


# ---------- building blocks ----------

def _signals(market: Market, inst: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Closed 1h bars and the strategies' signals on them, as the live scan sees them."""
    cfg = market.cfg
    df = closed_bars(market.bars(inst, cfg["timeframe"]), cfg["timeframe"])
    return df, backtest.signals(df, cfg, inst)


def _votes(sig: pd.DataFrame) -> dict[str, int]:
    last = sig.iloc[-1]
    return {v: int(last[v]) for v in VOTERS if v in sig.columns}


def _whale(market: Market, inst: dict, stance: dict | None = None) -> dict | None:
    coin = inst.get("whales")
    if not coin:
        return None
    row = (stance if stance is not None else market.whales()).get(coin)
    if not row:
        return {"coin": coin, "vote": None, "text": "No whale data recorded yet (run `trader scan` or Refresh)."}
    return {"coin": coin, "vote": int(row["vote"]), "n_long": int(row["n_long"]), "n_short": int(row["n_short"]),
            "long_usd": float(row["long_usd"]), "short_usd": float(row["short_usd"]),
            "time": str(row["time"]), "text": whales.describe(row)}


def _last_fire(df: pd.DataFrame, sig: pd.DataFrame) -> dict | None:
    fired = sig.index[sig["fire"].to_numpy()]
    if not len(fired):
        return None
    t = fired[-1]
    row = sig.loc[t]
    return {"time": t, "direction": int(row["direction"]), "agree": int(row["agree"]),
            "agreeing": row["agreeing"], "price": float(df.at[t, "close"]),
            "bars_ago": int(len(df) - 1 - df.index.get_loc(t))}


def _backtest_stats(name: str) -> dict | None:
    if not SUMMARY_FILE.is_file():
        return None
    return json.loads(SUMMARY_FILE.read_text()).get("instruments", {}).get(name, {}).get("out_of_sample")


# ---------- screens ----------

def overview(market: Market) -> dict:
    stance = market.whales()
    rows = []
    for inst in market.cfg["instruments"]:
        try:
            df, sig = _signals(market, inst)
            daily = market.bars(inst, "1d")
        except Exception as exc:  # one bad feed must not break the page
            rows.append({"name": inst["name"], "group": group(inst), "error": str(exc)})
            continue
        close = float(df["close"].iloc[-1])
        prev_day = df["close"][df.index <= df.index[-1] - pd.Timedelta(days=1)]
        last = sig.iloc[-1]
        direction = int(last["direction"])
        row = {
            "name": inst["name"], "group": group(inst), "price": close, "time": df.index[-1],
            "change_1d": close / float(prev_day.iloc[-1]) - 1 if len(prev_day) else None,
            "vol": volatility.latest(daily), "votes": _votes(sig),
            "consensus": {"direction": direction, "agree": int(last["agree"])},
            "last_signal": _last_fire(df, sig), "whales": _whale(market, inst, stance),
        }
        if direction:
            c = check(market, inst["name"], direction, _cached=(df, sig, daily, stance))
            row["check"] = {"direction": direction, "score": c["score"], "of": c["of"], "verdict": c["verdict"]}
        rows.append(row)
    return {"rows": rows, "demo": isinstance(market, DemoMarket), "note": NO_EDGE}


def chart(market: Market, name: str, timeframe: str = "1d", limit: int | None = None) -> dict:
    cfg = market.cfg
    inst = find_instrument(cfg, name)
    df, sig = _signals(market, inst)
    daily = market.bars(inst, "1d")
    bars = daily if timeframe == "1d" else df
    limit = limit or (750 if timeframe == "1d" else 1500)
    bars = bars.iloc[-limit:]
    start = bars.index[0]

    # Markers sit on the candle that contains the signal bar.
    fired = sig[sig["fire"] & (sig.index >= start)]
    at = bars.index[bars.index.searchsorted(fired.index, side="right") - 1]
    markers = [{"time": t, "direction": int(r["direction"]), "agree": int(r["agree"]), "agreeing": r["agreeing"]}
               for t, (_, r) in zip(at, fired.iterrows())]

    full = daily if timeframe == "1d" else df
    lines = {f"ema{p}": ema(full["close"], p).loc[start:] for p in (50, 200)}
    fc = volatility.forecast(daily)
    fc = fc[fc.index >= start.floor("D")]

    active = None
    last = _last_fire(df, sig)
    if last and last["bars_ago"] <= backtest.max_bars(df, cfg["exits"]):
        a = float(sig.loc[last["time"], "atr"])
        stop, target = risk.levels(last["direction"], last["price"], a, cfg["exits"])
        active = last | {"stop": stop, "target": target}

    return {
        "name": inst["name"], "group": group(inst), "timeframe": timeframe,
        "candles": [{"time": t, "open": r.open, "high": r.high, "low": r.low, "close": r.close}
                    for t, r in zip(bars.index, bars.itertuples())],
        "markers": markers,
        "lines": {k: [{"time": t, "value": v} for t, v in s.items() if np.isfinite(v)] for k, s in lines.items()},
        "vol": [{"time": t, "forecast": r.forecast, "actual": r.actual} for t, r in zip(fc.index, fc.itertuples())
                if np.isfinite(r.forecast)],
        "vol_now": volatility.latest(daily), "levels": levels.find(daily), "votes": _votes(sig),
        "consensus": {"direction": int(sig["direction"].iloc[-1]), "agree": int(sig["agree"].iloc[-1])},
        "active": active, "whales": _whale(market, inst), "demo": isinstance(market, DemoMarket),
    }


def check(market: Market, name: str, direction: int, _cached: tuple | None = None) -> dict:
    """Pre-trade check for going long (+1) or short (-1) at the last close."""
    cfg = market.cfg
    inst = find_instrument(cfg, name)
    if _cached:
        df, sig, daily, stance = _cached
    else:
        df, sig = _signals(market, inst)
        daily, stance = market.bars(inst, "1d"), market.whales()
    d = 1 if direction > 0 else -1
    side = "long" if d == 1 else "short"
    entry = float(df["close"].iloc[-1])
    vol = volatility.latest(daily)
    exits, acct = cfg["exits"], cfg["account"]

    # Stop and target sized by the forecast daily range instead of yesterday's ATR.
    move = vol["move"] or float(sig["atr"].iloc[-1])
    stop, target = risk.levels(d, entry, move, exits)
    units, at_risk = risk.position_size(acct["capital"], acct["risk_per_trade"], 1.0, entry, stop,
                                        inst.get("leverage", 1))
    notional = units * entry
    plan = {"entry": entry, "stop": stop, "target": target, "units": units, "risk": at_risk,
            "notional": notional, "margin": notional / inst.get("leverage", 1),
            "losses_to_limit": math.floor(acct["max_drawdown"] / at_risk) if at_risk else None,
            "stop_basis": "volatility forecast" if vol["move"] else "ATR"}

    checks = []
    votes = _votes(sig)
    pro = [k for k, v in votes.items() if v == d]
    con = [k for k, v in votes.items() if v == -d]
    status = "good" if len(pro) >= 2 and not con else "bad" if con and not pro else "warn"
    checks.append({"name": "Strategies", "status": status,
                   "detail": f"{len(pro)} of {len(votes)} vote {side}" + (f" ({', '.join(pro)})" if pro else "")
                   + (f"; against: {', '.join(con)}" if con else "")})

    reg = vol["regime"]
    if reg == "unknown":
        checks.append({"name": "Expected move", "status": "warn", "detail": "Not enough daily history to forecast"})
    else:
        detail = (f"About {vol['forecast_pct']:.2f}% a day over the next 10 days, "
                  f"higher than {vol['rank']:.0%} of past forecasts")
        status = {"jumpy": "good", "normal": "info", "calm": "warn"}[reg]
        hint = {"jumpy": "; big moves expected", "normal": "", "calm": "; small moves, costs weigh more"}[reg]
        checks.append({"name": "Expected move", "status": status, "detail": detail + hint})

    lv = levels.find(daily)
    ahead = [x for x in lv if (x["price"] - entry) * d > 0]
    if target is None:
        checks.append({"name": "Room to move", "status": "info", "detail": "No fixed target set"})
    elif ahead and (ahead[0]["price"] - entry) * d < (target - entry) * d:
        x = ahead[0]
        checks.append({"name": "Room to move", "status": "warn",
                       "detail": f"{x['kind'].title()} at {x['price']:.4g} ({x['touches']} touches) "
                                 f"before the target at {target:.4g}"})
    else:
        checks.append({"name": "Room to move", "status": "good", "detail": f"No known level before the target at {target:.4g}"})

    w = _whale(market, inst, stance)
    if w and w["vote"] is not None:
        status = "good" if w["vote"] == d else "bad" if w["vote"] == -d else "info"
        checks.append({"name": "Whales", "status": status, "detail": w["text"]})

    cost = notional * (inst.get("spread_bps", 0) / 10_000 + 2 * inst.get("commission", 0.0))
    share = cost / at_risk if at_risk else 0.0
    checks.append({"name": "Costs", "status": "good" if share < 0.05 else "warn" if share < 0.15 else "bad",
                   "detail": f"Spread and commission about ${cost:,.2f}, {share:.0%} of the risk"})

    stats = _backtest_stats(inst["name"])
    if not stats or not stats.get("trades"):
        checks.append({"name": "Track record", "status": "warn", "detail": "No backtest yet; run `trader backtest`"})
    else:
        pf = stats.get("profit_factor", 0)
        checks.append({"name": "Track record", "status": "good" if pf >= 1.2 else "warn" if pf >= 1 else "bad",
                       "detail": f"Out of sample: {stats['trades']} trades, profit factor {pf:.2f}, "
                                 f"win rate {stats.get('win_rate', 0):.0%}"})

    rated = [c for c in checks if c["status"] != "info"]
    score = sum(c["status"] == "good" for c in rated)
    bad = any(c["status"] == "bad" for c in rated)
    verdict = "against" if bad else "aligned" if score >= len(rated) - 1 else "mixed"
    return {"name": inst["name"], "direction": d, "plan": plan, "checks": checks, "score": score,
            "of": len(rated), "verdict": verdict, "vol": vol, "note": NO_EDGE}


def jsonable(obj):
    """Timestamps to unix seconds, NaN and inf to None, numpy scalars to Python."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return int(obj.timestamp())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
