"""Live scanning: evaluate the last closed bar of each instrument and emit new signals."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import backtest, data, notify, risk, whales

STATE_FILE = Path("data/state.json")
SUMMARY_FILE = Path("reports/summary.json")
_BAR = {"1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)}


def closed_bars(df: pd.DataFrame, timeframe: str, now: datetime | None = None) -> pd.DataFrame:
    """Drop the last bar if it is still forming."""
    now = now or datetime.now(timezone.utc)
    if len(df) and df.index[-1] + _BAR[timeframe] > now:
        return df.iloc[:-1]
    return df


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def evaluate(df: pd.DataFrame, inst: dict, cfg: dict) -> dict | None:
    """Return a signal dict if the last bar in `df` fires, else None."""
    sig = backtest.signals(df, cfg, inst)
    last = sig.iloc[-1]
    if not last["fire"] or not last["atr"] > 0:
        return None
    d = int(last["direction"])
    entry = float(df["close"].iloc[-1])
    stop, target = risk.levels(d, entry, float(last["atr"]), cfg["exits"])
    acct = cfg["account"]
    units, at_risk = risk.position_size(acct["capital"], acct["risk_per_trade"], float(last["size"]),
                                        entry, stop, inst.get("leverage", 1))
    return {
        "instrument": inst["name"], "time": df.index[-1], "direction": d,
        "agree": int(last["agree"]), "agreeing": last["agreeing"], "size": float(last["size"]),
        "voters": sum(c in sig.columns for c in ("trend", "breakout", "pullback", "cot")),
        "entry": entry, "stop": stop, "target": target, "units": units, "risk": at_risk,
        "max_bars": backtest.max_bars(df, cfg["exits"]),
    }


def scan(cfg: dict, only: list[str] | None = None) -> list[dict]:
    state = _load_json(STATE_FILE)
    summary = _load_json(SUMMARY_FILE).get("instruments", {})
    found = []
    crowd = {}
    if cfg.get("whales"):
        try:  # recorded every scan so the stance can be backtested later
            crowd = {r["coin"]: r for r in whales.record(cfg)}
        except Exception as exc:
            print(f"[whales] error: {exc}")
    for inst in cfg["instruments"]:
        if only and inst["name"] not in only:
            continue
        try:
            df = closed_bars(data.load(inst, cfg["timeframe"], use_cache=False), cfg["timeframe"])
        except Exception as exc:  # one bad feed must not stop the others
            print(f"[{inst['name']}] data error: {exc}")
            continue
        s = evaluate(df, inst, cfg)
        if not s:
            continue
        key = s["time"].isoformat()
        if state.get(inst["name"]) == key:
            continue
        s["stats"] = summary.get(inst["name"], {}).get("out_of_sample")
        if inst.get("whales") in crowd:
            s["whales"] = whales.describe(crowd[inst["whales"]])
        notify.send(notify.format_signal(s), cfg)
        state[inst["name"]] = key
        found.append(s)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    return found


def watch(cfg: dict, delay_s: int = 90) -> None:
    """Scan once per bar, shortly after each bar closes."""
    step = _BAR[cfg["timeframe"]]
    while True:
        scan(cfg)
        now = datetime.now(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        next_close = epoch + ((now - epoch) // step + 1) * step
        time.sleep(max(5.0, (next_close - now).total_seconds() + delay_s))
