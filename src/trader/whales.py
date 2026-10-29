"""What the top Hyperliquid traders hold right now, recorded every scan.

Hyperliquid is a crypto exchange where every account's positions are public. This
module picks the best traders from its public leaderboard and records whether they are
long or short BTC, ETH and SOL.

There is no honest history of this (today's leaderboard only shows who won in the past),
so it is recorded from now on and shown in signal messages. It only becomes a vote once
the recording is long enough to backtest it.

Wallet selection, fixed upfront: account value of at least `min_account` USD, profitable
over the last month and all time, the `top` best by last month's profit. Re-picked daily.
Stance: +1 if at least `threshold` of the wallets holding the coin are long, -1 if at least
`threshold` are short, else 0; needs `min_wallets` wallets holding it.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
INFO = "https://api.hyperliquid.xyz/info"
WALLETS_FILE = Path("data/whales_wallets.json")
LOG_FILE = Path("data/whales.csv")
DEFAULTS = {"top": 50, "min_account": 100_000, "threshold": 0.6, "min_wallets": 5, "coins": ["BTC", "ETH", "SOL"]}


def _window(row: dict, name: str) -> float:
    return float(dict(row["windowPerformances"]).get(name, {}).get("pnl", 0) or 0)


def pick_wallets(rows: list[dict], top: int, min_account: float) -> list[str]:
    ok = [r for r in rows if float(r["accountValue"]) >= min_account
          and _window(r, "month") > 0 and _window(r, "allTime") > 0]
    ok.sort(key=lambda r: _window(r, "month"), reverse=True)
    return [r["ethAddress"] for r in ok[:top]]


def wallets(params: dict, max_age_s: int = 86_400) -> list[str]:
    if WALLETS_FILE.is_file():
        saved = json.loads(WALLETS_FILE.read_text())
        if time.time() - saved["time"] < max_age_s:
            return saved["wallets"]
    resp = requests.get(LEADERBOARD, timeout=120)
    resp.raise_for_status()
    picked = pick_wallets(resp.json()["leaderboardRows"], params["top"], params["min_account"])
    WALLETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WALLETS_FILE.write_text(json.dumps({"time": time.time(), "wallets": picked}))
    return picked


def positions(address: str) -> dict[str, float]:
    """Signed position value in USD per coin (negative = short)."""
    resp = requests.post(INFO, json={"type": "clearinghouseState", "user": address}, timeout=30)
    resp.raise_for_status()
    out = {}
    for p in resp.json().get("assetPositions", []):
        pos = p["position"]
        size = float(pos["szi"])
        if size:
            out[pos["coin"]] = (1 if size > 0 else -1) * float(pos["positionValue"])
    return out


def stance(books: list[dict[str, float]], coin: str, threshold: float, min_wallets: int) -> dict:
    held = [b[coin] for b in books if coin in b]
    n_long, n_short = sum(v > 0 for v in held), sum(v < 0 for v in held)
    n = n_long + n_short
    vote = 0
    if n >= min_wallets:
        vote = 1 if n_long / n >= threshold else -1 if n_short / n >= threshold else 0
    return {"coin": coin, "wallets": len(books), "n_long": n_long, "n_short": n_short,
            "long_usd": round(sum(v for v in held if v > 0)), "short_usd": round(-sum(v for v in held if v < 0)),
            "vote": vote}


def record(cfg: dict) -> list[dict]:
    """Snapshot the top wallets' stance per coin and append it to data/whales.csv."""
    params = DEFAULTS | (cfg.get("whales") or {})
    books = []
    for address in wallets(params):
        try:
            books.append(positions(address))
        except requests.RequestException:
            continue
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = [{"time": now} | stance(books, c, params["threshold"], params["min_wallets"]) for c in params["coins"]]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(LOG_FILE, mode="a", header=not LOG_FILE.is_file(), index=False)
    return rows


def describe(row: dict) -> str:
    side = {1: "mostly LONG", -1: "mostly SHORT", 0: "split"}[int(row["vote"])]
    return (f"Top {row['wallets']} Hyperliquid traders on {row['coin']}: {side} "
            f"({row['n_long']} long ${row['long_usd']:,.0f} / {row['n_short']} short ${row['short_usd']:,.0f})")
