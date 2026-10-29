"""Bar-by-bar backtest with spreads, commissions and overnight swaps.

Rules (all deliberately conservative):
- Signals are computed on a closed bar; entry is at the NEXT bar's open.
- Spread is paid half on entry and half on exit.
- If stop and target are both inside one bar, the stop is assumed to hit first.
- Swaps are charged on notional for the time held, for longs and shorts alike.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import consensus, risk, strategies
from .indicators import atr


@dataclass
class Trade:
    instrument: str
    direction: int
    agree: int
    agreeing: str
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float | None
    exit_time: pd.Timestamp
    exit: float
    reason: str
    units: float
    risk: float
    pnl: float
    costs: float

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.risk if self.risk else 0.0


def signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    v = strategies.votes(df, cfg["strategies"])
    sig = consensus.combine(v, cfg["signal"]["min_agree"], cfg["signal"]["size_by_agreement"])
    sig["atr"] = atr(df, cfg["exits"]["atr_period"])
    return sig.join(v)


def run_instrument(df: pd.DataFrame, inst: dict, cfg: dict) -> list[Trade]:
    sig = signals(df, cfg)
    exits, acct = cfg["exits"], cfg["account"]
    half_spread = inst.get("spread_bps", 0) / 2 / 10_000
    commission = inst.get("commission", 0.0)
    swap_annual = inst.get("swap_annual", 0.0)
    leverage = inst.get("leverage", 1)

    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    times = df.index
    fire, direction = sig["fire"].to_numpy(), sig["direction"].to_numpy()
    size, agree, agreeing, atr_v = (sig[k].to_numpy() for k in ("size", "agree", "agreeing", "atr"))

    trades: list[Trade] = []
    i, n = 0, len(df)
    while i < n - 1:
        if not (fire[i] and np.isfinite(atr_v[i]) and atr_v[i] > 0):
            i += 1
            continue
        d = int(direction[i])
        e = i + 1
        entry = o[e] * (1 + d * half_spread)
        initial_stop, target = risk.levels(d, entry, atr_v[i], exits)
        stop = initial_stop
        units, at_risk = risk.position_size(acct["capital"], acct["risk_per_trade"], size[i], entry, stop, leverage)
        if units <= 0:
            i += 1
            continue

        trail_atr = exits.get("trail_atr", 0) or 0
        best = entry
        exit_idx, exit_raw, reason = None, None, None
        for j in range(e, min(n, e + exits["max_bars"])):
            hit_stop = l[j] <= stop if d == 1 else h[j] >= stop
            hit_target = target is not None and (h[j] >= target if d == 1 else l[j] <= target)
            if hit_stop:
                # A gap through the stop fills at the open, not the stop.
                exit_raw = min(o[j], stop) if d == 1 else max(o[j], stop)
                exit_idx, reason = j, "stop"
                break
            if hit_target:
                exit_idx, exit_raw, reason = j, target, "target"
                break
            # Opposite consensus closes the trade at the next open.
            if fire[j] and direction[j] == -d and j + 1 < n:
                exit_idx, exit_raw, reason = j + 1, o[j + 1], "reverse"
                break
            if trail_atr > 0 and np.isfinite(atr_v[j]):
                best = max(best, h[j]) if d == 1 else min(best, l[j])
                trail = best - d * trail_atr * atr_v[j]
                stop = max(stop, trail) if d == 1 else min(stop, trail)
        if exit_idx is None:
            exit_idx = min(n - 1, e + exits["max_bars"] - 1)
            exit_raw, reason = c[exit_idx], "time"

        exit_price = exit_raw * (1 - d * half_spread)
        held_days = (times[exit_idx] - times[e]).total_seconds() / 86_400
        notional = units * entry
        costs = notional * commission * 2 + notional * swap_annual * held_days / 365
        spread_cost = units * (abs(entry - o[e]) + abs(exit_price - exit_raw))
        pnl = d * (exit_price - entry) * units - costs
        trades.append(Trade(inst["name"], d, int(agree[i]), agreeing[i], times[e], entry, initial_stop, target,
                            times[exit_idx], exit_price, reason, units, at_risk, pnl, costs + spread_cost))
        # A new signal may fire on the exit bar itself (or on the reversal bar).
        i = exit_idx - 1 if reason == "reverse" else exit_idx
    return trades


def portfolio(trades: list[Trade], cfg: dict) -> list[Trade]:
    """Apply account-wide limits: max open positions, daily loss limit, drawdown hard stop."""
    acct = cfg["account"]
    accepted: list[Trade] = []
    booked: set[int] = set()
    open_until: list[pd.Timestamp] = []
    realised = 0.0
    peak = 0.0
    day_pnl: dict = {}
    for t in sorted(trades, key=lambda t: t.entry_time):
        # Realise everything that closed before this entry.
        for k, a in enumerate(accepted):
            if k not in booked and a.exit_time <= t.entry_time:
                booked.add(k)
                realised += a.pnl
                peak = max(peak, realised)
                day_pnl[a.exit_time.date()] = day_pnl.get(a.exit_time.date(), 0.0) + a.pnl
        if peak - realised >= acct["max_drawdown"]:
            break
        if day_pnl.get(t.entry_time.date(), 0.0) <= -acct["daily_loss_limit"] * acct["capital"]:
            continue
        open_until = [x for x in open_until if x > t.entry_time]
        if len(open_until) >= acct["max_open_positions"]:
            continue
        accepted.append(t)
        open_until.append(t.exit_time)
    return accepted


def metrics(trades: list[Trade], capital: float) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = np.array([t.pnl for t in trades])
    r = np.array([t.r_multiple for t in trades])
    ordered = sorted(trades, key=lambda t: t.exit_time)
    equity = np.cumsum([t.pnl for t in ordered])
    drawdown = np.max(np.maximum.accumulate(np.concatenate([[0], equity]))[1:] - equity) if len(equity) else 0.0
    daily = pd.Series([t.pnl for t in ordered], index=[t.exit_time.normalize() for t in ordered]).groupby(level=0).sum()
    if len(daily) > 1:
        days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(days, fill_value=0.0) / capital
        sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0
    else:
        sharpe = 0.0
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "trades": len(trades),
        "win_rate": float((pnl > 0).mean()),
        "avg_r": float(r.mean()),
        "expectancy": float(pnl.mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "total_pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / capital * 100),
        "max_drawdown": float(drawdown),
        "sharpe": sharpe,
        "costs": float(sum(t.costs for t in trades)),
    }


def split(df: pd.DataFrame, out_of_sample: float) -> pd.Timestamp:
    """Timestamp where the held-out (out-of-sample) part of the series starts."""
    return df.index[int(len(df) * (1 - out_of_sample))]


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    rows = [asdict(t) | {"r": t.r_multiple} for t in trades]
    return pd.DataFrame(rows)
