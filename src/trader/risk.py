"""Position sizing and stop/target levels."""

from __future__ import annotations


def levels(direction: int, entry: float, atr_value: float, exits: dict) -> tuple[float, float | None]:
    stop = entry - direction * exits["stop_atr"] * atr_value
    target_atr = exits.get("target_atr", 0) or 0
    target = entry + direction * target_atr * atr_value if target_atr > 0 else None
    return stop, target


def position_size(capital: float, risk_per_trade: float, size: float, entry: float, stop: float,
                  leverage: float) -> tuple[float, float]:
    """Units to trade so that hitting the stop loses `capital * risk_per_trade * size`.

    Capped so the notional never needs more margin than the account has at `leverage`.
    Returns (units, money at risk).
    """
    risk_amount = capital * risk_per_trade * size
    distance = abs(entry - stop)
    if distance <= 0 or entry <= 0:
        return 0.0, 0.0
    units = risk_amount / distance
    max_units = capital * leverage / entry
    units = min(units, max_units)
    return units, units * distance
