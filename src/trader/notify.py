"""Signal delivery: console and Telegram."""

from __future__ import annotations

import os

import requests


def format_signal(s: dict) -> str:
    side = "LONG" if s["direction"] == 1 else "SHORT"
    size = "full size" if s["size"] >= 1 else f"{s['size']:.0%} size"
    lines = [
        f"{side} {s['instrument']} ({size}, {s['agree']}/{s.get('voters', 3)} agree: {s['agreeing']})",
        f"Bar closed: {s['time']:%Y-%m-%d %H:%M} UTC",
        f"Entry ~ {s['entry']:.5g}",
        f"Stop    {s['stop']:.5g}",
    ]
    if s.get("target") is not None:
        lines.append(f"Target  {s['target']:.5g}")
    lines.append(f"Units ~ {s['units']:.4g}  (risk ~ ${s['risk']:.0f})")
    lines.append(f"Time stop after {s['max_bars']} bars")
    if s.get("stats"):
        st = s["stats"]
        lines.append(
            f"Backtest (out of sample): {st['trades']} trades, win {st['win_rate']:.0%}, "
            f"PF {st['profit_factor']:.2f}, avg {st['avg_r']:+.2f}R"
        )
    if s.get("whales"):
        lines.append(s["whales"] + " (info only, not a vote yet)")
    lines.append("Signal only. Check spread and news before placing it.")
    return "\n".join(lines)


def send(text: str, cfg: dict) -> None:
    ncfg = cfg.get("notify", {})
    if ncfg.get("console", True):
        print(text, "\n", flush=True)
    if ncfg.get("telegram"):
        token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat:
            return
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text},
            timeout=15,
        )
        resp.raise_for_status()
