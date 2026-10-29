"""Run the full backtest across instruments and judge it against agreed pass criteria."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import backtest, data

REPORT_DIR = Path("reports")

# Agreed upfront, before looking at results. Applied to the out-of-sample portfolio.
PASS_CRITERIA = {"min_trades": 100, "min_profit_factor": 1.2, "min_expectancy": 0.0}


def run(cfg: dict, only: list[str] | None = None, refresh: bool = False, frames: dict | None = None) -> dict:
    capital = cfg["account"]["capital"]
    oos = cfg["backtest"]["out_of_sample"]
    all_trades: list[backtest.Trade] = []
    oos_trades: list[backtest.Trade] = []
    per_inst: dict = {}
    for inst in cfg["instruments"]:
        if only and inst["name"] not in only:
            continue
        try:
            df = frames[inst["name"]] if frames else data.load(inst, cfg["timeframe"], refresh=refresh)
        except Exception as exc:
            print(f"[{inst['name']}] skipped, data error: {exc}")
            continue
        trades = backtest.run_instrument(df, inst, cfg)
        cut = backtest.split(df, oos)
        ins = [t for t in trades if t.entry_time < cut]
        out = [t for t in trades if t.entry_time >= cut]
        per_inst[inst["name"]] = {
            "bars": len(df), "from": str(df.index[0]), "to": str(df.index[-1]), "oos_from": str(cut),
            "in_sample": backtest.metrics(ins, capital), "out_of_sample": backtest.metrics(out, capital),
        }
        all_trades += trades
        oos_trades += out

    port_all = backtest.portfolio(all_trades, cfg)
    port_oos = backtest.portfolio(oos_trades, cfg)
    summary = {
        "instruments": per_inst,
        "portfolio": {"all": backtest.metrics(port_all, capital), "out_of_sample": backtest.metrics(port_oos, capital)},
        "criteria": PASS_CRITERIA | {"max_drawdown": cfg["account"]["max_drawdown"]},
    }
    summary["verdict"] = verdict(summary["portfolio"]["out_of_sample"], cfg)

    REPORT_DIR.mkdir(exist_ok=True)
    backtest.trades_frame(port_all).to_csv(REPORT_DIR / "trades.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def verdict(m: dict, cfg: dict) -> dict:
    if not m.get("trades"):
        return {"pass": False, "reasons": ["no out-of-sample trades"]}
    reasons = []
    if m["trades"] < PASS_CRITERIA["min_trades"]:
        reasons.append(f"only {m['trades']} trades (need {PASS_CRITERIA['min_trades']})")
    if m["profit_factor"] < PASS_CRITERIA["min_profit_factor"]:
        reasons.append(f"profit factor {m['profit_factor']:.2f} < {PASS_CRITERIA['min_profit_factor']}")
    if m["expectancy"] <= PASS_CRITERIA["min_expectancy"]:
        reasons.append(f"expectancy ${m['expectancy']:.2f} per trade is not positive")
    if m["max_drawdown"] >= cfg["account"]["max_drawdown"]:
        reasons.append(f"max drawdown ${m['max_drawdown']:.0f} hits the ${cfg['account']['max_drawdown']} limit")
    return {"pass": not reasons, "reasons": reasons}


def table(summary: dict) -> str:
    rows = []
    for name, s in summary["instruments"].items():
        for part in ("in_sample", "out_of_sample"):
            m = s[part]
            rows.append({"instrument": name, "part": "IS" if part == "in_sample" else "OOS", **_fmt(m)})
    for part in ("all", "out_of_sample"):
        rows.append({"instrument": "PORTFOLIO", "part": "ALL" if part == "all" else "OOS",
                     **_fmt(summary["portfolio"][part])})
    out = pd.DataFrame(rows).to_string(index=False)
    v = summary["verdict"]
    out += "\n\nVerdict: " + ("PASS" if v["pass"] else "FAIL") + "".join(f"\n  - {r}" for r in v["reasons"])
    return out


def _fmt(m: dict) -> dict:
    if not m.get("trades"):
        return {"trades": 0}
    return {
        "trades": m["trades"], "win%": round(m["win_rate"] * 100, 1), "avgR": round(m["avg_r"], 2),
        "PF": round(m["profit_factor"], 2), "pnl$": round(m["total_pnl"]), "maxDD$": round(m["max_drawdown"]),
        "sharpe": round(m["sharpe"], 2), "costs$": round(m["costs"]),
    }
