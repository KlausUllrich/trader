"""Does the COT vote add an edge? Run from the repo root:
    python research/cot_daily.py is        # design period 2010-2017
    python research/cot_daily.py oos D1    # check period 2018-2023, run once for the pick

Long daily history (Yahoo, from 2010) for the five markets with a COT contract.
Everything from 2024-06 on was already used to pick daily voting, so it is cut off here.
Daily bars: signal on the close of day D, entry at the open of day D+1, full costs.

Pre-registered candidates (fixed before running):
  D0  the three technical votes only (the current daily setup)
  D1  + COT "follow": copy the speculators' 4-week change in net position
  D2  + COT "hedgers": the commercial hedgers' net position at a 3-year extreme
"""
import json
import sys

import pandas as pd

from trader import backtest, data
from trader.config import load_config

MODE = sys.argv[1] if len(sys.argv) > 1 else "is"
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
PERIODS = {"is": ("2010-01-01", "2018-01-01"), "oos": ("2018-01-01", "2024-01-01")}
WARMUP_FROM = "2008-01-01"

COT = {
    "GOLD": {"code": "088691", "report": "disagg"},
    "SILVER": {"code": "084691", "report": "disagg"},
    "BRENT": {"code": "067651", "report": "disagg"},  # WTI positioning; ICE Brent is not in the CFTC report
    "NATGAS": {"code": "023651", "report": "disagg"},
    "USA100": {"code": "209742", "report": "tff"},
}
CANDIDATES = {"D0": None, "D1": {"rule": "follow"}, "D2": {"rule": "hedgers"}}

base = load_config("config.example.yaml")
base["timeframe"] = "1d"
base["signal"].pop("timeframe", None)
base["signal"]["size_by_agreement"] = {2: 0.5, 3: 1.0, 4: 1.0}
insts = [dict(i, cot=COT[i["name"]]) for i in base["instruments"] if i["name"] in COT]
start, end = PERIODS[MODE]

frames = {}
for inst in insts:
    df = data.load(inst, "1d")
    frames[inst["name"]] = df[(df.index >= WARMUP_FROM) & (df.index < end)]

for name, cot_params in CANDIDATES.items():
    if ONLY and name not in ONLY:
        continue
    cfg = json.loads(json.dumps(base))
    if cot_params:
        cfg["strategies"]["cot"] = cot_params
    trades = []
    print(f"\n=== {name} {MODE} {start}..{end}")
    for inst in insts:
        tr = [t for t in backtest.run_instrument(frames[inst["name"]], inst, cfg) if t.entry_time >= pd.Timestamp(start, tz="UTC")]
        m = backtest.metrics(tr, 10_000)
        trades += tr
        if m["trades"]:
            print(f"  {inst['name']:7s} n={m['trades']:4d} win={m['win_rate']*100:5.1f} PF={m['profit_factor']:5.2f} "
                  f"pnl={m['total_pnl']:7.0f} DD={m['max_drawdown']:6.0f}")
    pm = backtest.metrics(backtest.portfolio(trades, cfg), 10_000)
    print(f"  PORTFOLIO trades={pm['trades']} PF={pm['profit_factor']:.2f} exp=${pm['expectancy']:.2f} "
          f"pnl=${pm['total_pnl']:.0f} DD=${pm['max_drawdown']:.0f} sharpe={pm['sharpe']:.2f}")
