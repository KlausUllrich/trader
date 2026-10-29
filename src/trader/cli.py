"""Command line: `trader backtest`, `trader scan`, `trader watch`, `trader fetch`."""

from __future__ import annotations

import argparse

from . import data, live, report, whales
from .config import load_config, load_dotenv


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trader", description="Consensus trade signals (signals only, no execution)")
    p.add_argument("-c", "--config", help="config file (default: config.yaml, else config.example.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="backtest all instruments with costs, in- and out-of-sample")
    b.add_argument("-i", "--instrument", action="append", help="limit to these instruments")
    b.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")

    f = sub.add_parser("fetch", help="download and cache price history")
    f.add_argument("-i", "--instrument", action="append")

    s = sub.add_parser("scan", help="check the last closed bar once and send new signals")
    s.add_argument("-i", "--instrument", action="append")

    sub.add_parser("watch", help="scan after every bar close, forever")
    sub.add_parser("whales", help="record and show what the top Hyperliquid traders hold")

    args = p.parse_args(argv)
    load_dotenv()
    cfg = load_config(args.config)

    if args.cmd == "backtest":
        summary = report.run(cfg, args.instrument, refresh=args.refresh)
        print(report.table(summary))
        print("\nTrades: reports/trades.csv  Summary: reports/summary.json")
    elif args.cmd == "fetch":
        for inst in cfg["instruments"]:
            if args.instrument and inst["name"] not in args.instrument:
                continue
            try:
                df = data.load(inst, cfg["timeframe"], refresh=True)
                print(f"{inst['name']}: {len(df)} bars {df.index[0]} .. {df.index[-1]}")
            except Exception as exc:
                print(f"{inst['name']}: error {exc}")
    elif args.cmd == "scan":
        found = live.scan(cfg, args.instrument)
        if not found:
            print("No new signals.")
    elif args.cmd == "watch":
        live.watch(cfg)
    elif args.cmd == "whales":
        for row in whales.record(cfg):
            print(whales.describe(row))


if __name__ == "__main__":
    main()
