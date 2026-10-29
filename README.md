# trader

A signal tool for high-confidence trades at ActivTrades. It watches commodities,
US tech and crypto on the 1 hour chart, and sends a Telegram message when at
least two of three independent strategies agree. **It never places trades.** You
decide and place them yourself.

Pure Python, runs on Linux, macOS and Windows.

## How a signal is made

1. **Three independent strategies** vote on every closed bar (+1 long, -1 short, 0 neutral):
   - `trend`: price and EMA 50 on the same side of EMA 200
   - `breakout`: a 20 bar range breakout in the last 5 bars, confirmed by 24 bar momentum
   - `pullback`: RSI turning back up through 40 (or down through 60)
2. **Consensus**: 3 agree means full size, 2 agree with none opposing means half size, anything else means no signal.
3. **Exits are fixed upfront**: stop at 2 ATR, target at 3 ATR, time stop after 48 bars, optional trailing stop.
4. **Risk**: 1% of capital on a full-size signal, capped by ActivTrades' retail leverage per asset class,
   at most 4 open positions, no new signals after a 3% losing day, and a hard stop at the configured max drawdown.

Everything is in `config.example.yaml`.

## Setup

```bash
git clone https://github.com/KlausUllrich/trader.git ~/trader && cd ~/trader
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yaml config.yaml   # then adjust
cp .env.example .env                 # Telegram bot token and chat id
pytest
```

## Use

```bash
trader fetch       # download and cache 1h history
trader backtest    # in-sample vs out-of-sample report with costs, plus a PASS/FAIL verdict
trader scan        # check the last closed bar once, send any new signals
trader watch       # scan after every bar close, forever
```

`trader backtest` writes `reports/trades.csv` and `reports/summary.json`. The live
signal message includes that instrument's out-of-sample stats from the last backtest.

To run the scan every hour on Manjaro (or any systemd Linux):

```bash
mkdir -p ~/.config/systemd/user && cp deploy/trader-scan.* ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now trader-scan.timer
```

## Data

| Market | Source | History on 1h |
|---|---|---|
| Commodities, Nasdaq, US stocks | Yahoo Finance (futures and stocks) | about 2 years |
| Crypto | Binance via ccxt | 4 years by default |
| Exact ActivTrades prices | CSV export from the MT5 demo | as much as you export |

Yahoo and Binance prices are close to, but not the same as, ActivTrades CFD prices.
Spreads and swaps in the config are estimates. Check the real values in your demo
account (MT5: right-click a symbol, then Specification) and correct them.

## Before trusting it with money

The verdict uses criteria fixed before seeing results: at least 100 out-of-sample
trades, profit factor of 1.2 or more, positive expectancy after costs, and a drawdown
below your limit. If it fails, change the idea, not the parameters until it passes.
Then run `trader watch` for 2 to 3 months on the demo account and compare the
signals with the backtest before risking real money.

Sanity check: on random-walk data the tool correctly finds no edge (profit factor
about 0.8 after costs, verdict FAIL). A strategy that looks great on noise would be a bug.

See [docs/DESIGN.md](docs/DESIGN.md) for the reasoning.
