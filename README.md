# trader

A signal tool for high-confidence trades at ActivTrades. It watches commodities,
US tech and crypto on the 1 hour chart, and sends a Telegram message when at
least two of three independent strategies agree. **It never places trades.** You
decide and place them yourself.

Pure Python, runs on Linux, macOS and Windows.

## How a signal is made

1. **Three independent strategies** vote on every closed daily bar (+1 long, -1 short, 0 neutral).
   The scan still runs hourly and sends the signal on the first hourly bar after the daily close.
   Voting on the 1h bars themselves had no edge, see [Results so far](#results-so-far):
   - `trend`: price and EMA 50 on the same side of EMA 200
   - `breakout`: a 20 bar range breakout in the last 5 bars, confirmed by 24 bar momentum
   - `pullback`: RSI turning back up through 40 (or down through 60)
2. **Consensus**: 3 agree means full size, 2 agree with none opposing means half size, anything else means no signal.
3. **Exits are fixed upfront**: stop at 2 daily ATR, target at 3 daily ATR, time stop after 15 days, optional trailing stop.
4. **Risk**: 1% of capital on a full-size signal, capped by ActivTrades' retail leverage per asset class,
   at most 4 open positions, no new signals after a 3% losing day, and a hard stop at the configured max drawdown.

Everything is in `config.example.yaml`.

## Setup

Linux / macOS:

```bash
git clone https://github.com/KlausUllrich/trader.git ~/trader && cd ~/trader
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yaml config.yaml   # then adjust
cp .env.example .env                 # Telegram bot token and chat id
pytest
```

Windows (cmd). Note the double quotes, cmd does not understand single quotes:

```bat
git clone https://github.com/KlausUllrich/trader.git C:\Projects\trader
cd C:\Projects\trader
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy config.example.yaml config.yaml
copy .env.example .env
pytest
```

In PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

## Use

```bash
trader fetch       # download and cache 1h history
trader backtest    # in-sample vs out-of-sample report with costs, plus a PASS/FAIL verdict
trader scan        # check the last closed bar once, send any new signals
trader watch       # scan after every bar close, forever
trader whales      # record and show what the top Hyperliquid traders hold
```

`trader backtest` writes `reports/trades.csv` and `reports/summary.json`. The live
signal message includes that instrument's out-of-sample stats from the last backtest.

To run the scan every hour on Manjaro (or any systemd Linux):

```bash
mkdir -p ~/.config/systemd/user && cp deploy/trader-scan.* ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now trader-scan.timer
```

On Windows, either leave `trader watch` running in a terminal, or create an hourly
Task Scheduler job that runs `C:\Projects\trader\.venv\Scripts\trader.exe scan` with
`C:\Projects\trader` as the start folder.

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

## Results so far

Same rules and costs, data up to 2026-10-29, judged on the held-out last 30%:

| Votes on | OOS trades | Profit factor | P&L | Max DD | Verdict |
|---|---|---|---|---|---|
| 1h bars | 923 | 0.84 | -$4,652 | $5,061 | FAIL |
| daily bars | 68 | 1.20 | +$326 | $411 | FAIL (too few trades) |

Daily voting was picked from five ideas tested on the in-sample part only, then
checked once on the held-out part. Its result is not yet evidence of an edge:
with about 68 trades, 4 in 20 random-walk runs reach a profit factor of 1.2 too.
It needs a longer forward test before any money goes on it.

On 2010-2017 daily data (gold, silver, oil, gas, Nasdaq) the daily voting lost money,
profit factor 0.81, so the result above was most likely luck. Adding the CFTC large-trader
positioning as a 4th vote (`research/cot_daily.py`) only lifted it to 0.89, so that vote is
off by default. 2018-2023 has not been looked at yet and is kept for the next idea.

Trading jumpy markets (`research/jumpy_daily.py`, 2010-2017, 3x spread when jumpy): a 20-day
breakout on every day had a profit factor of 0.92. Taking it only when ATR(14) > 1.5 x ATR(100)
gave just 16 trades in 8 years (PF 0.94), and fading oil spikes gave 6 trades (PF 0.38).

Chart patterns (`research/chart_daily.py`, 2010-2017): an objective Elliott wave 3 entry
(zigzag swings, Fibonacci retrace and target) PF 0.74 over 84 trades; a simple "leading
candle" (opposite colour, body bigger than the previous range) PF 0.91 over 543; weekly
support/resistance breaks with weekly trend PF 0.95 over 346.

Constellation search (`research/constellations.py`): 34 conditions, every combination of up
to 3, long and short (13,158 tests) on 2010-2017. Hit = 2 ATR in your favour before 2 ATR
against, within 10 days (base rate about 37%). None passed the significance bar. The best,
a 20-day breakout while volatility is low (ATR14 < 0.75 x ATR100), hit 68% over 50 events;
checked once on 2018-2023 it hit 40% over 55 events, against a base of 36%.

Per market group and across markets (`research/groups.py`): split into energy, metals, US
tech and crypto, adding 28 conditions from moves in Nasdaq, Bitcoin, oil, gold, silver,
10-year Treasuries and the US dollar. No group gave a stronger signal, and lead-lag links
are weak (correlations of 0.12 or less). Combining each group's 10 best with the 10 best
cross-market ones made the hit rate worse, not better. All 9 group picks, checked once on
2018-2023, fell back to near the base rate.

Volatility (`research/volatility.py`, `research/volatility_hourly.py`): the size of the next
10 days' moves is predictable (rank correlation 0.77 on untouched 2000-2009, 0.72 on
2018-2023; ATR(14) alone gets 0.76 and 0.70). Of the days flagged "big move expected", 71-81%
really were in the top 20%. But a direction-free breakout with a small stop (orders 0.5 ATR
away, stop 0.5 ATR, target 2 ATR) loses: replayed on 1h bars it won 22% of 758 trades on
those days, PF 0.64 (0.70 on every day).

See [docs/DESIGN.md](docs/DESIGN.md) for the reasoning.
