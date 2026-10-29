# Design notes

## Goal

Signals for high-confidence trades at ActivTrades (CFDs on commodities, US tech
and crypto), 1 hour timeframe, starting capital 10,000 USD, maximum tolerated
drawdown 5,000 USD. Signals only, no automatic execution.

## Why not the MT5 Python package

ActivTrades has no public REST API. Automation normally goes through MetaTrader 5,
but the official `MetaTrader5` Python package only runs on Windows. To stay
cross-platform (Manjaro first), prices come from free public sources instead, and an
MT5 CSV export can be used where exact broker prices matter.

## Why three strategies and a consensus rule

Single indicators fire often and fail often. Requiring agreement between unrelated
ideas (trend direction, fresh momentum, end of a pullback) trades less often but
filters out many weak setups. The strategies are deliberately simple so the backtest
can be trusted and nothing is fitted to noise.

## Why the LLM is not the decision maker

A language model's judgment can't be backtested honestly, because it has seen the
history. Claude builds and maintains the code; the rules decide.

## Honest expectations

ActivTrades states that 84% of retail CFD accounts lose money. On the 1 hour chart,
spreads and overnight swaps are a large share of each trade's expected move, so many
ideas that look good before costs fail after them. The backtest charges those costs
on every trade for exactly that reason.

## Next steps

1. Run `trader backtest` on real data and review the verdict.
2. Correct spreads and swaps from the ActivTrades demo account.
3. Forward test with `trader watch` for 2 to 3 months, logging every signal.
4. Optional: a news filter (economic calendar) to suppress signals before high-impact events.
