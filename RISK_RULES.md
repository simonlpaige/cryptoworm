# CryptoBot - Risk Management Rules

## PORTFOLIO STRUCTURE (80/20)
- **HODL (80%):** $400 in BTC, buy and hold. Never traded by the bot.
- **Grid Pool (20%):** $100 for active grid trading. All risk rules below apply to this pool only.
- **Rationale:** 9-year backtest showed HODL returned 8,022% vs Grid's +3%. Grid's edge is defensive (saves money in crashes) but can't compete in bulls. This split captures both.

## ABSOLUTE RULES (never override)

### Position Limits
- **Paper Phase (current):** Max 20% of Grid Pool per trade ($20 on $100), 3 concurrent positions
- **Live Phase (future):** Revert to 10% per trade, 3 concurrent
- No leverage above 2x (1x preferred in Phase 1-2)

### Loss Limits
- Stop-loss required on EVERY position - no exceptions
- **Paper Phase:** Daily 10% / Weekly 15% / Drawdown pause 20% (of Grid Pool)
- **Live Phase:** Daily 5% / Weekly 10% / Drawdown pause 15%
- **Drawdown breaker at 20%** - backtested as the single best improvement (turned -$114 into +$15 over 9 years)

### Asset Restrictions
- Phase 1-2: BTC and ETH ONLY
- No meme coins, no alt-coins under $1B market cap
- No futures/options until Phase 3 (if ever)
- No margin trading above 2x

### Withdrawal/Security
- API keys: trade-only permissions, NO withdrawal
- No agent can request fund transfers
- Simon is the only person who can deposit or withdraw
- All API keys rotated monthly

## ESCALATION PROTOCOL
1. **Yellow alert** (5% daily loss): Sentinel notifies CEO, reduces position sizes by 50%
2. **Red alert** (10% weekly loss): Sentinel pauses new trades, CEO reviews all open positions
3. **Emergency stop** (15% drawdown): Sentinel closes ALL positions, alerts Simon via Telegram
4. **Manual override**: Simon can pause/resume at any time via Telegram command

## PERFORMANCE TRACKING
- Log every trade in TRADE_LOG.md
- Daily P&L summary in memory/YYYY-MM-DD.md
- Weekly performance report to Simon (win rate, total P&L, max drawdown, Sharpe ratio)
- Monthly strategy review: keep, modify, or kill each active strategy
