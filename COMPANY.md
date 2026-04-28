# CryptoBot — AI Crypto Trading Organization

## Mission
Build an autonomous AI trading system that grows a $500 Bitcoin account through research-driven, risk-managed strategies. Start with paper trading and research. Graduate to live trading only after consistent simulated performance.

## Phase Plan

### Phase 1: Research & Training (NOW — no live money)
- Study market microstructure, sentiment signals, on-chain data
- Backtest strategies against historical data
- Paper trade on Pionex/Binance testnet
- Build signal detection pipeline (Reddit, news, Telegram, on-chain)
- Target: 30 days of paper trading with documented performance

### Phase 2: Small Live ($500 account)
- Deploy best-performing strategy from Phase 1
- Max 2% risk per trade ($10)
- Daily performance logging
- Auto-pause if drawdown exceeds 15%
- Target: 3 months of live data

### Phase 3: Scale (if Phase 2 profitable)
- Increase position sizes
- Add strategies
- Consider additional capital

## Risk Rules (HARD LIMITS)
1. Never risk more than 2% of account per trade
2. Auto-stop-loss on every position
3. No leverage above 2x (preferably 1x spot only to start)
4. Daily max loss: 5% of account
5. If account drops 15% from peak: pause ALL trading, CEO reviews
6. No meme coins, no micro-caps — BTC and ETH only in Phase 1-2
7. All strategies must be backtested before live deployment
8. Simon approves any strategy change that affects risk parameters

## Accounts Needed (Simon's tasks)
See SIMON_TASKS.md
