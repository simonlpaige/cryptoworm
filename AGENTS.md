# CryptoBot — Agent Structure

## Organization Chart

```
Simon (Owner)
  └── CEO (Strategy & Oversight)
        ├── Analyst (Market Research & Signals)
        ├── Quant (Backtesting & Strategy Development)  
        ├── Trader (Execution — paper then live)
        └── Sentinel (Risk Management & Monitoring)
```

## Agent Roles

### CEO — Chief Trading Strategist
- **Model:** Opus (strategic decisions)
- **Heartbeat:** Every 4 hours during market hours
- **Responsibilities:**
  - Set weekly trading thesis based on Analyst research
  - Approve/reject new strategies from Quant
  - Review Trader performance daily
  - Trigger emergency stop via Sentinel if needed
  - Weekly report to Simon

### Analyst — Market Intelligence
- **Model:** Sonnet (high volume research)
- **Heartbeat:** Every 2 hours
- **Responsibilities:**
  - Monitor Reddit (r/bitcoin, r/cryptocurrency, r/CryptoMarkets)
  - Track crypto Twitter/X sentiment
  - Scan financial news (CoinDesk, The Block, Bloomberg crypto)
  - Monitor on-chain data (whale movements, exchange flows)
  - Produce daily signal report for CEO
  - Track Fed/macro events that move crypto

### Quant — Strategy Development
- **Model:** Opus (complex analysis)
- **Heartbeat:** Daily
- **Responsibilities:**
  - Backtest trading strategies against historical data
  - Develop and refine entry/exit rules
  - Build signal scoring models
  - Paper trade new strategies for minimum 14 days before recommending live
  - Document all strategies with clear rules in STRATEGIES.md

### Trader — Execution
- **Model:** Sonnet (fast execution)
- **Heartbeat:** Every 15 minutes during active trades
- **Responsibilities:**
  - Execute trades per approved strategy
  - Manage open positions (stop-losses, take-profits)
  - Log every trade in TRADE_LOG.md
  - Never deviate from approved strategy parameters
  - Phase 1: Paper trading only
  - Phase 2: Live execution with strict limits

### Sentinel — Risk Management
- **Model:** Sonnet (monitoring)  
- **Heartbeat:** Every 30 minutes
- **Responsibilities:**
  - Monitor portfolio drawdown in real-time
  - Enforce daily loss limits (5% max)
  - Enforce position size limits (2% per trade)
  - Alert CEO if any risk threshold breached
  - Auto-pause trading if 15% drawdown from peak
  - Can override Trader and close positions immediately
  - Sentinel has VETO POWER over any trade

## Auto-Learning Rules
Apply closed-loop learning to all CryptoBot operations:

### Detect → Diagnose → Encode → Verify
1. **API failures** (SSL, timeout, rate limit): The Kraken client now has exponential backoff + session rebuild. If a new failure pattern emerges, encode it into `kraken_client.py` retry logic, don't just log and ignore.
2. **Strategy underperformance**: The training engine already reverts after 3 consecutive degradations. Additionally: if a strategy produces 0 trades over 7+ days, investigate whether its conditions are unreachable in current market regime and log as `[auto-learn]`.
3. **Data source failures**: If Binance (451 from US IP), ApeWisdom, or any research source fails repeatedly, disable it in config and add a note. Don't retry broken sources every cycle.
4. **Market regime mismatches**: If the bot is running grid strategies in a trending market (or momentum strategies in a range), the training engine should detect and log the mismatch. Encode regime detection improvements into `researcher.py`.
5. **Cost/compute waste**: Strategies confirmed dead-weight in backtests (EMA/MACD, Bollinger - 1 trade each in 6 months) stay disabled. Don't re-enable without new evidence.

### GOAP-Structured Decision Making
The training engine follows GOAP principles:
- **World State**: current balance, open positions, market regime, strategy performance scores
- **Goals**: maximize risk-adjusted returns while staying within RISK_RULES.md bounds
- **Actions**: tune parameters, enable/disable strategies, adjust risk limits, trigger research
- **Preconditions**: each action requires specific state (e.g., "tune grid" requires grid to have trades)
- **Effects**: each action changes world state predictably (e.g., widening grid range increases trade frequency)

### Model Tiering
- The bot runs as a Python process, not an LLM agent. No model cost.
- If CryptoBot agents are ever rebuilt as LLM agents, use: Gemma4 for monitoring/execution, Sonnet for analysis, Opus only for strategic decisions.

## Security Rules
1. No agent can withdraw funds - only trade
2. No agent can change risk parameters without CEO + Simon approval
3. API keys are read-only for Analyst, trade-only for Trader (no withdrawal)
4. All agents log every action to daily memory file
5. Simon gets weekly P&L report via Telegram
