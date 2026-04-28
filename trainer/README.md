# CryptoBot Self-Improvement Engine

## How It Works

The trainer is a **recursive optimization loop** that runs inside the bot every ~60 minutes (12 ticks). It follows a strict cycle:

```
┌─────────────┐
│   ANALYZE   │  Score each strategy's recent performance
│             │  (win rate, R:R, drawdown, SL/TP hit rates)
└──────┬──────┘
       ▼
┌─────────────┐
│  RESEARCH   │  Fetch live market context:
│             │  - Fear & Greed Index + trend
│             │  - Volatility regime (ATR, BB width)
│             │  - Trend direction
└──────┬──────┘
       ▼
┌─────────────┐
│  DIAGNOSE   │  Map issues to root causes:
│             │  - "stops_too_tight" → widen SL
│             │  - "low_win_rate" → stricter filters
│             │  - "bad_risk_reward" → widen TP
│             │  - "shorts_underperforming" → tighten short RSI
│             │  - "high_volatility" → widen stops, reduce size
└──────┬──────┘
       ▼
┌─────────────┐
│    TUNE     │  Adjust ONE parameter per strategy per cycle
│             │  within research-backed bounds only
│             │  (max 20% of range per step)
└──────┬──────┘
       ▼
┌─────────────┐
│     LOG     │  Record everything:
│             │  - trainer/tuning_log.json (all changes)
│             │  - trainer/reports/cycle_NNNN.json
│             │  - trainer/param_overrides.json (live values)
└──────┬──────┘
       ▼
┌─────────────┐
│   REVERT?   │  If PnL degrades 3 cycles in a row,
│             │  reset ALL overrides to defaults
└──────┬──────┘
       ▼
     WAIT 60m → repeat
```

## Safety Guardrails

1. **Bounded parameters** — Every parameter has a min/max from research. The tuner can never go outside.
2. **One change per strategy per cycle** — Prevents overfitting spiral.
3. **20% max step size** — Gradual changes, not wild swings.
4. **Auto-revert** — 3 consecutive PnL drops → full reset to defaults.
5. **Complete audit trail** — Every change logged with timestamp, reason, before/after.

## Files

| File | Purpose |
|------|---------|
| `engine.py` | Main loop — orchestrates the cycle |
| `analyzer.py` | Scores each strategy's performance |
| `researcher.py` | Fetches market context + research knowledge base |
| `tuner.py` | Maps issues to parameter adjustments |
| `param_loader.py` | Hot-reloads overrides for strategies to use |
| `param_overrides.json` | Current parameter values (auto-managed) |
| `tuning_log.json` | Full history of all changes |
| `training_state.json` | Engine state (cycles, reverts) |
| `reports/` | Per-cycle JSON reports |

## Running Standalone

```bash
# Single cycle (test)
python -m trainer.engine --once

# Continuous (default: 60min intervals)
python -m trainer.engine
```

## Research Sources

Parameters are bounded by data from:
- **TrendRider** backtests (Jan 2025 – Mar 2026)
- **Reddit** r/Daytrading, r/BitcoinMarkets consensus
- **alternative.me** Fear & Greed Index analysis
- **Standard TA** (Bollinger, RSI, EMA/MACD defaults)

## Adding New Research

To add a new research source:
1. Add a fetch function in `researcher.py`
2. Add its recommendations to `build_market_context()`
3. Map recommendations to parameter adjustments in `tuner.py:generate_adjustments()`
4. Add parameter bounds to `RESEARCH_PARAMS` in `researcher.py`
