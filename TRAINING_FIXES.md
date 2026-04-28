# Training Pipeline Fixes — 2026-04-11

## Summary

Six changes to fix trade closing, wire up the training loop, disable dead-weight strategies, and improve backtester data quality.

---

## 1. Trade Closing Fix (`utils/risk_manager.py`)

**Root cause**: Two bot processes (PIDs 5704, 8772) run simultaneously with separate in-memory states. When one closes a position and saves, the other overwrites the file with stale state, resurrecting closed positions. The bot log proves 3 trades DID close historically (2 TPs, 1 SL) but their records were lost from `bot_state.json`.

**Fixes**:
- `reload_state()` — merges disk + memory state before every SL/TP check. Positions closed in either copy stay closed. Balance takes the max (never regresses from closed-trade gains).
- Atomic `save_state()` — writes to `.tmp` then `os.replace()` to prevent partial writes.
- **Position max-age timeout** — force-closes positions older than 168h (7 days) to prevent stale positions accumulating forever.
- **Detailed per-tick logging** — every open position now logs its distance to TP/SL at DEBUG level for diagnosis.

## 2. Strategies Now Use Trainer Param Overrides

**Root cause**: The trainer wrote to `param_overrides.json` but all strategies used hardcoded values from `config.py`. 102 training cycles of tuning had zero effect.

**Fixes** (all strategies now call `trainer.param_loader` accessors):
- `ema_macd.py` — ADX threshold, RSI long/short ranges, SL%, TP%
- `bollinger.py` — BB period, std devs, RSI oversold/overbought, ADX max
- `sentiment.py` — fear/greed thresholds, SL%, TP%
- `rsi_divergence.py` — RSI long/short thresholds, SL%, TP%

## 3. Dead Strategies Disabled (`config.py`, `bot.py`)

Backtest evidence: EMA/MACD, Bollinger, and tariff_whiplash each produced 1 trade in 6 months. They waste compute and confuse the trainer.

**New config flags** (all respected in `bot.py`):
- `ENABLE_EMA_MACD = False`
- `ENABLE_BOLLINGER = False`
- `ENABLE_TARIFF_WHIPLASH = False`
- `ENABLE_RSI_DIVERGENCE = True` (kept for signal diversity)
- `ENABLE_CONGRESS_FRONTRUN = True` (50% WR, 2.42 PF — best novel strategy)

Bot now logs which strategies are active at startup.

## 4. Training Loop: Real Backtests Replace Synthetic Data (`trainer/engine.py`)

**Root cause**: With zero closed trades, the trainer fell back to `simulate_backtest_trades()` which generated bar-by-bar synthetic "trades" — pure noise. The trainer tuned against noise, degraded 3 times, and the meta-learner reset to defaults twice.

**Fix**: New `inject_backtest_results()` runs the real backtester against historical data (CryptoCompare hourly candles). Results are cached for 6 hours. The trainer now evaluates against actual backtest win_rate, Sharpe ratio, and profit factor instead of synthetic noise.

## 5. Discovery Validation Pipeline (`trainer/discovery.py`, `trainer/engine.py`)

**Root cause**: `propose_strategy()` created proposals with status "proposed" but nothing ever promoted them. 11 proposals sat idle (some with 83-84% WR).

**Fix**: New `validate_proposals()` auto-promotes qualifying proposals from "proposed" to "testing" when:
- confidence >= "medium"
- sample_size >= 10
- expected_win_rate >= 65%

Called every 6th training cycle during the discovery step. Max 2 promotions per cycle.

## 6. Backtester Data Quality (`trainer/backtester.py`)

**Root cause**: CoinGecko was primary source — daily candles for months 4-6, hourly only for months 1-3. Daily candles flatten intraday signals, making most strategies appear to fire once.

**Fix**: CryptoCompare now tried first (free, no API key, full hourly data via pagination with `toTs`). Falls back to CoinGecko then Kraken. This gives 6 months of true hourly candles for backtesting.

---

## Files Changed

| File | Change |
|------|--------|
| `utils/risk_manager.py` | reload_state(), atomic save, max-age timeout, tick logging |
| `strategies/ema_macd.py` | Uses param_loader for ADX, RSI, SL, TP |
| `strategies/bollinger.py` | Uses param_loader for BB period/std, RSI, ADX |
| `strategies/sentiment.py` | Uses param_loader for fear/greed thresholds, SL, TP |
| `strategies/rsi_divergence.py` | Uses param_loader for RSI thresholds, SL, TP |
| `config.py` | Added ENABLE_EMA_MACD, ENABLE_BOLLINGER, ENABLE_RSI_DIVERGENCE flags |
| `bot.py` | Strategy enable/disable checks, active strategy logging |
| `trainer/engine.py` | inject_backtest_results() replaces inject_simulated_trades() |
| `trainer/discovery.py` | validate_proposals() auto-promotion |
| `trainer/backtester.py` | CryptoCompare as primary data source |

## Expected Impact

- Positions should now close reliably (state merge prevents dual-process corruption)
- Trainer tuning now takes effect on live strategies (was completely disconnected)
- Focused compute: grid + sentiment + congress_frontrun + RSI div (4 strategies vs 8)
- Training evaluates against real backtest data instead of noise
- Discovery proposals with >65% WR auto-promote to testing
- Backtests on full hourly data should show more realistic trade counts
