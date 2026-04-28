# CryptoBot — Trading Strategies

## Strategy 1: Grid Bot (Passive — Phase 1)
- **Instrument:** BTC/USDT
- **Platform:** Pionex (free grid bot)
- **Allocation:** $400 of $500 (keep $100 reserve)
- **Grid range:** Set around current BTC price ±10%
- **Expected return:** 5-15% annually in sideways market
- **Risk:** If BTC drops below grid range, holds BTC at loss until recovery
- **Backtest required:** Yes — test against last 6 months of BTC price data

## Strategy 2: Sentiment-Driven Swing (Active — Phase 1 paper only)
- **Instrument:** BTC/USDT
- **Trigger:** Analyst's daily sentiment score crosses threshold
- **Long Entry:** Buy when extreme fear (≤25) + positive divergence (price >1% above 24h low)
- **Short Entry:** Sell when extreme greed (≥75) + negative divergence (price >1% below 24h high)
- **Long Exit:** Sell at 5% profit, 2% stop-loss, or when FNG crosses into extreme greed
- **Short Exit:** Cover at 5% profit, 2% stop-loss, or when FNG crosses into extreme fear
- **Position size:** Max 2% of account per trade
- **Holding period:** 2-7 days
- **Backtest required:** Yes — test against Fear & Greed Index + price data

## Strategy 3: EMA/MACD Momentum (Active — Phase 1 paper)
- **Source:** TrendRider backtests (Jan 2025 – Mar 2026), 62% win rate, 1.95 profit factor
- **Instrument:** BTC/USD
- **Timeframe:** 1h candles
- **Long Entry:** 12/26 EMA crossover + MACD histogram positive & increasing + RSI 40-70
- **Short Entry:** EMA bearish cross + MACD histogram negative & decreasing + RSI 30-60
- **Filter:** ADX > 25 (only trade trending markets)
- **Exit:** MACD histogram flips sign
- **SL/TP:** 1.8% stop / 3.2% target
- **Position size:** Max 2% of account
- **Cooldown:** 1h between entries

## Strategy 4: Bollinger Band Mean Reversion (Active — Phase 1 paper)
- **Source:** TrendRider backtests, 64% win rate on ETH/BTC 1h with RSI filter
- **Instrument:** BTC/USD
- **Timeframe:** 1h candles
- **Long Entry:** Price closes below lower BB (20,2) + RSI < 30
- **Short Entry:** Price closes above upper BB (20,2) + RSI > 70
- **Filter:** ADX < 30 (only trade ranging/choppy markets)
- **Exit:** Price reaches middle band (20-period SMA)
- **SL:** 1% beyond entry
- **Position size:** Max 2% of account
- **Cooldown:** 2h between entries

## Strategy 5: RSI Divergence (Active — Phase 1 paper)
- **Source:** TrendRider + Reddit r/Daytrading consensus, 58% win rate, 1:1.5 R:R
- **Instrument:** BTC/USD
- **Timeframe:** 4h candles (best signal-to-noise per Reddit)
- **Long Entry:** Bullish divergence (price lower low + RSI higher low) + RSI < 40
- **Short Entry:** Bearish divergence (price higher high + RSI lower high) + RSI > 60
- **SL/TP:** 2% stop / 3% target (1:1.5 risk-reward)
- **Max hold:** 7 days
- **Position size:** Max 2% of account
- **Cooldown:** 4h between entries

## Strategy Approval Process
1. Quant develops and backtests strategy (minimum 6 months historical data)
2. Quant paper trades for minimum 14 days
3. Quant presents results to CEO with win rate, max drawdown, Sharpe ratio
4. CEO approves/rejects
5. If approved for live: Trader implements with half the recommended position size for first 2 weeks
6. Simon notified of any new live strategy deployment
