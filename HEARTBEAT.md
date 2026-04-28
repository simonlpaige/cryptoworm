# CryptoBot Heartbeat Checklist

## CEO (every 4 hours)
- [ ] Check Analyst's latest signal report
- [ ] Review any open positions (Trader)
- [ ] Check Sentinel risk dashboard
- [ ] Update weekly thesis if new macro data
- [ ] Assign research tasks if gaps identified

## Analyst (every 2 hours)
- [ ] Check r/bitcoin, r/cryptocurrency, r/CryptoMarkets for sentiment shift
- [ ] Check Crypto Fear & Greed Index
- [ ] Scan CoinDesk / The Block headlines
- [ ] Check BTC price action vs key levels
- [ ] Log findings in memory/YYYY-MM-DD.md
- [ ] If major signal detected → alert CEO immediately

## Quant (daily)
- [ ] Run backtests on any new strategy ideas
- [ ] Update paper trade positions
- [ ] Log paper trade P&L
- [ ] If paper trade hits 14-day threshold → prepare CEO recommendation

## Trader (every 15 min during active trades)
- [ ] Check open position P&L
- [ ] Verify stop-losses are in place
- [ ] Execute any CEO-approved trades
- [ ] Log all trades in TRADE_LOG.md

## Sentinel (every 30 min)
- [ ] Check portfolio value vs high-water mark
- [ ] Check daily P&L vs 5% limit
- [ ] Check position sizes vs 2% limit
- [ ] If ANY limit breached → force close positions + alert CEO + alert Simon
