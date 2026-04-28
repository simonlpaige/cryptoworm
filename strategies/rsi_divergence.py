"""
Strategy 5: RSI Divergence
Sources: TrendRider backtests, Reddit r/Daytrading, r/BitcoinMarkets
- Bullish divergence: price makes lower low, RSI makes higher low → long
- Bearish divergence: price makes higher high, RSI makes lower high → short
- Best on 4h timeframes per Reddit consensus
- Confirmed at S/R levels (uses recent swing highs/lows)
- 58% win rate, 1.5 risk-reward per backtests
"""
import logging
from typing import Optional, List
from datetime import datetime, timedelta

import config
from utils.risk_manager import RiskManager
from utils.kraken_client import KrakenClient
from strategies.ema_macd import calc_rsi
from trainer.param_loader import rsi_div_long_threshold, rsi_div_short_threshold, rsi_div_sl, rsi_div_tp

logger = logging.getLogger("cryptobot.rsi_div")


def find_swing_lows(prices: list, window: int = 5) -> list:
    """Find swing low indices (local minima)."""
    swings = []
    for i in range(window, len(prices) - window):
        if all(prices[i] <= prices[i - j] for j in range(1, window + 1)) and \
           all(prices[i] <= prices[i + j] for j in range(1, window + 1)):
            swings.append(i)
    return swings


def find_swing_highs(prices: list, window: int = 5) -> list:
    """Find swing high indices (local maxima)."""
    swings = []
    for i in range(window, len(prices) - window):
        if all(prices[i] >= prices[i - j] for j in range(1, window + 1)) and \
           all(prices[i] >= prices[i + j] for j in range(1, window + 1)):
            swings.append(i)
    return swings


def rsi_series(prices: list, period: int = 14) -> list:
    """Calculate RSI for each point in the series (returns aligned to prices)."""
    if len(prices) < period + 1:
        return []
    result = [None] * (period + 1)
    for i in range(period + 1, len(prices) + 1):
        r = calc_rsi(prices[:i], period)
        result.append(r)
    return result


class RsiDivergence:
    """RSI divergence strategy — catches reversals on 4h timeframes."""

    def __init__(self, kraken: KrakenClient, risk: RiskManager):
        self.kraken = kraken
        self.risk = risk
        self._cooldown_minutes = 120  # 2h cooldown between entries
        self._last_entry_time = None
        self._lookback_swings = 15  # how far back to look for divergence

    def evaluate(self, current_price: float) -> list:
        actions = []

        # Use 4h candles for divergence (240 min)
        ohlc = self.kraken.get_ohlc(interval=240, count=100)
        if not ohlc or len(ohlc) < 30:
            logger.debug("Not enough 4h OHLC data for RSI divergence")
            return actions

        closes = [c["close"] for c in ohlc]

        # ── Exit logic ───────────────────────────────────────────────────
        for pos in list(self.risk.positions):
            if pos["status"] != "open" or pos["strategy"] != "rsi_divergence":
                continue
            # Time-based exit: 7 days max
            opened = datetime.fromisoformat(pos["opened_at"])
            if datetime.utcnow() - opened > timedelta(days=7):
                logger.info("RSI divergence position %s held 7 days — closing", pos["id"])
                result = self.risk.close_position(pos["id"], current_price)
                if result:
                    actions.append(result)

        has_open = any(p["status"] == "open" and p["strategy"] == "rsi_divergence"
                       for p in self.risk.positions)
        if has_open:
            return actions

        # Cooldown
        if self._last_entry_time and (datetime.utcnow() - self._last_entry_time) < timedelta(minutes=self._cooldown_minutes):
            return actions

        # Calculate RSI series
        rsi_vals = rsi_series(closes)
        if len(rsi_vals) < len(closes):
            rsi_vals = [None] * (len(closes) - len(rsi_vals)) + rsi_vals

        # Find swing points
        swing_lows = find_swing_lows(closes, window=2)
        swing_highs = find_swing_highs(closes, window=2)

        # ── Bullish divergence (long signal) ─────────────────────────────
        # Price: lower low, RSI: higher low
        recent_lows = [i for i in swing_lows if i >= len(closes) - self._lookback_swings]
        if len(recent_lows) >= 2:
            prev_low_idx = recent_lows[-2]
            curr_low_idx = recent_lows[-1]
            prev_rsi = rsi_vals[prev_low_idx] if prev_low_idx < len(rsi_vals) else None
            curr_rsi = rsi_vals[curr_low_idx] if curr_low_idx < len(rsi_vals) else None

            if prev_rsi and curr_rsi:
                price_lower_low = closes[curr_low_idx] < closes[prev_low_idx]
                rsi_higher_low = curr_rsi > prev_rsi

                if price_lower_low and rsi_higher_low and curr_rsi < rsi_div_long_threshold() + 5:
                    logger.info("BULLISH DIVERGENCE: price LL (%.2f < %.2f), RSI HL (%.1f > %.1f)",
                                closes[curr_low_idx], closes[prev_low_idx], curr_rsi, prev_rsi)
                    action = self._open_position("buy", current_price, curr_rsi)
                    if action:
                        actions.append(action)

        # ── Bearish divergence (short signal) ────────────────────────────
        # Price: higher high, RSI: lower high
        recent_highs = [i for i in swing_highs if i >= len(closes) - self._lookback_swings]
        if len(recent_highs) >= 2:
            prev_high_idx = recent_highs[-2]
            curr_high_idx = recent_highs[-1]
            prev_rsi = rsi_vals[prev_high_idx] if prev_high_idx < len(rsi_vals) else None
            curr_rsi = rsi_vals[curr_high_idx] if curr_high_idx < len(rsi_vals) else None

            if prev_rsi and curr_rsi:
                price_higher_high = closes[curr_high_idx] > closes[prev_high_idx]
                rsi_lower_high = curr_rsi < prev_rsi

                if price_higher_high and rsi_lower_high and curr_rsi > rsi_div_short_threshold() - 5:
                    logger.info("BEARISH DIVERGENCE: price HH (%.2f > %.2f), RSI LH (%.1f < %.1f)",
                                closes[curr_high_idx], closes[prev_high_idx], curr_rsi, prev_rsi)
                    action = self._open_position("sell", current_price, curr_rsi)
                    if action:
                        actions.append(action)

        return actions

    def _open_position(self, side: str, price: float, rsi: float) -> Optional[dict]:
        can_open, reason = self.risk.can_open_position(price, side=side, strategy="rsi_divergence")
        if not can_open:
            logger.info("RSI divergence %s blocked: %s", side, reason)
            return None

        size_btc = self.risk.position_size_btc(price)
        sl_pct = rsi_div_sl()
        tp_pct = rsi_div_tp()
        if side == "buy":
            stop_loss = price * (1 - sl_pct / 100)
            take_profit = price * (1 + tp_pct / 100)
        else:
            stop_loss = price * (1 + sl_pct / 100)
            take_profit = price * (1 - tp_pct / 100)

        pos = self.risk.open_position(
            side=side,
            price=price,
            size_btc=size_btc,
            strategy="rsi_divergence",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._last_entry_time = datetime.utcnow()
        logger.info("RSI Divergence %s: price=$%.2f, RSI=%.1f", side.upper(), price, rsi)
        return pos
