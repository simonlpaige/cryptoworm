"""
Strategy 3: EMA/MACD Momentum
Source: TrendRider backtests (Jan 2025 - Mar 2026)
- 12/26 EMA crossover + MACD histogram confirmation
- RSI filter: only enter when RSI 40-70 (longs) or 30-60 (shorts)
- ADX filter: only trade when ADX > 25 (trending market)
- Backtested: 62% win rate, 1.95 profit factor on BTC/ETH/SOL 1h
"""
import logging
from typing import Optional, List
from datetime import datetime, timedelta

import config
from utils.risk_manager import RiskManager
from utils.kraken_client import KrakenClient
from trainer.param_loader import ema_adx_threshold, ema_rsi_long_range, ema_rsi_short_range, ema_sl, ema_tp
logger = logging.getLogger("cryptobot.ema_macd")


def ema(prices: list, period: int) -> list:
    """Calculate EMA from a price list."""
    if len(prices) < period:
        return []
    k = 2 / (period + 1)
    result = [sum(prices[:period]) / period]
    for price in prices[period:]:
        result.append(price * k + result[-1] * (1 - k))
    return result


def calc_rsi(prices: list, period: int = 14) -> Optional[float]:
    """Calculate RSI from closing prices."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = deltas[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0.001
    avg_loss = sum(losses) / period if losses else 0.001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_adx(highs: list, lows: list, closes: list, period: int = 14) -> Optional[float]:
    """Simplified ADX calculation."""
    if len(closes) < period * 2:
        return None
    # True Range
    trs = []
    plus_dms = []
    minus_dms = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
        plus_dm = max(highs[i] - highs[i - 1], 0) if (highs[i] - highs[i - 1]) > (lows[i - 1] - lows[i]) else 0
        minus_dm = max(lows[i - 1] - lows[i], 0) if (lows[i - 1] - lows[i]) > (highs[i] - highs[i - 1]) else 0
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)

    if len(trs) < period:
        return None

    # Smoothed averages
    atr = sum(trs[:period]) / period
    plus_di_sum = sum(plus_dms[:period]) / period
    minus_di_sum = sum(minus_dms[:period]) / period

    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        plus_di_sum = (plus_di_sum * (period - 1) + plus_dms[i]) / period
        minus_di_sum = (minus_di_sum * (period - 1) + minus_dms[i]) / period

    if atr == 0:
        return None
    plus_di = 100 * plus_di_sum / atr
    minus_di = 100 * minus_di_sum / atr
    di_sum = plus_di + minus_di
    if di_sum == 0:
        return None
    dx = 100 * abs(plus_di - minus_di) / di_sum
    return dx  # simplified — true ADX smooths DX over another period


class EmaMacdMomentum:
    """EMA/MACD momentum strategy for trending markets."""

    def __init__(self, kraken: KrakenClient, risk: RiskManager):
        self.kraken = kraken
        self.risk = risk
        self._last_signal = None  # "long" or "short" or None
        self._signal_time = None
        self._cooldown_minutes = 30  # don't re-enter same direction within 30min

    def evaluate(self, current_price: float) -> list:
        """Evaluate EMA/MACD signals. Returns list of actions."""
        actions = []

        # Fetch OHLC data (1h candles, last 100)
        ohlc = self.kraken.get_ohlc(interval=60, count=100)
        if not ohlc or len(ohlc) < 30:
            logger.debug("Not enough OHLC data for EMA/MACD")
            return actions

        closes = [c["close"] for c in ohlc]
        highs = [c["high"] for c in ohlc]
        lows = [c["low"] for c in ohlc]

        # Calculate indicators
        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        if not ema12 or not ema26:
            return actions

        # Align EMAs (ema26 starts later)
        offset = len(ema12) - len(ema26)
        ema12_aligned = ema12[offset:]

        # MACD line and histogram
        macd_line = [e12 - e26 for e12, e26 in zip(ema12_aligned, ema26)]
        signal_line = ema(macd_line, 9) if len(macd_line) >= 9 else []
        if not signal_line:
            return actions

        sl_offset = len(macd_line) - len(signal_line)
        macd_hist = [m - s for m, s in zip(macd_line[sl_offset:], signal_line)]

        if len(macd_hist) < 2:
            return actions

        # Current values
        curr_hist = macd_hist[-1]
        prev_hist = macd_hist[-2]
        rsi = calc_rsi(closes)
        adx = calc_adx(highs, lows, closes)

        # ADX filter — only trade trending markets (tunable via trainer)
        adx_thresh = ema_adx_threshold()
        if adx is not None and adx < adx_thresh:
            logger.debug("ADX=%.1f < %.0f, skipping EMA/MACD (ranging market)", adx, adx_thresh)
            return actions

        # Check cooldown
        if self._signal_time and (datetime.utcnow() - self._signal_time) < timedelta(minutes=self._cooldown_minutes):
            return actions

        has_open = any(p["status"] == "open" and p["strategy"] == "ema_macd"
                       for p in self.risk.positions)
        if has_open:
            # Check for exit signals on open positions
            for pos in list(self.risk.positions):
                if pos["status"] != "open" or pos["strategy"] != "ema_macd":
                    continue
                # Exit long if MACD histogram turns negative
                if pos["side"] == "buy" and curr_hist < 0 and prev_hist >= 0:
                    logger.info("EMA/MACD histogram flipped negative — closing long")
                    result = self.risk.close_position(pos["id"], current_price)
                    if result:
                        actions.append(result)
                # Exit short if MACD histogram turns positive
                elif pos["side"] == "sell" and curr_hist > 0 and prev_hist <= 0:
                    logger.info("EMA/MACD histogram flipped positive — closing short")
                    result = self.risk.close_position(pos["id"], current_price)
                    if result:
                        actions.append(result)
            return actions

        # ── Trend filter: 50-period SMA ─────────────────────────────────
        sma50 = sum(closes[-50:]) / min(len(closes), 50) if len(closes) >= 50 else None

        # ── Entry signals ────────────────────────────────────────────────
        # LONG: histogram positive and increasing, RSI in tunable range, price > SMA50
        rsi_long_lo, rsi_long_hi = ema_rsi_long_range()
        rsi_short_lo, rsi_short_hi = ema_rsi_short_range()
        if curr_hist > 0 and curr_hist > prev_hist:
            if rsi and rsi_long_lo <= rsi <= rsi_long_hi:
                if sma50 is None or current_price > sma50:
                    action = self._open_position("buy", current_price, rsi, adx)
                    if action:
                        actions.append(action)
                        self._last_signal = "long"
                        self._signal_time = datetime.utcnow()
                else:
                    logger.debug("EMA/MACD long blocked: price $%.2f < SMA50 $%.2f", current_price, sma50)

        # SHORT: histogram negative and decreasing, RSI in tunable range, price < SMA50
        elif curr_hist < 0 and curr_hist < prev_hist:
            if rsi and rsi_short_lo <= rsi <= rsi_short_hi:
                if sma50 is None or current_price < sma50:
                    action = self._open_position("sell", current_price, rsi, adx)
                    if action:
                        actions.append(action)
                        self._last_signal = "short"
                        self._signal_time = datetime.utcnow()
                else:
                    logger.debug("EMA/MACD short blocked: price $%.2f > SMA50 $%.2f", current_price, sma50)

        return actions

    def _open_position(self, side: str, price: float, rsi: float, adx: Optional[float]) -> Optional[dict]:
        can_open, reason = self.risk.can_open_position(price, side=side, strategy="ema_macd")
        if not can_open:
            logger.info("EMA/MACD %s blocked: %s", side, reason)
            return None

        size_btc = self.risk.position_size_btc(price)
        sl_pct = ema_sl()
        tp_pct = ema_tp()
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
            strategy="ema_macd",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        logger.info("EMA/MACD %s: price=$%.2f, RSI=%.1f, ADX=%s",
                     side.upper(), price, rsi, f"{adx:.1f}" if adx else "N/A")
        return pos
