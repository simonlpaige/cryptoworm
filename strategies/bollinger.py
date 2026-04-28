"""
Strategy 4: Bollinger Band Mean Reversion
Source: TrendRider backtests (2025-2026)
- 20-period BB with 2 std devs on 1h charts
- Buy when price closes below lower band + RSI < 30
- Short when price closes above upper band + RSI > 70
- ADX filter: only trade when ADX < 30 (ranging/choppy markets)
- Backtested: 64% win rate on ETH/BTC 1h with RSI filter
"""
import logging
import math
from typing import Optional, List
from datetime import datetime, timedelta

import config
from utils.risk_manager import RiskManager
from utils.kraken_client import KrakenClient
from strategies.ema_macd import calc_rsi, calc_adx
from trainer.param_loader import bb_period, bb_std, bb_rsi_oversold, bb_rsi_overbought, bb_adx_max

logger = logging.getLogger("cryptobot.bollinger")


def calc_bollinger(closes: list, period: int = 20, num_std: float = 2.0) -> Optional[dict]:
    """Calculate Bollinger Bands."""
    if len(closes) < period:
        return None
    recent = closes[-period:]
    sma = sum(recent) / period
    variance = sum((x - sma) ** 2 for x in recent) / period
    std = math.sqrt(variance)
    return {
        "upper": sma + num_std * std,
        "middle": sma,
        "lower": sma - num_std * std,
        "std": std,
        "bandwidth": (2 * num_std * std) / sma * 100,  # % width
    }


class BollingerMeanReversion:
    """Bollinger Band mean reversion for ranging markets."""

    def __init__(self, kraken: KrakenClient, risk: RiskManager):
        self.kraken = kraken
        self.risk = risk
        self._cooldown_minutes = 60  # don't re-enter within 1h
        self._last_entry_time = None

    def evaluate(self, current_price: float) -> list:
        actions = []

        ohlc = self.kraken.get_ohlc(interval=60, count=100)
        if not ohlc or len(ohlc) < 25:
            logger.debug("Not enough OHLC data for Bollinger")
            return actions

        closes = [c["close"] for c in ohlc]
        highs = [c["high"] for c in ohlc]
        lows = [c["low"] for c in ohlc]

        bb = calc_bollinger(closes, period=bb_period(), num_std=bb_std())
        if not bb:
            return actions

        rsi = calc_rsi(closes)
        adx = calc_adx(highs, lows, closes)

        # ADX filter — only trade ranging markets (tunable)
        adx_max = bb_adx_max()
        if adx is not None and adx >= adx_max:
            logger.debug("ADX=%.1f >= %.0f, skipping Bollinger (trending market)", adx, adx_max)
            return actions

        # Cooldown check
        if self._last_entry_time and (datetime.utcnow() - self._last_entry_time) < timedelta(minutes=self._cooldown_minutes):
            return actions

        # ── Exit logic for open positions ────────────────────────────────
        for pos in list(self.risk.positions):
            if pos["status"] != "open" or pos["strategy"] != "bollinger":
                continue
            # Longs: take profit at middle band
            if pos["side"] == "buy" and current_price >= bb["middle"]:
                logger.info("Bollinger long hit middle band ($%.2f) — closing", bb["middle"])
                result = self.risk.close_position(pos["id"], current_price)
                if result:
                    actions.append(result)
            # Shorts: take profit at middle band
            elif pos["side"] == "sell" and current_price <= bb["middle"]:
                logger.info("Bollinger short hit middle band ($%.2f) — closing", bb["middle"])
                result = self.risk.close_position(pos["id"], current_price)
                if result:
                    actions.append(result)

        has_open = any(p["status"] == "open" and p["strategy"] == "bollinger"
                       for p in self.risk.positions)
        if has_open:
            return actions

        # ── Entry signals ────────────────────────────────────────────────
        # Alternative entry: if price is >1.5 std devs outside the band, enter regardless of RSI
        extreme_lower = bb["lower"] - 1.5 * bb["std"]
        extreme_upper = bb["upper"] + 1.5 * bb["std"]
        rsi_os = bb_rsi_oversold() + 5   # relaxed: e.g. 30 → 35
        rsi_ob = bb_rsi_overbought() - 5  # relaxed: e.g. 70 → 65

        if current_price <= extreme_lower:
            # Extreme dip — enter long regardless of RSI
            action = self._open_position("buy", current_price, bb, rsi or 30)
            if action:
                actions.append(action)
        elif current_price >= extreme_upper:
            # Extreme spike — enter short regardless of RSI
            action = self._open_position("sell", current_price, bb, rsi or 70)
            if action:
                actions.append(action)
        # LONG: price below lower band + RSI oversold (tunable)
        elif current_price <= bb["lower"] and rsi is not None and rsi < rsi_os:
            action = self._open_position("buy", current_price, bb, rsi)
            if action:
                actions.append(action)
        # SHORT: price above upper band + RSI overbought (tunable)
        elif current_price >= bb["upper"] and rsi is not None and rsi > rsi_ob:
            action = self._open_position("sell", current_price, bb, rsi)
            if action:
                actions.append(action)

        return actions

    def _open_position(self, side: str, price: float, bb: dict, rsi: float) -> Optional[dict]:
        can_open, reason = self.risk.can_open_position(price, side=side, strategy="bollinger")
        if not can_open:
            logger.info("Bollinger %s blocked: %s", side, reason)
            return None

        size_btc = self.risk.position_size_btc(price)
        if side == "buy":
            stop_loss = price * (1 - 1.0 / 100)  # 1% below recent low
            take_profit = bb["middle"]  # target: middle band
        else:
            stop_loss = price * (1 + 1.0 / 100)
            take_profit = bb["middle"]

        pos = self.risk.open_position(
            side=side,
            price=price,
            size_btc=size_btc,
            strategy="bollinger",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._last_entry_time = datetime.utcnow()
        logger.info("Bollinger %s: price=$%.2f, lower=$%.2f, upper=$%.2f, RSI=%.1f",
                     side.upper(), price, bb["lower"], bb["upper"], rsi)
        return pos
