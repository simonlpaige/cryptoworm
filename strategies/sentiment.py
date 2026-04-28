"""
Strategy 2: Sentiment-Driven Swing Trade
- Uses Fear & Greed Index as signal
- Buy on extreme fear (<=25) with positive divergence
- Sell at 5-8% profit or 2% stop-loss
- Holding period: 2-7 days
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

import config
from utils.risk_manager import RiskManager
from utils.kraken_client import KrakenClient
from trainer.param_loader import fear_threshold, greed_threshold, sentiment_tp, sentiment_sl
logger = logging.getLogger("cryptobot.sentiment")


class SentimentSwing:
    """Paper-trading sentiment swing strategy."""

    # Fear & Greed updates once daily — only evaluate every 12 hours (72 ticks at 5min)
    EVAL_INTERVAL_TICKS = 72

    def __init__(self, kraken: KrakenClient, risk: RiskManager):
        self.kraken = kraken
        self.risk = risk
        self._last_fng: Optional[dict] = None
        self._prev_fng: Optional[dict] = None
        self._last_fetch = 0.0
        self._fetch_interval = 3600  # fetch FNG at most once per hour
        self._tick_count = 0
        self._last_eval_tick = -self.EVAL_INTERVAL_TICKS  # allow first tick to evaluate

    def _fetch_fear_greed(self) -> Optional[dict]:
        """Fetch current Fear & Greed Index."""
        import time
        if time.time() - self._last_fetch < self._fetch_interval and self._last_fng:
            return self._last_fng

        try:
            resp = requests.get(config.SENTIMENT_API_URL, params={"limit": 2}, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if len(data) >= 2:
                self._prev_fng = {
                    "value": int(data[1]["value"]),
                    "classification": data[1]["value_classification"],
                }
                self._last_fng = {
                    "value": int(data[0]["value"]),
                    "classification": data[0]["value_classification"],
                }
            elif len(data) == 1:
                self._prev_fng = self._last_fng
                self._last_fng = {
                    "value": int(data[0]["value"]),
                    "classification": data[0]["value_classification"],
                }
            self._last_fetch = time.time()
            logger.info("Fear & Greed Index: %s (%s)",
                        self._last_fng["value"], self._last_fng["classification"])
            return self._last_fng
        except Exception as e:
            logger.error("Failed to fetch Fear & Greed: %s", e)
            return self._last_fng  # return cached if available

    def _has_positive_divergence(self, current_price: float) -> bool:
        """
        Simple divergence check: FNG is in extreme fear but price is not
        making new lows (price above 24h low by >1%).
        """
        ticker = self.kraken.get_ticker()
        if not ticker:
            return False
        low_24h = ticker["low_24h"]
        # Price is >0.5% above 24h low = potential positive divergence
        return current_price > low_24h * 1.005

    def evaluate(self, current_price: float) -> list:
        """Evaluate sentiment signals and manage positions.
        Only checks for new entries every 72 ticks (~12h) since Fear & Greed updates daily.
        Exit checks still run every tick for open positions.
        """
        actions = []
        self._tick_count += 1

        fng = self._fetch_fear_greed()
        if not fng:
            return actions

        # Always check exits on open positions (every tick)
        for pos in list(self.risk.positions):
            if pos["status"] != "open" or pos["strategy"] != "sentiment":
                continue
            # Time-based exit: close after 7 days max
            opened = datetime.fromisoformat(pos["opened_at"])
            if datetime.utcnow() - opened > timedelta(days=7):
                logger.info("Sentiment position %s held 7 days — closing", pos["id"])
                result = self.risk.close_position(pos["id"], current_price)
                if result:
                    actions.append(result)
                continue
            # Sentiment-based exit: close longs on extreme greed, close shorts on extreme fear
            if pos["side"] == "buy" and fng["value"] >= greed_threshold():
                logger.info("Extreme greed (%d) — closing long sentiment position", fng["value"])
                result = self.risk.close_position(pos["id"], current_price)
                if result:
                    actions.append(result)
            elif pos["side"] == "sell" and fng["value"] <= fear_threshold():
                logger.info("Extreme fear (%d) — closing short sentiment position", fng["value"])
                result = self.risk.close_position(pos["id"], current_price)
                if result:
                    actions.append(result)

        # Throttle entry checks: only every EVAL_INTERVAL_TICKS ticks (~12h)
        if (self._tick_count - self._last_eval_tick) < self.EVAL_INTERVAL_TICKS:
            return actions
        self._last_eval_tick = self._tick_count
        logger.info("Sentiment entry check (tick %d, every %d ticks)", self._tick_count, self.EVAL_INTERVAL_TICKS)

        # ── Check for entry signals ──────────────────────────────────────
        has_open = any(p["status"] == "open" and p["strategy"] == "sentiment"
                       for p in self.risk.positions)

        if not has_open:
            # BUY signal: extreme fear + positive divergence (tunable)
            if fng["value"] <= fear_threshold():
                if self._has_positive_divergence(current_price):
                    action = self._open_long(current_price)
                    if action:
                        actions.append(action)
                else:
                    logger.info("Extreme fear (%d) but no positive divergence — waiting",
                                fng["value"])
            # SHORT signal: extreme greed + negative divergence (tunable)
            elif fng["value"] >= greed_threshold():
                if self._has_negative_divergence(current_price):
                    action = self._open_short(current_price)
                    if action:
                        actions.append(action)
                else:
                    logger.info("Extreme greed (%d) but no negative divergence — waiting",
                                fng["value"])

        return actions

    def _has_negative_divergence(self, current_price: float) -> bool:
        """
        Negative divergence: FNG is in extreme greed but price is not
        making new highs (price below 24h high by >1%).
        """
        ticker = self.kraken.get_ticker()
        if not ticker:
            return False
        high_24h = ticker["high_24h"]
        # Price is >0.5% below 24h high = potential negative divergence
        return current_price < high_24h * 0.995

    def _open_long(self, price: float) -> Optional[dict]:
        """Open a sentiment-driven long position."""
        can_open, reason = self.risk.can_open_position(price, side="buy", strategy="sentiment")
        if not can_open:
            logger.info("Sentiment buy blocked: %s", reason)
            return None

        size_btc = self.risk.position_size_btc(price)
        sl_pct = sentiment_sl()
        tp_pct = sentiment_tp()
        stop_loss = price * (1 - sl_pct / 100)
        take_profit = price * (1 + tp_pct / 100)

        pos = self.risk.open_position(
            side="buy",
            price=price,
            size_btc=size_btc,
            strategy="sentiment",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        logger.info("Sentiment BUY: FNG=%d, price=$%.2f, size=%.6f BTC",
                     self._last_fng["value"], price, size_btc)
        return pos

    def _open_short(self, price: float) -> Optional[dict]:
        """Open a sentiment-driven short position (paper only)."""
        can_open, reason = self.risk.can_open_position(price, side="sell", strategy="sentiment")
        if not can_open:
            logger.info("Sentiment short blocked: %s", reason)
            return None

        size_btc = self.risk.position_size_btc(price)
        sl_pct = sentiment_sl()
        tp_pct = sentiment_tp()
        stop_loss = price * (1 + sl_pct / 100)   # SL above entry
        take_profit = price * (1 - tp_pct / 100)  # TP below entry

        pos = self.risk.open_position(
            side="sell",
            price=price,
            size_btc=size_btc,
            strategy="sentiment",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        logger.info("Sentiment SHORT: FNG=%d, price=$%.2f, size=%.6f BTC",
                     self._last_fng["value"], price, size_btc)
        return pos
