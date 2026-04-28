"""
Regime Detector — classifies market as 'trending', 'ranging', or 'neutral'.

Uses ADX (Average Directional Index) + ATR (Average True Range) to determine
the current market regime. This controls which strategies are active:
  - Trending (ADX > 25): momentum strategies only (sentiment, rsi_divergence, congress_frontrun)
  - Ranging  (ADX < 20): grid only
  - Neutral  (20 <= ADX <= 25): all strategies run
"""
import logging
from typing import Optional

from strategies.ema_macd import calc_adx

logger = logging.getLogger("cryptobot.regime")

# ADX thresholds for regime classification
ADX_TRENDING_THRESHOLD = 25
ADX_RANGING_THRESHOLD = 20


def calc_atr(highs: list, lows: list, closes: list, period: int = 14) -> Optional[float]:
    """Calculate Average True Range over the given period."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    # Smoothed ATR (Wilder's method)
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


class RegimeDetector:
    """Classifies the market regime using ADX and ATR from OHLC data."""

    def __init__(self, kraken_client):
        self.kraken = kraken_client
        self._current_regime = "neutral"
        self._adx_value = None
        self._atr_value = None

    @property
    def regime(self) -> str:
        return self._current_regime

    @property
    def adx(self) -> Optional[float]:
        return self._adx_value

    @property
    def atr(self) -> Optional[float]:
        return self._atr_value

    def update(self) -> str:
        """Fetch fresh OHLC data and recalculate regime.

        Returns the regime string: 'trending', 'ranging', or 'neutral'.
        """
        ohlc = self.kraken.get_ohlc(interval=60, count=100)
        if not ohlc or len(ohlc) < 30:
            logger.debug("Not enough OHLC data for regime detection — defaulting to neutral")
            self._current_regime = "neutral"
            return self._current_regime

        highs = [c["high"] for c in ohlc]
        lows = [c["low"] for c in ohlc]
        closes = [c["close"] for c in ohlc]

        self._adx_value = calc_adx(highs, lows, closes)
        self._atr_value = calc_atr(highs, lows, closes)

        if self._adx_value is None:
            self._current_regime = "neutral"
        elif self._adx_value > ADX_TRENDING_THRESHOLD:
            self._current_regime = "trending"
        elif self._adx_value < ADX_RANGING_THRESHOLD:
            self._current_regime = "ranging"
        else:
            self._current_regime = "neutral"

        logger.info("Regime: %s (ADX=%.1f, ATR=%.2f)",
                     self._current_regime.upper(),
                     self._adx_value if self._adx_value else 0,
                     self._atr_value if self._atr_value else 0)
        return self._current_regime
