"""
Regime Detector - classifies market as 'trending', 'ranging', or 'neutral'.

Two views of the same data:

ADX/ATR view (the simple one):
  - Trending (ADX > 25): momentum strategies only
  - Ranging  (ADX < 20): grid only
  - Neutral  (20-25):    all strategies run

Change-point view (the new one):
  detect_regime_change() runs PELT on the close series. PELT looks for
  the timestamps where the underlying price process broke - the moments
  where what worked yesterday stops working today. If we're trading
  against a trend that just broke, we want to know about it. If
  ruptures isn't installed we fall back to a rolling-std heuristic so
  this method never crashes the bot.
"""
import logging
import math
from typing import Optional, List, Dict

from strategies.ema_macd import calc_adx

try:
    import ruptures
except ImportError:
    ruptures = None

logger = logging.getLogger("cryptoworm.regime")

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
        # Cache last detection so we can warn on mid-trend changes
        self._last_change_result: Dict = {}

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
            logger.debug("Not enough OHLC data for regime detection - defaulting to neutral")
            self._current_regime = "neutral"
            return self._current_regime

        highs = [c["high"] for c in ohlc]
        lows = [c["low"] for c in ohlc]
        closes = [c["close"] for c in ohlc]

        self._adx_value = calc_adx(highs, lows, closes)
        self._atr_value = calc_atr(highs, lows, closes)

        prev_regime = self._current_regime
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

        # Run PELT change-point detection alongside ADX. Warn if a
        # structural break landed while we still think we're trending -
        # that's when momentum strategies print false signals.
        try:
            change = self.detect_regime_change(closes)
            self._last_change_result = change
            if change.get("change_detected") and prev_regime == "trending":
                logger.warning(
                    "Regime change detected mid-trend (segment age=%d, breakpoints=%s) - "
                    "treat trending signals with extra suspicion",
                    change.get("current_segment_age", 0),
                    change.get("breakpoints", []),
                )
        except Exception as e:
            logger.debug("change-point detection failed: %s", e)

        return self._current_regime

    def detect_regime_change(self, closes: List[float], sensitivity: str = "medium") -> dict:
        """Detect structural breaks in the price series via PELT.

        Returns: {
            'change_detected': bool,
            'breakpoints': list[int] (indices where breaks were found),
            'current_segment_age': int (candles since last breakpoint),
            'method': 'pelt' or 'rolling_std' (fallback),
        }

        sensitivity tunes the penalty term: 'low' = fewer breaks
        reported, 'high' = more breaks. The default 'medium' is what
        you want unless you're tuning by hand.
        """
        if not closes or len(closes) < 20:
            return {
                "change_detected": False,
                "breakpoints": [],
                "current_segment_age": len(closes) if closes else 0,
                "method": "pelt" if ruptures else "rolling_std",
            }

        # Penalty controls how reluctant PELT is to call a break. Bigger
        # penalty -> fewer, more confident breaks. We scale by log(n)
        # which is the BIC-style heuristic.
        penalty_map = {"low": 20.0, "medium": 10.0, "high": 5.0}
        base_pen = penalty_map.get(sensitivity, 10.0)
        penalty = base_pen * math.log(len(closes))

        if ruptures is not None:
            try:
                signal = [[c] for c in closes]
                algo = ruptures.Pelt(model="rbf").fit(signal)
                bkps = algo.predict(pen=penalty)
                # ruptures returns end-of-segment indices including final n
                breakpoints = [b for b in bkps if b < len(closes)]
                last = breakpoints[-1] if breakpoints else 0
                return {
                    "change_detected": len(breakpoints) > 0,
                    "breakpoints": breakpoints,
                    "current_segment_age": len(closes) - last,
                    "method": "pelt",
                }
            except Exception as e:
                logger.debug("PELT failed (%s); falling back to rolling-std", e)

        # Fallback: rolling-std change detection. We compare the most
        # recent window's standard deviation to the prior window. If
        # the ratio swings hard, call it a break.
        win = max(10, len(closes) // 10)
        if len(closes) < 2 * win:
            return {
                "change_detected": False,
                "breakpoints": [],
                "current_segment_age": len(closes),
                "method": "rolling_std",
            }
        breakpoints = []
        for i in range(2 * win, len(closes), win):
            recent = closes[i - win:i]
            prior = closes[i - 2 * win:i - win]
            std_recent = _std(recent)
            std_prior = _std(prior)
            if std_prior <= 0:
                continue
            ratio = std_recent / std_prior
            threshold = {"low": 3.0, "medium": 2.0, "high": 1.5}.get(sensitivity, 2.0)
            if ratio > threshold or ratio < 1.0 / threshold:
                breakpoints.append(i)
        last = breakpoints[-1] if breakpoints else 0
        return {
            "change_detected": len(breakpoints) > 0,
            "breakpoints": breakpoints,
            "current_segment_age": len(closes) - last,
            "method": "rolling_std",
        }


def _std(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var) if var > 0 else 0.0
