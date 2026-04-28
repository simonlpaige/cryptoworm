"""
Feature Engineering Pipeline
=============================
Computes features from OHLC data for ML signal generation (XGBoost).
Outputs a flat dict of named features. Missing data returns None per feature
(XGBoost handles NaN natively).
"""
import logging
import math
import time
from typing import Optional, Dict, Any, List

import requests

from strategies.ema_macd import calc_rsi, calc_adx, ema

logger = logging.getLogger("cryptobot.features")

# Fear & Greed cache (shared across calls, max once per hour)
_fng_cache: Optional[Dict[str, Any]] = None
_fng_cache_time: float = 0.0
_FNG_CACHE_TTL = 3600  # 1 hour


def _safe_log(x: float) -> Optional[float]:
    """Safe log return."""
    if x is None or x <= 0:
        return None
    try:
        return math.log(x)
    except (ValueError, OverflowError):
        return None


def _returns(closes: List[float], period: int) -> Optional[float]:
    """Simple return over N periods."""
    if len(closes) < period + 1:
        return None
    prev = closes[-(period + 1)]
    if prev == 0:
        return None
    return (closes[-1] - prev) / prev


def _log_returns(closes: List[float], period: int) -> Optional[float]:
    """Log return over N periods."""
    if len(closes) < period + 1:
        return None
    prev = closes[-(period + 1)]
    curr = closes[-1]
    if prev <= 0 or curr <= 0:
        return None
    try:
        return math.log(curr / prev)
    except (ValueError, OverflowError):
        return None


def _realized_volatility(closes: List[float], window: int) -> Optional[float]:
    """Realized volatility as std of log returns over window."""
    if len(closes) < window + 1:
        return None
    log_rets = []
    for i in range(-window, 0):
        prev = closes[i - 1]
        curr = closes[i]
        if prev <= 0 or curr <= 0:
            continue
        try:
            log_rets.append(math.log(curr / prev))
        except (ValueError, OverflowError):
            continue
    if len(log_rets) < 3:
        return None
    mean = sum(log_rets) / len(log_rets)
    variance = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
    return math.sqrt(variance)


def _calc_macd(closes: List[float]) -> Optional[Dict[str, float]]:
    """Calculate MACD line, signal, histogram."""
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    if not ema12 or not ema26:
        return None
    offset = len(ema12) - len(ema26)
    ema12_aligned = ema12[offset:]
    macd_line = [e12 - e26 for e12, e26 in zip(ema12_aligned, ema26)]
    signal_line = ema(macd_line, 9) if len(macd_line) >= 9 else None
    if not signal_line:
        return None
    return {
        "macd_line": macd_line[-1],
        "macd_signal": signal_line[-1],
        "macd_histogram": macd_line[-1] - signal_line[-1],
    }


def _calc_bollinger_width(closes: List[float], period: int = 20) -> Optional[float]:
    """Bollinger Band width (upper - lower) / middle."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    sma = sum(window) / period
    if sma == 0:
        return None
    std = math.sqrt(sum((x - sma) ** 2 for x in window) / period)
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / sma


def _calc_atr_ratio(highs: List[float], lows: List[float], closes: List[float],
                    period: int = 14) -> Optional[float]:
    """ATR as a ratio of current price."""
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
    atr = sum(trs[-period:]) / period
    if closes[-1] == 0:
        return None
    return atr / closes[-1]


def _calc_obv_trend(closes: List[float], volumes: List[float], period: int = 14) -> Optional[float]:
    """OBV trend: slope of OBV over last N periods, normalized."""
    if len(closes) < period + 1 or len(volumes) < period + 1:
        return None
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    recent_obv = obv[-period:]
    if len(recent_obv) < 2:
        return None
    # Simple linear slope
    n = len(recent_obv)
    x_mean = (n - 1) / 2
    y_mean = sum(recent_obv) / n
    num = sum((i - x_mean) * (recent_obv[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    slope = num / den
    # Normalize by mean OBV magnitude
    if abs(y_mean) < 1e-10:
        return 0.0
    return slope / abs(y_mean)


def _calc_vwap_deviation(closes: List[float], volumes: List[float],
                         period: int = 20) -> Optional[float]:
    """VWAP deviation: (price - VWAP) / price."""
    if len(closes) < period or len(volumes) < period:
        return None
    c = closes[-period:]
    v = volumes[-period:]
    total_vol = sum(v)
    if total_vol == 0 or closes[-1] == 0:
        return None
    vwap = sum(c[i] * v[i] for i in range(period)) / total_vol
    return (closes[-1] - vwap) / closes[-1]


def _calc_roc(closes: List[float], period: int) -> Optional[float]:
    """Rate of change."""
    if len(closes) < period + 1:
        return None
    prev = closes[-(period + 1)]
    if prev == 0:
        return None
    return (closes[-1] - prev) / prev * 100


def _calc_momentum_divergence(closes: List[float]) -> Optional[float]:
    """Momentum divergence: difference between short-term and long-term momentum."""
    roc_short = _calc_roc(closes, 6)
    roc_long = _calc_roc(closes, 24)
    if roc_short is None or roc_long is None:
        return None
    return roc_short - roc_long


def _calc_volatility_regime(highs: List[float], lows: List[float],
                            closes: List[float]) -> Optional[int]:
    """Volatility regime: 0=low, 1=normal, 2=high based on ATR percentile.

    Computes rolling ATR values and checks where current ATR sits relative
    to the distribution.
    """
    period = 14
    if len(closes) < period + 50:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period + 30:
        return None
    # Rolling ATR values
    atr_values = []
    atr = sum(trs[:period]) / period
    atr_values.append(atr)
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        atr_values.append(atr)
    if len(atr_values) < 10:
        return None
    current_atr = atr_values[-1]
    sorted_atrs = sorted(atr_values)
    rank = sum(1 for a in sorted_atrs if a <= current_atr)
    percentile = rank / len(sorted_atrs)
    if percentile < 0.25:
        return 0  # low vol
    elif percentile > 0.75:
        return 2  # high vol
    return 1  # normal


def _fetch_fear_greed() -> Optional[int]:
    """Fetch Fear & Greed index, cached for 1 hour."""
    global _fng_cache, _fng_cache_time
    if _fng_cache is not None and (time.time() - _fng_cache_time) < _FNG_CACHE_TTL:
        return _fng_cache.get("value")
    try:
        resp = requests.get("https://api.alternative.me/fng/",
                            params={"limit": 1}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            _fng_cache = {"value": int(data[0]["value"])}
            _fng_cache_time = time.time()
            return _fng_cache["value"]
    except Exception as e:
        logger.warning("Failed to fetch Fear & Greed for features: %s", e)
    return _fng_cache.get("value") if _fng_cache else None


def compute_features(ohlc: List[Dict[str, float]],
                     funding_rate: Optional[float] = None,
                     funding_rate_avg: Optional[float] = None,
                     funding_rate_trend: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Compute all features from OHLC data for XGBoost input.

    Args:
        ohlc: List of candle dicts with open/high/low/close/volume keys.
        funding_rate: Current funding rate (from funding_rate monitor).
        funding_rate_avg: 7-day average funding rate.
        funding_rate_trend: Funding rate trend (positive = rising).

    Returns:
        Flat dict of named features. None for unavailable features.
    """
    if not ohlc or len(ohlc) < 30:
        logger.debug("Not enough OHLC data for features (need 30, got %d)",
                      len(ohlc) if ohlc else 0)
        return {}

    closes = [c["close"] for c in ohlc]
    highs = [c["high"] for c in ohlc]
    lows = [c["low"] for c in ohlc]
    volumes = [c["volume"] for c in ohlc]

    features: Dict[str, Optional[float]] = {}

    # ── Price returns ────────────────────────────────────────────────
    features["return_1h"] = _returns(closes, 1)
    features["return_4h"] = _returns(closes, 4)
    features["return_24h"] = _returns(closes, 24)
    features["return_7d"] = _returns(closes, 168) if len(closes) > 168 else None
    features["log_return_1h"] = _log_returns(closes, 1)
    features["log_return_4h"] = _log_returns(closes, 4)
    features["log_return_24h"] = _log_returns(closes, 24)

    # ── Realized volatility ──────────────────────────────────────────
    features["rvol_6h"] = _realized_volatility(closes, 6)
    features["rvol_24h"] = _realized_volatility(closes, 24)
    features["rvol_72h"] = _realized_volatility(closes, 72) if len(closes) > 72 else None

    # ── Technical indicators ─────────────────────────────────────────
    features["rsi_14"] = calc_rsi(closes)
    macd = _calc_macd(closes)
    if macd:
        features["macd_line"] = macd["macd_line"]
        features["macd_signal"] = macd["macd_signal"]
        features["macd_histogram"] = macd["macd_histogram"]
    else:
        features["macd_line"] = None
        features["macd_signal"] = None
        features["macd_histogram"] = None

    features["bollinger_width"] = _calc_bollinger_width(closes)
    features["atr_ratio"] = _calc_atr_ratio(highs, lows, closes)
    features["obv_trend"] = _calc_obv_trend(closes, volumes)
    features["vwap_deviation"] = _calc_vwap_deviation(closes, volumes)

    # ── Momentum ─────────────────────────────────────────────────────
    features["roc_6"] = _calc_roc(closes, 6)
    features["roc_12"] = _calc_roc(closes, 12)
    features["roc_24"] = _calc_roc(closes, 24)
    features["momentum_divergence"] = _calc_momentum_divergence(closes)

    # ── Regime ───────────────────────────────────────────────────────
    features["adx"] = calc_adx(highs, lows, closes)
    features["volatility_regime"] = _calc_volatility_regime(highs, lows, closes)

    # ── Sentiment ────────────────────────────────────────────────────
    features["fear_greed"] = _fetch_fear_greed()

    # ── Funding rate ─────────────────────────────────────────────────
    features["funding_rate"] = funding_rate
    features["funding_rate_avg"] = funding_rate_avg
    features["funding_rate_trend"] = funding_rate_trend

    return features
