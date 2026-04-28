"""
Researcher — fetches fresh intel from external sources and maps it
to actionable parameter adjustments.

Sources:
  1. Fear & Greed Index (current market regime)
  2. Kraken OHLC (volatility regime detection)
  3. Binance Futures: Funding Rate, Open Interest, Long/Short Ratio
     (free, no auth required)

Each source returns a "market_context" dict that the Tuner uses
to decide which parameters to adjust.
"""
import logging
import math
import requests
from typing import Optional, Dict
from datetime import datetime

import config

logger = logging.getLogger("cryptobot.trainer.researcher")


def fetch_fear_greed() -> Optional[dict]:
    """Current Fear & Greed Index + recent trend."""
    try:
        resp = requests.get("https://api.alternative.me/fng/", params={"limit": 7}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        current = int(data[0]["value"])
        week_avg = sum(int(d["value"]) for d in data) / len(data)
        trend = "rising" if current > week_avg else "falling" if current < week_avg else "flat"
        return {
            "current": current,
            "classification": data[0]["value_classification"],
            "week_avg": round(week_avg, 1),
            "trend": trend,
        }
    except Exception as e:
        logger.error("Failed to fetch Fear & Greed: %s", e)
        return None


def detect_volatility_regime(ohlc_data: list) -> Optional[dict]:
    """Detect current volatility regime from OHLC data.
    
    Returns 'high', 'normal', or 'low' volatility + metrics.
    Uses ATR and Bollinger Band width as signals.
    """
    if not ohlc_data or len(ohlc_data) < 20:
        return None

    closes = [c["close"] for c in ohlc_data]
    highs = [c["high"] for c in ohlc_data]
    lows = [c["low"] for c in ohlc_data]

    # ATR (14-period)
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    atr_14 = sum(trs[-14:]) / 14 if len(trs) >= 14 else sum(trs) / len(trs)
    atr_pct = (atr_14 / closes[-1]) * 100

    # Bollinger bandwidth
    recent_20 = closes[-20:]
    sma = sum(recent_20) / 20
    std = math.sqrt(sum((x - sma) ** 2 for x in recent_20) / 20)
    bb_width = (2 * 2 * std) / sma * 100

    # Classify
    if atr_pct > 3.5:
        regime = "high"
    elif atr_pct < 1.5:
        regime = "low"
    else:
        regime = "normal"

    # Trend detection (simple: is price above or below 20 SMA)
    trend = "bullish" if closes[-1] > sma else "bearish"

    return {
        "atr_pct": round(atr_pct, 2),
        "bb_width_pct": round(bb_width, 2),
        "regime": regime,
        "trend": trend,
        "current_price": closes[-1],
        "sma_20": round(sma, 2),
    }


def fetch_funding_rate() -> Optional[dict]:
    """Fetch BTC perpetual futures funding rate from Binance (free, no auth).
    
    Funding rate interpretation:
    - Positive (>0.03%): longs pay shorts → market overleveraged long → reversal risk
    - Negative (<-0.01%): shorts pay longs → potential short squeeze
    - Extreme either way → volatile, widen stops
    
    Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=10
    """
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": "BTCUSDT", "limit": 10},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None

        rates = [float(d["fundingRate"]) for d in data]
        current_rate = rates[0]
        avg_rate = sum(rates) / len(rates)

        # Determine signal
        if current_rate > 0.0003:  # >0.03%
            signal = "overleveraged_long"
        elif current_rate < -0.0001:  # <-0.01%
            signal = "overleveraged_short"
        else:
            signal = "neutral"

        # Detect extreme (either direction > 2x typical)
        is_extreme = abs(current_rate) > 0.0005

        return {
            "current_rate": round(current_rate * 100, 5),  # as percentage
            "avg_rate_10": round(avg_rate * 100, 5),
            "signal": signal,
            "is_extreme": is_extreme,
            "source": "binance_futures_funding_rate",
        }
    except Exception as e:
        logger.error("Failed to fetch funding rate: %s", e)
        return None


def fetch_derivatives_sentiment() -> Optional[dict]:
    """Fetch OI and Long/Short ratio from Binance Futures (free, no auth).
    
    Open Interest:
    - Rising OI + rising price = trend confirmation (new money entering)
    - Rising OI + falling price = shorts piling in
    - Falling OI = positions closing (trend exhaustion)
    
    Long/Short Ratio (contrarian):
    - Extreme longs (>0.7) → bearish signal
    - Extreme shorts (<0.4) → bullish signal
    
    Endpoints:
    - GET https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT
    - GET https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=4h&limit=10
    - GET https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=4h&limit=10
    """
    result = {
        "open_interest": None,
        "long_short_ratio": None,
        "top_trader_ratio": None,
        "signal": "neutral",
        "source": "binance_futures_derivatives",
    }

    # ── Open Interest ─────────────────────────────────────────────────────
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": "BTCUSDT"},
            timeout=10,
        )
        resp.raise_for_status()
        oi_data = resp.json()
        current_oi = float(oi_data.get("openInterest", 0))
        result["open_interest"] = {
            "value": round(current_oi, 2),
            "symbol": oi_data.get("symbol"),
        }
    except Exception as e:
        logger.error("Failed to fetch open interest: %s", e)

    # ── Global Long/Short Ratio ───────────────────────────────────────────
    try:
        resp = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "4h", "limit": 10},
            timeout=10,
        )
        resp.raise_for_status()
        ls_data = resp.json()
        if ls_data:
            current_ls = float(ls_data[0].get("longShortRatio", 1.0))
            long_pct = float(ls_data[0].get("longAccount", 0.5))
            short_pct = float(ls_data[0].get("shortAccount", 0.5))

            # Calculate OI trend from recent history if we have it
            if len(ls_data) >= 3:
                # Check if long ratio is trending up or down
                recent_ratios = [float(d.get("longShortRatio", 1.0)) for d in ls_data[:5]]
                ls_trend = "increasing" if recent_ratios[0] > recent_ratios[-1] else "decreasing"
            else:
                ls_trend = "stable"

            result["long_short_ratio"] = {
                "current": round(current_ls, 4),
                "long_pct": round(long_pct, 4),
                "short_pct": round(short_pct, 4),
                "trend": ls_trend,
            }
    except Exception as e:
        logger.error("Failed to fetch long/short ratio: %s", e)

    # ── Top Trader Position Ratio ─────────────────────────────────────────
    try:
        resp = requests.get(
            "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
            params={"symbol": "BTCUSDT", "period": "4h", "limit": 10},
            timeout=10,
        )
        resp.raise_for_status()
        top_data = resp.json()
        if top_data:
            top_ls = float(top_data[0].get("longShortRatio", 1.0))
            result["top_trader_ratio"] = {
                "current": round(top_ls, 4),
                "long_pct": round(float(top_data[0].get("longAccount", 0.5)), 4),
                "short_pct": round(float(top_data[0].get("shortAccount", 0.5)), 4),
            }
    except Exception as e:
        logger.error("Failed to fetch top trader ratio: %s", e)

    # ── Derive Combined Signal ────────────────────────────────────────────
    ls = result.get("long_short_ratio")
    top = result.get("top_trader_ratio")

    if ls:
        long_pct = ls["long_pct"]
        if long_pct > 0.70:
            # Extreme longs = contrarian bearish
            result["signal"] = "extreme_long_contrarian_bearish"
        elif long_pct < 0.40:
            # Extreme shorts = contrarian bullish
            result["signal"] = "extreme_short_contrarian_bullish"
        elif ls["trend"] == "increasing" and long_pct > 0.55:
            result["signal"] = "trend_confirmation_long"
        elif ls["trend"] == "decreasing" and long_pct < 0.50:
            result["signal"] = "trend_confirmation_short"
        else:
            result["signal"] = "neutral"

    return result


def build_market_context(ohlc_data: Optional[list] = None) -> dict:
    """Build complete market context from all sources."""
    context = {
        "timestamp": datetime.utcnow().isoformat(),
        "fear_greed": fetch_fear_greed(),
        "volatility": detect_volatility_regime(ohlc_data) if ohlc_data else None,
        "funding_rate": fetch_funding_rate() if getattr(config, 'ENABLE_BINANCE_RESEARCH', True) else None,
        "derivatives_sentiment": fetch_derivatives_sentiment() if getattr(config, 'ENABLE_BINANCE_RESEARCH', True) else None,
    }

    # Derive regime recommendation
    recommendations = []

    fg = context["fear_greed"]
    vol = context["volatility"]
    funding = context["funding_rate"]
    deriv = context["derivatives_sentiment"]

    if fg:
        if fg["current"] <= 25:
            recommendations.append("extreme_fear_bias_long")
        elif fg["current"] >= 75:
            recommendations.append("extreme_greed_bias_short")

    if vol:
        if vol["regime"] == "high":
            recommendations.append("widen_stops")
            recommendations.append("reduce_position_size")
            recommendations.append("prefer_momentum_strategies")
        elif vol["regime"] == "low":
            recommendations.append("tighten_stops")
            recommendations.append("prefer_mean_reversion")
            recommendations.append("increase_grid_density")

        if vol["trend"] == "bullish":
            recommendations.append("bullish_trend")
        else:
            recommendations.append("bearish_trend")

    # Binance Funding Rate signals
    if funding:
        if funding["signal"] == "overleveraged_long":
            recommendations.append("funding_rate_overleveraged_long")
        elif funding["signal"] == "overleveraged_short":
            recommendations.append("funding_rate_overleveraged_short")
        if funding.get("is_extreme"):
            recommendations.append("funding_rate_extreme")

    # Binance OI + Long/Short signals
    if deriv:
        sig = deriv["signal"]
        if sig == "extreme_long_contrarian_bearish":
            recommendations.append("oi_trend_exhaustion")
        elif sig == "extreme_short_contrarian_bullish":
            recommendations.append("oi_trend_confirmation")
        elif sig == "trend_confirmation_long":
            recommendations.append("oi_trend_confirmation")
        elif sig == "trend_confirmation_short":
            recommendations.append("oi_trend_exhaustion")

        # Shrinking OI signal (check if OI value looks low — heuristic)
        oi = deriv.get("open_interest")
        if oi and oi["value"] < 50000:  # rough threshold in BTC terms
            recommendations.append("oi_shrinking")

    context["recommendations"] = recommendations
    return context


# ── Research knowledge base for parameter tuning ─────────────────────────
# These are the research-backed parameter ranges from our sources.
# The tuner stays within these bounds.

RESEARCH_PARAMS = {
    "sentiment": {
        "fear_threshold": {"min": 15, "max": 30, "default": 25, "source": "alternative.me analysis"},
        "greed_threshold": {"min": 70, "max": 85, "default": 75, "source": "alternative.me analysis"},
        "take_profit_pct": {"min": 3.0, "max": 10.0, "default": 5.0, "source": "backtests"},
        "stop_loss_pct": {"min": 1.0, "max": 4.0, "default": 2.0, "source": "backtests"},
    },
    "ema_macd": {
        "rsi_long_low": {"min": 35, "max": 50, "default": 40, "source": "TrendRider"},
        "rsi_long_high": {"min": 60, "max": 75, "default": 70, "source": "TrendRider"},
        "rsi_short_low": {"min": 25, "max": 40, "default": 30, "source": "TrendRider"},
        "rsi_short_high": {"min": 55, "max": 65, "default": 60, "source": "TrendRider"},
        "adx_threshold": {"min": 20, "max": 35, "default": 25, "source": "TrendRider"},
        "stop_loss_pct": {"min": 1.2, "max": 3.0, "default": 1.8, "source": "TrendRider ATR-based"},
        "take_profit_pct": {"min": 2.0, "max": 5.0, "default": 3.2, "source": "TrendRider"},
    },
    "bollinger": {
        "bb_period": {"min": 15, "max": 30, "default": 20, "source": "standard"},
        "bb_std": {"min": 1.5, "max": 2.5, "default": 2.0, "source": "standard"},
        "rsi_oversold": {"min": 20, "max": 35, "default": 30, "source": "TrendRider"},
        "rsi_overbought": {"min": 65, "max": 80, "default": 70, "source": "TrendRider"},
        "adx_max": {"min": 25, "max": 40, "default": 30, "source": "TrendRider"},
    },
    "rsi_divergence": {
        "rsi_long_threshold": {"min": 30, "max": 50, "default": 40, "source": "Reddit consensus"},
        "rsi_short_threshold": {"min": 55, "max": 75, "default": 60, "source": "Reddit consensus"},
        "stop_loss_pct": {"min": 1.5, "max": 3.5, "default": 2.0, "source": "1:1.5 R:R"},
        "take_profit_pct": {"min": 2.5, "max": 5.0, "default": 3.0, "source": "1:1.5 R:R"},
    },
    "grid": {
        "grid_range_pct": {"min": 5.0, "max": 20.0, "default": 10.0, "source": "standard"},
        "grid_levels": {"min": 5, "max": 20, "default": 10, "source": "standard"},
    },
}
