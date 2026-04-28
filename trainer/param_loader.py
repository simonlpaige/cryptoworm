"""
Parameter Loader — reads tuner overrides and applies them to strategy instances.

Call this at bot startup and after each training cycle to hot-reload parameters.
Strategies check these values instead of hardcoded config constants.
"""
import json
import logging
import os

import config

logger = logging.getLogger("cryptobot.trainer.param_loader")

OVERRIDES_FILE = os.path.join(config.BOT_DIR, "trainer", "param_overrides.json")

_cache = {}
_cache_mtime = 0


def _load():
    global _cache, _cache_mtime
    if not os.path.exists(OVERRIDES_FILE):
        _cache = {}
        return
    mtime = os.path.getmtime(OVERRIDES_FILE)
    if mtime != _cache_mtime:
        with open(OVERRIDES_FILE, "r") as f:
            _cache = json.load(f)
        _cache_mtime = mtime
        logger.debug("Reloaded param overrides (mtime=%s)", mtime)


def get(strategy: str, param: str, default):
    """Get a parameter value. Returns override if set, else default."""
    _load()
    return _cache.get(strategy, {}).get(param, default)


# ── Convenience accessors ────────────────────────────────────────────────

# Sentiment
def fear_threshold():
    return get("sentiment", "fear_threshold", config.FEAR_THRESHOLD)

def greed_threshold():
    return get("sentiment", "greed_threshold", config.GREED_THRESHOLD)

def sentiment_tp():
    return get("sentiment", "take_profit_pct", config.SWING_TAKE_PROFIT_PCT)

def sentiment_sl():
    return get("sentiment", "stop_loss_pct", config.SWING_STOP_LOSS_PCT)

# EMA/MACD
def ema_adx_threshold():
    return get("ema_macd", "adx_threshold", 25)

def ema_rsi_long_range():
    low = get("ema_macd", "rsi_long_low", 40)
    high = get("ema_macd", "rsi_long_high", 70)
    return (low, high)

def ema_rsi_short_range():
    low = get("ema_macd", "rsi_short_low", 30)
    high = get("ema_macd", "rsi_short_high", 60)
    return (low, high)

def ema_sl():
    return get("ema_macd", "stop_loss_pct", 1.8)

def ema_tp():
    return get("ema_macd", "take_profit_pct", 3.2)

# Bollinger
def bb_period():
    return int(get("bollinger", "bb_period", 20))

def bb_std():
    return get("bollinger", "bb_std", 2.0)

def bb_rsi_oversold():
    return get("bollinger", "rsi_oversold", 30)

def bb_rsi_overbought():
    return get("bollinger", "rsi_overbought", 70)

def bb_adx_max():
    return get("bollinger", "adx_max", 30)

# RSI Divergence
def rsi_div_long_threshold():
    return get("rsi_divergence", "rsi_long_threshold", 40)

def rsi_div_short_threshold():
    return get("rsi_divergence", "rsi_short_threshold", 60)

def rsi_div_sl():
    return get("rsi_divergence", "stop_loss_pct", 2.0)

def rsi_div_tp():
    return get("rsi_divergence", "take_profit_pct", 3.0)

# Grid
def grid_range():
    return get("grid", "grid_range_pct", config.GRID_RANGE_PCT)

def grid_levels():
    return int(get("grid", "grid_levels", config.GRID_LEVELS))
