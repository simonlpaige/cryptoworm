"""
Tuner — takes analysis + market context and adjusts strategy parameters.

Rules:
  1. Never move a parameter outside its research-backed bounds
  2. Only adjust one parameter per strategy per cycle (avoid overfitting)
  3. Log every change with reason, before/after, and source
  4. If a strategy is healthy, don't touch it
  5. Small incremental changes only (max 20% of range per cycle)
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import config
from trainer.researcher import RESEARCH_PARAMS

logger = logging.getLogger("cryptobot.trainer.tuner")

TUNING_LOG = os.path.join(config.BOT_DIR, "trainer", "tuning_log.json")
OVERRIDES_FILE = os.path.join(config.BOT_DIR, "trainer", "param_overrides.json")
META_OVERRIDES_FILE = os.path.join(config.BOT_DIR, "trainer", "meta_overrides.json")

_DEFAULT_TUNING_STRENGTH = 0.10


def get_tuning_strength() -> float:
    """Read tuning_strength from meta_overrides.json (set by meta-learner).
    Falls back to 0.10 if file missing or key absent."""
    try:
        if os.path.exists(META_OVERRIDES_FILE):
            with open(META_OVERRIDES_FILE, "r") as f:
                overrides = json.load(f)
            return float(overrides.get("tuning_strength", _DEFAULT_TUNING_STRENGTH))
    except Exception:
        pass
    return _DEFAULT_TUNING_STRENGTH


def load_overrides() -> dict:
    """Load current parameter overrides."""
    if os.path.exists(OVERRIDES_FILE):
        with open(OVERRIDES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_overrides(overrides: dict):
    """Persist parameter overrides."""
    os.makedirs(os.path.dirname(OVERRIDES_FILE), exist_ok=True)
    with open(OVERRIDES_FILE, "w") as f:
        json.dump(overrides, f, indent=2)


def log_tuning(entry: dict):
    """Append a tuning event to the log."""
    os.makedirs(os.path.dirname(TUNING_LOG), exist_ok=True)
    log = []
    if os.path.exists(TUNING_LOG):
        with open(TUNING_LOG, "r") as f:
            log = json.load(f)
    log.append(entry)
    # Keep last 500 entries
    if len(log) > 500:
        log = log[-500:]
    with open(TUNING_LOG, "w") as f:
        json.dump(log, f, indent=2)


def get_current_value(strategy: str, param: str, overrides: dict) -> float:
    """Get current parameter value (override or default)."""
    if strategy in overrides and param in overrides[strategy]:
        return overrides[strategy][param]
    bounds = RESEARCH_PARAMS.get(strategy, {}).get(param)
    if bounds:
        return bounds["default"]
    return 0


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def compute_adjustment(
    strategy: str,
    param: str,
    direction: str,  # "increase" or "decrease"
    current: float,
    bounds: dict,
    strength: float = None,  # None = read from meta_overrides (default 0.10)
) -> Tuple[float, str]:
    """Compute a safe parameter adjustment. Returns (new_value, reason).
    strength defaults to meta_overrides.tuning_strength (meta-learner controlled).
    """
    if strength is None:
        strength = get_tuning_strength()
    param_range = bounds["max"] - bounds["min"]
    step = param_range * strength

    if direction == "increase":
        new_val = current + step
    else:
        new_val = current - step

    new_val = clamp(new_val, bounds["min"], bounds["max"])

    # Round nicely
    if isinstance(bounds["default"], int):
        new_val = int(round(new_val))
    else:
        new_val = round(new_val, 2)

    return new_val, f"{param}: {current} → {new_val} ({direction}, step={step:.2f})"


def generate_adjustments(analysis: dict, market_context: dict) -> List[dict]:
    """
    The brain: maps issues + market context to specific parameter changes.
    Returns list of adjustment dicts.

    Key design: market-context-driven adjustments fire even when there are
    no closed trades (status == "no_data"). Issue-based adjustments still
    require actual trade data.
    """
    adjustments = []
    overrides = load_overrides()
    recommendations = market_context.get("recommendations", [])
    vol = market_context.get("volatility", {})

    for strat_name, strat_analysis in analysis.get("strategies", {}).items():
        has_trade_data = strat_analysis.get("status") != "no_data"

        # Skip entirely only if: has trade data AND is healthy AND not overridden
        # (no_data strategies still get market-context adjustments below)
        if has_trade_data and strat_analysis.get("status") == "healthy" and strat_name not in overrides:
            continue

        bounds_map = RESEARCH_PARAMS.get(strat_name, {})
        if not bounds_map:
            continue

        issues = strat_analysis.get("issues", []) if has_trade_data else []

        # Only one adjustment per strategy per cycle
        adjustment = None

        # ── Issue-based adjustments (require actual trade data) ───────────
        for issue in issues:
            if adjustment:
                break

            # ── Stops too tight → widen stop loss ────────────────────────
            if "stops_too_tight" in issue:
                param = "stop_loss_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "increase", curr, bounds_map[param], 0.15)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Stops too tight ({strat_analysis.get('sl_hits', 0)} SL hits). {reason}",
                            "issue": issue,
                        }

            # ── Bad risk-reward → widen take profit ──────────────────────
            elif "bad_risk_reward" in issue:
                param = "take_profit_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "increase", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Bad R:R ({strat_analysis.get('risk_reward', 0):.2f}). {reason}",
                            "issue": issue,
                        }

            # ── Low win rate → tighten entry filters ─────────────────────
            elif "low_win_rate" in issue:
                # For EMA/MACD: raise ADX threshold (stricter trend filter)
                if strat_name == "ema_macd" and "adx_threshold" in bounds_map:
                    param = "adx_threshold"
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "increase", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Low win rate ({strat_analysis['win_rate']}%). Stricter trend filter. {reason}",
                            "issue": issue,
                        }
                # For sentiment: widen fear threshold (more selective)
                elif strat_name == "sentiment" and "fear_threshold" in bounds_map:
                    param = "fear_threshold"
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "decrease", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Low win rate. More extreme fear required. {reason}",
                            "issue": issue,
                        }

            # ── Shorts underperforming → tighten short entry RSI ─────────
            elif "shorts_underperforming" in issue:
                if strat_name == "ema_macd" and "rsi_short_high" in bounds_map:
                    param = "rsi_short_high"
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "decrease", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Shorts underperforming (win rate {strat_analysis['short_win_rate']}%). {reason}",
                            "issue": issue,
                        }

        # ── Market-context-driven adjustments ────────────────────────────
        # These fire regardless of trade data (including no_data strategies)
        if not adjustment and vol:
            if "widen_stops" in recommendations:
                param = "stop_loss_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "increase", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"High volatility regime (ATR={vol.get('atr_pct', '?')}%). {reason}",
                            "issue": "high_volatility",
                        }

            elif "tighten_stops" in recommendations:
                param = "stop_loss_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "decrease", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Low volatility regime (ATR={vol.get('atr_pct', '?')}%). {reason}",
                            "issue": "low_volatility",
                        }

            # ── Binance: Funding rate overleveraged long → tighten longs ──
            elif "funding_rate_overleveraged_long" in recommendations and strat_name == "sentiment":
                param = "fear_threshold"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "decrease", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Funding rate overleveraged long — require deeper fear for long entries. {reason}",
                            "issue": "funding_rate_overleveraged_long",
                        }

            # ── Binance: Funding rate overleveraged short → widen TP ──────
            elif "funding_rate_overleveraged_short" in recommendations and strat_name in ("sentiment", "ema_macd"):
                param = "take_profit_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "increase", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Funding rate negative (overleveraged short) — potential squeeze, let longs run. {reason}",
                            "issue": "funding_rate_overleveraged_short",
                        }

            # ── Binance: Extreme funding → widen stops (high vol signal) ──
            elif "funding_rate_extreme" in recommendations:
                param = "stop_loss_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "increase", curr, bounds_map[param], 0.12)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"Extreme funding rate — market unstable, widen stops. {reason}",
                            "issue": "funding_rate_extreme",
                        }

            # ── Binance: OI trend confirmation → widen TP ─────────────────
            elif "oi_trend_confirmation" in recommendations and strat_name in ("sentiment", "ema_macd"):
                param = "take_profit_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "increase", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"OI rising with price — new money entering, let winners run. {reason}",
                            "issue": "oi_trend_confirmation",
                        }

            # ── Binance: OI exhaustion → tighten TP ───────────────────────
            elif "oi_trend_exhaustion" in recommendations and strat_name in ("sentiment", "ema_macd"):
                param = "take_profit_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "decrease", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"OI shrinking — trend exhaustion, take profits faster. {reason}",
                            "issue": "oi_trend_exhaustion",
                        }

            # ── Binance: OI shrinking → tighten stops ─────────────────────
            elif "oi_shrinking" in recommendations:
                param = "stop_loss_pct"
                if param in bounds_map:
                    curr = get_current_value(strat_name, param, overrides)
                    new_val, reason = compute_adjustment(
                        strat_name, param, "decrease", curr, bounds_map[param], 0.1)
                    if new_val != curr:
                        adjustment = {
                            "strategy": strat_name,
                            "param": param,
                            "old_value": curr,
                            "new_value": new_val,
                            "reason": f"OI falling — positions closing, tighten stops to protect gains. {reason}",
                            "issue": "oi_shrinking",
                        }

        if adjustment:
            adjustments.append(adjustment)

    return adjustments


def apply_adjustments(adjustments: List[dict]) -> dict:
    """Apply parameter adjustments and save. Returns summary."""
    overrides = load_overrides()
    applied = []

    for adj in adjustments:
        strat = adj["strategy"]
        param = adj["param"]
        if strat not in overrides:
            overrides[strat] = {}
        overrides[strat][param] = adj["new_value"]

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": strat,
            "param": param,
            "old_value": adj["old_value"],
            "new_value": adj["new_value"],
            "reason": adj["reason"],
            "issue": adj.get("issue", ""),
        }
        log_tuning(log_entry)
        applied.append(log_entry)

        logger.info("TUNED %s.%s: %s → %s (%s)",
                     strat, param, adj["old_value"], adj["new_value"], adj["reason"])

    save_overrides(overrides)
    return {"applied": applied, "total": len(applied)}
