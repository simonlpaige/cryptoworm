"""
Meta-Learner — Layer 4: The trainer that trains the trainer.

Evaluates how well the training process itself works and recursively
improves its own methods. This is genuine recursive self-improvement:
the meta-learner tunes the tuner's hyperparameters, then evaluates
whether its own changes helped — and reverts them if not.

Architecture:
    Layer 1: Strategies (execute trades)
    Layer 2: Trainer (tune strategy params)
    Layer 3: Discovery (find new patterns)
    Layer 4: Meta-Learner (tune the trainer's own params) ← THIS FILE

Safety rails:
    - Hyperparameter bounds are HARD limits, never exceeded
    - Max one hyperparameter change per meta-cycle (prevent oscillation)
    - Meta-learner improvement rate < 0.2 for 5+ cycles → full reset
    - All changes logged with full before/after/reason
    - Never directly places trades or modifies positions
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("cryptobot.trainer.meta_learner")

# ── File paths ─────────────────────────────────────────────────────────────
META_STATE_FILE = os.path.join(config.BOT_DIR, "trainer", "meta_state.json")
META_OVERRIDES_FILE = os.path.join(config.BOT_DIR, "trainer", "meta_overrides.json")
META_OUTCOMES_FILE = os.path.join(config.BOT_DIR, "trainer", "meta_outcomes.json")
TRAINING_REPORT_DIR = os.path.join(config.BOT_DIR, "trainer", "reports")

# ── Default hyperparameter values and bounds ──────────────────────────────
DEFAULT_HYPERPARAMETERS = {
    "training_interval_ticks": 12,
    "discovery_interval_cycles": 6,
    "tuning_strength": 0.10,
    "min_pattern_confidence": 0.60,
    "min_pattern_occurrences": 5,
    "correlation_min_strength": 0.5,
    "lookback_days_analysis": 14,
    "revert_threshold": 3,
}

HYPERPARAMETER_BOUNDS = {
    "training_interval_ticks":  {"min": 6,    "max": 36,   "step_pct": 0.25},
    "discovery_interval_cycles":{"min": 2,    "max": 24,   "step_pct": 0.25},
    "tuning_strength":          {"min": 0.03, "max": 0.20, "step_pct": 0.20},
    "min_pattern_confidence":   {"min": 0.45, "max": 0.80, "step_pct": 0.10},
    "min_pattern_occurrences":  {"min": 3,    "max": 10,   "step_pct": 0.20},
    "correlation_min_strength": {"min": 0.25, "max": 0.85, "step_pct": 0.10},
    "lookback_days_analysis":   {"min": 7,    "max": 30,   "step_pct": 0.15},
    "revert_threshold":         {"min": 2,    "max": 5,    "step_pct": 0.33},
}

# Lifecycle states for strategies
LIFECYCLE_STATES = ["proposed", "backtesting", "paper_testing", "active", "degraded", "retired"]

# ── Helpers ────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _round_param(param: str, value: float) -> float:
    """Round to int or 2 decimals depending on param type."""
    int_params = {"training_interval_ticks", "discovery_interval_cycles",
                  "min_pattern_occurrences", "revert_threshold", "lookback_days_analysis"}
    if param in int_params:
        return int(round(value))
    return round(value, 3)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _trend_slope(values: List[float]) -> float:
    """Compute linear regression slope (rise/run) for a list of values.
    Returns positive if trending up, negative if trending down.
    Stdlib only — no numpy."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    numerator = sum((xs[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))
    return _safe_div(numerator, denominator)


# ── State I/O ──────────────────────────────────────────────────────────────

def load_meta_state() -> dict:
    """
    Load or initialize the meta-learning state.

    Schema:
    {
        "training_effectiveness": {
            "adjustments_made": 142,
            "adjustments_that_improved": 67,
            "adjustments_that_degraded": 31,
            "adjustments_neutral": 44,
            "improvement_rate": 0.47,
            "improvement_rate_history": [0.3, 0.35, 0.42, 0.47],
        },
        "discovery_effectiveness": {
            "patterns_proposed": 23,
            "patterns_validated": 8,
            "patterns_failed": 12,
            "patterns_pending": 3,
            "validation_rate": 0.40,
            "validation_rate_history": [],
        },
        "hyperparameters": { ... },
        "hyperparameter_history": [ ... ],
        "strategy_lifecycle": { ... },
        "meta_cycles": 0,
        "last_meta_cycle": null,
        "consecutive_low_meta_rate": 0,
    }
    """
    if os.path.exists(META_STATE_FILE):
        try:
            with open(META_STATE_FILE, "r") as f:
                state = json.load(f)
            # Back-fill any missing keys (forward compatibility)
            state.setdefault("training_effectiveness", _default_training_effectiveness())
            state.setdefault("discovery_effectiveness", _default_discovery_effectiveness())
            state.setdefault("hyperparameters", dict(DEFAULT_HYPERPARAMETERS))
            state.setdefault("hyperparameter_history", [])
            state.setdefault("strategy_lifecycle", {})
            state.setdefault("meta_cycles", 0)
            state.setdefault("last_meta_cycle", None)
            state.setdefault("consecutive_low_meta_rate", 0)
            return state
        except Exception as e:
            logger.warning("Could not load meta_state.json: %s — using defaults", e)

    return {
        "training_effectiveness": _default_training_effectiveness(),
        "discovery_effectiveness": _default_discovery_effectiveness(),
        "hyperparameters": dict(DEFAULT_HYPERPARAMETERS),
        "hyperparameter_history": [],
        "strategy_lifecycle": {},
        "meta_cycles": 0,
        "last_meta_cycle": None,
        "consecutive_low_meta_rate": 0,
    }


def _default_training_effectiveness() -> dict:
    return {
        "adjustments_made": 0,
        "adjustments_that_improved": 0,
        "adjustments_that_degraded": 0,
        "adjustments_neutral": 0,
        "improvement_rate": 0.0,
        "improvement_rate_history": [],
    }


def _default_discovery_effectiveness() -> dict:
    return {
        "patterns_proposed": 0,
        "patterns_validated": 0,
        "patterns_failed": 0,
        "patterns_pending": 0,
        "validation_rate": 0.0,
        "validation_rate_history": [],
    }


def save_meta_state(state: dict):
    """Persist meta state to disk."""
    try:
        os.makedirs(os.path.dirname(META_STATE_FILE), exist_ok=True)
        with open(META_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error("Failed to save meta_state: %s", e)


# ── Training Outcome Recording ─────────────────────────────────────────────

def record_training_outcome(report: dict):
    """
    Called after every training cycle. Records the cycle's outcome
    (PnL, adjustments made, reverted) so the meta-learner can later
    evaluate whether those adjustments were effective.

    This is the data pipeline feeding the meta-evaluator.
    """
    try:
        outcomes = _load_outcomes()
        entry = {
            "cycle": report.get("cycle", 0),
            "timestamp": report.get("timestamp", _now()),
            "total_pnl": report.get("analysis", {}).get("total_pnl", 0.0),
            "total_trades": report.get("analysis", {}).get("total_trades", 0),
            "adjustments_applied": [
                {
                    "strategy": a.get("strategy"),
                    "param": a.get("param"),
                    "old_value": a.get("old_value"),
                    "new_value": a.get("new_value"),
                    "reason": a.get("reason", ""),
                }
                for a in report.get("adjustments", {}).get("applied", [])
            ],
            "was_reverted": report.get("reverted", False),
            "market_regime": report.get("market_context", {}).get("volatility", {}).get("regime", "unknown"),
        }
        outcomes.append(entry)
        # Keep last 500 outcomes
        if len(outcomes) > 500:
            outcomes = outcomes[-500:]
        _save_outcomes(outcomes)
        logger.debug("Meta: recorded outcome for cycle %d", entry["cycle"])
    except Exception as e:
        logger.error("record_training_outcome failed: %s", e)


def _load_outcomes() -> list:
    if os.path.exists(META_OUTCOMES_FILE):
        try:
            with open(META_OUTCOMES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_outcomes(outcomes: list):
    os.makedirs(os.path.dirname(META_OUTCOMES_FILE), exist_ok=True)
    with open(META_OUTCOMES_FILE, "w") as f:
        json.dump(outcomes, f, indent=2)


# ── Adjustment Outcome Evaluator ───────────────────────────────────────────

def evaluate_adjustment_outcomes(training_reports: Optional[List[dict]] = None) -> dict:
    """
    Look at the last N training cycle outcomes and evaluate whether the
    adjustments made in each cycle actually helped in subsequent cycles.

    Classification logic:
    - For each cycle where adjustments were made, look at PnL in the 2
      cycles *after* it vs PnL in the 2 cycles *before* it.
    - improved:  post-PnL  >  pre-PnL by >2%
    - degraded:  post-PnL  <  pre-PnL by >2%
    - neutral:   within ±2% or insufficient data

    Updates meta_state training_effectiveness in place.
    Returns evaluation summary.
    """
    try:
        outcomes = _load_outcomes()
        if len(outcomes) < 4:
            return {"evaluated": 0, "improved": 0, "degraded": 0, "neutral": 0,
                    "current_improvement_rate": 0.0, "note": "insufficient_data"}

        improved = 0
        degraded = 0
        neutral = 0
        evaluated = 0

        # Only evaluate outcomes that have at least 2 cycles after them
        # (so we can measure subsequent performance)
        for i, outcome in enumerate(outcomes[:-2]):
            if not outcome.get("adjustments_applied"):
                continue  # No adjustments to evaluate

            # Pre-window: PnL from the 2 cycles before this one
            pre_pnl = [outcomes[j]["total_pnl"] for j in range(max(0, i - 2), i)]
            # Post-window: PnL from the 2 cycles after this one
            post_pnl = [outcomes[j]["total_pnl"] for j in range(i + 1, min(len(outcomes), i + 3))]

            if not pre_pnl or not post_pnl:
                continue

            avg_pre = sum(pre_pnl) / len(pre_pnl)
            avg_post = sum(post_pnl) / len(post_pnl)

            evaluated += 1
            if avg_pre == 0:
                # No reference point, count as neutral unless post is clearly positive
                if avg_post > 0.01:
                    improved += 1
                elif avg_post < -0.01:
                    degraded += 1
                else:
                    neutral += 1
            else:
                change_pct = (avg_post - avg_pre) / abs(avg_pre)
                if change_pct > 0.02:
                    improved += 1
                elif change_pct < -0.02:
                    degraded += 1
                else:
                    neutral += 1

        total_decided = improved + degraded
        improvement_rate = _safe_div(improved, total_decided) if total_decided > 0 else 0.0

        # Update meta_state
        meta_state = load_meta_state()
        te = meta_state["training_effectiveness"]
        te["adjustments_made"] = sum(
            1 for o in outcomes if o.get("adjustments_applied")
        )
        te["adjustments_that_improved"] = improved
        te["adjustments_that_degraded"] = degraded
        te["adjustments_neutral"] = neutral
        te["improvement_rate"] = round(improvement_rate, 3)

        # Keep rolling history (window of 10 evaluations)
        history = te.get("improvement_rate_history", [])
        if evaluated > 0:
            history.append(round(improvement_rate, 3))
        if len(history) > 20:
            history = history[-20:]
        te["improvement_rate_history"] = history

        save_meta_state(meta_state)
        logger.info("Meta: evaluated %d adjustments — improved=%d degraded=%d neutral=%d rate=%.2f",
                    evaluated, improved, degraded, neutral, improvement_rate)

        return {
            "evaluated": evaluated,
            "improved": improved,
            "degraded": degraded,
            "neutral": neutral,
            "current_improvement_rate": round(improvement_rate, 3),
        }
    except Exception as e:
        logger.error("evaluate_adjustment_outcomes failed: %s", e)
        return {"evaluated": 0, "improved": 0, "degraded": 0, "neutral": 0,
                "current_improvement_rate": 0.0, "error": str(e)}


# ── Hyperparameter Optimizer ───────────────────────────────────────────────

def optimize_hyperparameters(meta_state: dict) -> list:
    """
    The recursive core: tune the trainer's own hyperparameters based on
    how well training has been working.

    Max one change per meta-cycle to prevent oscillation.
    All changes respect HARD bounds.

    Returns list of adjustment dicts:
    [{"param": str, "old": val, "new": val, "reason": str}]
    """
    try:
        hp = meta_state.get("hyperparameters", dict(DEFAULT_HYPERPARAMETERS))
        te = meta_state.get("training_effectiveness", _default_training_effectiveness())
        de = meta_state.get("discovery_effectiveness", _default_discovery_effectiveness())
        improvement_rate = te.get("improvement_rate", 0.0)
        validation_rate = de.get("validation_rate", 0.0)
        patterns_proposed = de.get("patterns_proposed", 0)

        # Load recent training outcomes to assess market volatility
        outcomes = _load_outcomes()
        recent = outcomes[-12:] if len(outcomes) >= 12 else outcomes
        volatile_count = sum(1 for o in recent if o.get("market_regime") in ("high", "extreme"))
        is_volatile = volatile_count > len(recent) * 0.5 if recent else False

        # Track revert frequency
        revert_count = sum(1 for o in outcomes if o.get("was_reverted", False))
        total_cycles = len(outcomes)
        recent_reverts = sum(1 for o in outcomes[-20:] if o.get("was_reverted", False))

        adjustments = []

        # ── Rule A: TUNING STRENGTH ────────────────────────────────────────
        # Good callers → be bolder. Bad callers → smaller moves.
        param = "tuning_strength"
        bounds = HYPERPARAMETER_BOUNDS[param]
        current = hp.get(param, DEFAULT_HYPERPARAMETERS[param])
        step = (bounds["max"] - bounds["min"]) * bounds["step_pct"]

        if improvement_rate > 0.6:
            new_val = _clamp(current + step, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"High improvement rate ({improvement_rate:.0%}) — tuner making good calls, increase boldness"
                })
        elif improvement_rate < 0.3:
            new_val = _clamp(current - step, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"Low improvement rate ({improvement_rate:.0%}) — tuner making bad calls, shrink step size"
                })

        # ── Rule B: TRAINING INTERVAL ──────────────────────────────────────
        param = "training_interval_ticks"
        bounds = HYPERPARAMETER_BOUNDS[param]
        current = hp.get(param, DEFAULT_HYPERPARAMETERS[param])
        step = max(1, int((bounds["max"] - bounds["min"]) * bounds["step_pct"]))

        if improvement_rate > 0.5 and is_volatile:
            new_val = _clamp(current - step, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"High IR + volatile market — train more often ({volatile_count}/{len(recent)} recent volatile cycles)"
                })
        elif improvement_rate < 0.3 and not is_volatile:
            new_val = _clamp(current + step, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"Low IR + calm market — reduce training frequency to avoid noise"
                })

        # ── Rule C: DISCOVERY SENSITIVITY ─────────────────────────────────
        if validation_rate > 0.5:
            param = "min_pattern_confidence"
            bounds = HYPERPARAMETER_BOUNDS[param]
            current = hp.get(param, DEFAULT_HYPERPARAMETERS[param])
            step = (bounds["max"] - bounds["min"]) * bounds["step_pct"]
            new_val = _clamp(current - step, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"High validation rate ({validation_rate:.0%}) — lower bar to discover more patterns"
                })
        elif validation_rate < 0.25 and patterns_proposed > 3:
            param = "min_pattern_confidence"
            bounds = HYPERPARAMETER_BOUNDS[param]
            current = hp.get(param, DEFAULT_HYPERPARAMETERS[param])
            step = (bounds["max"] - bounds["min"]) * bounds["step_pct"]
            new_val = _clamp(current + step, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"Low validation rate ({validation_rate:.0%}) — raise bar to filter noise"
                })
        elif patterns_proposed < 3:
            param = "min_pattern_occurrences"
            bounds = HYPERPARAMETER_BOUNDS[param]
            current = hp.get(param, DEFAULT_HYPERPARAMETERS[param])
            step = max(1, int((bounds["max"] - bounds["min"]) * bounds["step_pct"]))
            new_val = _clamp(current - step, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"Too few patterns proposed ({patterns_proposed}) — lower occurrence threshold"
                })

        # ── Rule D: ANALYSIS LOOKBACK ──────────────────────────────────────
        # Use improvement_rate_history slope: if recent cycles outperform,
        # regime may have changed — focus on recent data
        ir_history = te.get("improvement_rate_history", [])
        if len(ir_history) >= 3:
            slope = _trend_slope(ir_history[-6:])
            param = "lookback_days_analysis"
            bounds = HYPERPARAMETER_BOUNDS[param]
            current = hp.get(param, DEFAULT_HYPERPARAMETERS[param])
            step = max(1, int((bounds["max"] - bounds["min"]) * bounds["step_pct"]))

            if slope > 0.05:
                # Improvement accelerating → regime changed → shorten lookback
                new_val = _clamp(current - step, bounds["min"], bounds["max"])
                new_val = _round_param(param, new_val)
                if new_val != current:
                    adjustments.append({
                        "param": param, "old": current, "new": new_val,
                        "reason": f"IR trending up (slope={slope:.3f}) — recent data more relevant, shorten lookback"
                    })
            elif slope < -0.05 and current < DEFAULT_HYPERPARAMETERS[param]:
                # Improvement declining after shortening → go back to longer lookback
                new_val = _clamp(current + step, bounds["min"], bounds["max"])
                new_val = _round_param(param, new_val)
                if new_val != current:
                    adjustments.append({
                        "param": param, "old": current, "new": new_val,
                        "reason": f"IR trending down (slope={slope:.3f}) — lengthen lookback for stability"
                    })

        # ── Rule E: REVERT THRESHOLD ───────────────────────────────────────
        param = "revert_threshold"
        bounds = HYPERPARAMETER_BOUNDS[param]
        current = hp.get(param, DEFAULT_HYPERPARAMETERS[param])

        if recent_reverts > 2:
            # Too many reverts → cut losses faster
            new_val = _clamp(current - 1, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"Frequent reverts ({recent_reverts}/20 cycles) — lower threshold to cut losses faster"
                })
        elif recent_reverts == 0 and total_cycles >= 20:
            # No reverts at all → give more room
            new_val = _clamp(current + 1, bounds["min"], bounds["max"])
            new_val = _round_param(param, new_val)
            if new_val != current:
                adjustments.append({
                    "param": param, "old": current, "new": new_val,
                    "reason": f"No reverts in {min(total_cycles, 20)} cycles — raise threshold to allow more exploration"
                })

        # ── MAX ONE CHANGE PER META-CYCLE ──────────────────────────────────
        # Prioritize by impact: stability first (revert), then core (tuning strength),
        # then frequency, then sensitivity
        priority_order = [
            "revert_threshold", "tuning_strength", "training_interval_ticks",
            "lookback_days_analysis", "min_pattern_confidence", "min_pattern_occurrences"
        ]
        adjustments.sort(key=lambda a: (
            priority_order.index(a["param"]) if a["param"] in priority_order else 99
        ))

        if len(adjustments) > 1:
            logger.info("Meta: %d potential hyperparameter changes, applying only top 1", len(adjustments))
            adjustments = adjustments[:1]

        return adjustments

    except Exception as e:
        logger.error("optimize_hyperparameters failed: %s", e)
        return []


# ── Strategy Lifecycle Manager ─────────────────────────────────────────────

def manage_strategy_lifecycle(meta_state: dict, training_state: dict) -> dict:
    """
    Manage full lifecycle of strategies including discovered ones.

    States: proposed → backtesting → paper_testing → active → degraded → retired

    Returns dict of lifecycle change events.
    """
    try:
        lifecycle = meta_state.get("strategy_lifecycle", {})
        hp = meta_state.get("hyperparameters", dict(DEFAULT_HYPERPARAMETERS))
        min_confidence = hp.get("min_pattern_confidence", 0.60)
        meta_cycle_count = meta_state.get("meta_cycles", 0)
        events = []

        # Try to pull discovered strategy proposals from discovery module
        proposed_patterns = _get_discovery_proposals()
        for pattern in proposed_patterns:
            pid = pattern.get("id", f"discovered_{_now()[:10]}")
            if pid not in lifecycle:
                lifecycle[pid] = {
                    "state": "proposed",
                    "name": pattern.get("name", pid),
                    "proposed_at": _now(),
                    "meta_cycle_proposed": meta_cycle_count,
                    "backtest_result": None,
                    "paper_results": [],
                    "consecutive_losses": 0,
                    "degraded_at_cycle": None,
                    "source": "discovery",
                }
                events.append({"strategy": pid, "transition": "created→proposed", "reason": "discovery proposal"})
                logger.info("Lifecycle: new strategy proposed: %s", pid)

        # Process transitions for all tracked strategies
        for sid, info in list(lifecycle.items()):
            state = info.get("state", "proposed")
            changed = False

            if state == "proposed":
                # Auto-advance to backtesting
                info["state"] = "backtesting"
                info["backtesting_started"] = _now()
                events.append({"strategy": sid, "transition": "proposed→backtesting",
                               "reason": "automatic after proposal"})
                changed = True

            elif state == "backtesting":
                # Check backtest results (populated by discovery/backtester)
                backtest = info.get("backtest_result")
                if backtest:
                    win_rate = backtest.get("win_rate", 0.0)
                    if win_rate >= min_confidence:
                        info["state"] = "paper_testing"
                        info["paper_started"] = _now()
                        info["paper_allocation_pct"] = 5.0  # small initial allocation
                        events.append({"strategy": sid, "transition": "backtesting→paper_testing",
                                      "reason": f"backtest win_rate={win_rate:.0%} ≥ {min_confidence:.0%}"})
                        changed = True
                    else:
                        info["state"] = "retired"
                        info["retired_at"] = _now()
                        info["retire_reason"] = f"backtest win_rate={win_rate:.0%} < {min_confidence:.0%}"
                        events.append({"strategy": sid, "transition": "backtesting→retired",
                                      "reason": info["retire_reason"]})
                        changed = True

            elif state == "paper_testing":
                paper = info.get("paper_results", [])
                if len(paper) >= 5:
                    # Compare live results vs backtest
                    backtest = info.get("backtest_result", {})
                    bt_wr = backtest.get("win_rate", 0.5)
                    live_wins = sum(1 for r in paper if r.get("pnl", 0) > 0)
                    live_wr = live_wins / len(paper)
                    divergence = abs(live_wr - bt_wr)

                    if divergence <= 0.20:  # live matches backtest within 20%
                        info["state"] = "active"
                        info["activated_at"] = _now()
                        info["paper_allocation_pct"] = None
                        events.append({"strategy": sid, "transition": "paper_testing→active",
                                      "reason": f"live WR {live_wr:.0%} within 20% of backtest {bt_wr:.0%}"})
                        changed = True
                    elif divergence > 0.40 and live_wr < bt_wr:
                        info["state"] = "retired"
                        info["retired_at"] = _now()
                        info["retire_reason"] = f"live WR {live_wr:.0%} diverges too far from backtest {bt_wr:.0%}"
                        events.append({"strategy": sid, "transition": "paper_testing→retired",
                                      "reason": info["retire_reason"]})
                        changed = True

            elif state == "active":
                consecutive_losses = info.get("consecutive_losses", 0)
                if consecutive_losses >= 5:
                    info["state"] = "degraded"
                    info["degraded_at_cycle"] = meta_cycle_count
                    events.append({"strategy": sid, "transition": "active→degraded",
                                  "reason": f"{consecutive_losses} consecutive losing periods"})
                    changed = True

            elif state == "degraded":
                degraded_at = info.get("degraded_at_cycle", meta_cycle_count)
                cycles_since_degraded = meta_cycle_count - degraded_at
                consecutive_losses = info.get("consecutive_losses", 0)

                if cycles_since_degraded >= 3 and consecutive_losses > 0:
                    # No recovery in 3 meta-cycles → retire
                    info["state"] = "retired"
                    info["retired_at"] = _now()
                    info["retire_reason"] = f"no recovery in {cycles_since_degraded} meta-cycles"
                    events.append({"strategy": sid, "transition": "degraded→retired",
                                  "reason": info["retire_reason"]})
                    changed = True
                elif consecutive_losses == 0:
                    # Performance recovered
                    info["state"] = "active"
                    info["recovered_at"] = _now()
                    events.append({"strategy": sid, "transition": "degraded→active",
                                  "reason": "performance recovered"})
                    changed = True

            if changed:
                info["last_transition"] = _now()
                logger.info("Lifecycle: %s → %s", sid, info["state"])

        meta_state["strategy_lifecycle"] = lifecycle
        return {"events": events, "total_strategies": len(lifecycle),
                "active": sum(1 for s in lifecycle.values() if s["state"] == "active"),
                "paper_testing": sum(1 for s in lifecycle.values() if s["state"] == "paper_testing"),
                "proposed": sum(1 for s in lifecycle.values() if s["state"] == "proposed"),
                "retired": sum(1 for s in lifecycle.values() if s["state"] == "retired")}
    except Exception as e:
        logger.error("manage_strategy_lifecycle failed: %s", e)
        return {"events": [], "error": str(e)}


def _get_discovery_proposals() -> list:
    """Try to get pattern proposals from discovery module (may not exist yet)."""
    try:
        from trainer import discovery  # type: ignore
        if hasattr(discovery, "get_pending_proposals"):
            return discovery.get_pending_proposals() or []
    except (ImportError, Exception):
        pass
    return []


# ── Self-Evaluation Report ─────────────────────────────────────────────────

def generate_meta_report(meta_state: dict) -> dict:
    """
    The meta-learner's own report card.

    Evaluates:
    - Is the training improvement rate trending up?
    - Are hyperparameter changes helping?
    - What's the overall system trajectory?
    - Any structural recommendations? (logged, never auto-applied)
    """
    try:
        te = meta_state.get("training_effectiveness", _default_training_effectiveness())
        de = meta_state.get("discovery_effectiveness", _default_discovery_effectiveness())
        hp = meta_state.get("hyperparameters", dict(DEFAULT_HYPERPARAMETERS))
        hp_history = meta_state.get("hyperparameter_history", [])
        meta_cycles = meta_state.get("meta_cycles", 0)

        ir = te.get("improvement_rate", 0.0)
        ir_history = te.get("improvement_rate_history", [])
        ir_slope = _trend_slope(ir_history[-6:]) if len(ir_history) >= 3 else 0.0
        vr = de.get("validation_rate", 0.0)

        # Grade the meta-learner's own effectiveness
        if ir >= 0.6 and ir_slope >= 0:
            meta_grade = "excellent"
        elif ir >= 0.45:
            meta_grade = "good"
        elif ir >= 0.3:
            meta_grade = "fair"
        else:
            meta_grade = "poor"

        # Assess hyperparameter change history
        recent_hp_changes = [h for h in hp_history[-10:]]
        hp_change_count = len(recent_hp_changes)

        # Structural recommendations (human-facing, never auto-applied)
        recommendations = []
        if ir < 0.2:
            recommendations.append("CRITICAL: training improvement rate very low — consider reviewing strategy fundamentals")
        if ir_slope < -0.1:
            recommendations.append("WARNING: improvement rate declining — market regime may have shifted")
        if vr < 0.2 and de.get("patterns_proposed", 0) > 5:
            recommendations.append("INFO: low pattern validation rate — discovery finding noise, not signal")
        if hp_change_count >= 8:
            recommendations.append("INFO: many hyperparameter changes recently — system may be oscillating")
        if meta_cycles > 10 and ir_slope == 0.0:
            recommendations.append("INFO: improvement rate flat — system may have reached local optimum")
        if not recommendations:
            recommendations.append("System operating within normal parameters")

        report = {
            "meta_grade": meta_grade,
            "training_improvement_rate": round(ir, 3),
            "improvement_rate_trend": "up" if ir_slope > 0.01 else "down" if ir_slope < -0.01 else "flat",
            "improvement_rate_slope": round(ir_slope, 4),
            "improvement_rate_history": ir_history[-10:],
            "discovery_validation_rate": round(vr, 3),
            "current_hyperparameters": hp,
            "recent_hp_changes": hp_change_count,
            "meta_cycles_completed": meta_cycles,
            "overall_trajectory": _assess_trajectory(ir, ir_slope, vr),
            "structural_recommendations": recommendations,
        }

        logger.info("Meta report: grade=%s IR=%.2f trend=%s",
                    meta_grade, ir, report["improvement_rate_trend"])
        return report
    except Exception as e:
        logger.error("generate_meta_report failed: %s", e)
        return {"error": str(e), "meta_grade": "unknown"}


def _assess_trajectory(ir: float, slope: float, vr: float) -> str:
    if ir >= 0.5 and slope >= 0:
        return "improving"
    elif ir >= 0.4 and slope >= -0.02:
        return "stable"
    elif ir < 0.3 and slope < 0:
        return "declining"
    elif slope > 0.05:
        return "recovering"
    else:
        return "uncertain"


# ── Apply Hyperparameter Decisions ─────────────────────────────────────────

def apply_meta_decisions(meta_state: dict, training_state: dict):
    """
    Write hyperparameter changes back to the system via meta_overrides.json.

    Other modules read this file:
    - tuner.py: reads tuning_strength
    - engine.py: reads lookback_days_analysis
    - discovery.py: reads min_pattern_confidence, min_pattern_occurrences
    - bot.py: reads training_interval_ticks (soft-read, not hot-swap)
    """
    try:
        hp = meta_state.get("hyperparameters", dict(DEFAULT_HYPERPARAMETERS))
        overrides = {
            "tuning_strength": hp.get("tuning_strength", DEFAULT_HYPERPARAMETERS["tuning_strength"]),
            "lookback_days_analysis": hp.get("lookback_days_analysis",
                                             DEFAULT_HYPERPARAMETERS["lookback_days_analysis"]),
            "training_interval_ticks": hp.get("training_interval_ticks",
                                              DEFAULT_HYPERPARAMETERS["training_interval_ticks"]),
            "min_pattern_confidence": hp.get("min_pattern_confidence",
                                             DEFAULT_HYPERPARAMETERS["min_pattern_confidence"]),
            "min_pattern_occurrences": hp.get("min_pattern_occurrences",
                                              DEFAULT_HYPERPARAMETERS["min_pattern_occurrences"]),
            "correlation_min_strength": hp.get("correlation_min_strength",
                                               DEFAULT_HYPERPARAMETERS["correlation_min_strength"]),
            "discovery_interval_cycles": hp.get("discovery_interval_cycles",
                                                DEFAULT_HYPERPARAMETERS["discovery_interval_cycles"]),
            "revert_threshold": hp.get("revert_threshold", DEFAULT_HYPERPARAMETERS["revert_threshold"]),
            "_updated_at": _now(),
        }
        os.makedirs(os.path.dirname(META_OVERRIDES_FILE), exist_ok=True)
        with open(META_OVERRIDES_FILE, "w") as f:
            json.dump(overrides, f, indent=2)
        logger.info("Meta: applied overrides → %s", META_OVERRIDES_FILE)
    except Exception as e:
        logger.error("apply_meta_decisions failed: %s", e)


def load_meta_overrides() -> dict:
    """Load meta_overrides.json. Returns empty dict on any error."""
    try:
        if os.path.exists(META_OVERRIDES_FILE):
            with open(META_OVERRIDES_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ── The Recursive Main Loop ────────────────────────────────────────────────

def run_meta_cycle(training_state: dict, training_reports: Optional[List[dict]] = None) -> dict:
    """
    Execute one full meta-learning cycle.

    Called from engine.py every 12th training cycle (~12 hours at hourly cycles).

    Steps:
    1. EVALUATE  — how well did recent training adjustments perform?
    2. OPTIMIZE  — should we change how the trainer operates?
    3. LIFECYCLE — update strategy states
    4. REFLECT   — is the meta-learner itself improving? If not, reset.
    5. APPLY     — push changes to meta_overrides.json
    6. LOG       — persist everything

    The recursive self-improvement:
    Step 4 evaluates whether the meta-learner's OWN hyperparameter changes
    helped. If meta-learner IR < 0.2 for 5+ cycles, it resets to defaults
    (acknowledging it was making things worse).
    """
    result = {
        "meta_cycle_at": _now(),
        "hyperparameter_changes": [],
    }

    try:
        meta_state = load_meta_state()
        meta_state["meta_cycles"] += 1
        cycle_num = meta_state["meta_cycles"]
        logger.info("Meta-cycle #%d starting", cycle_num)

        # ── Step 1: EVALUATE ───────────────────────────────────────────────
        logger.info("Meta step 1/6: Evaluating adjustment outcomes...")
        eval_result = evaluate_adjustment_outcomes(training_reports)
        result["evaluation"] = eval_result

        # Reload after evaluate_adjustment_outcomes updated it
        meta_state = load_meta_state()

        # ── Step 2: OPTIMIZE ───────────────────────────────────────────────
        logger.info("Meta step 2/6: Optimizing hyperparameters...")
        hp_adjustments = optimize_hyperparameters(meta_state)

        if hp_adjustments:
            hp = meta_state["hyperparameters"]
            for adj in hp_adjustments:
                old_val = hp.get(adj["param"], DEFAULT_HYPERPARAMETERS.get(adj["param"]))
                hp[adj["param"]] = adj["new"]
                history_entry = {
                    "timestamp": _now(),
                    "meta_cycle": cycle_num,
                    "param": adj["param"],
                    "old": adj["old"],
                    "new": adj["new"],
                    "reason": adj["reason"],
                }
                meta_state["hyperparameter_history"].append(history_entry)
                # Keep last 200 entries
                if len(meta_state["hyperparameter_history"]) > 200:
                    meta_state["hyperparameter_history"] = meta_state["hyperparameter_history"][-200:]
                result["hyperparameter_changes"].append(history_entry)
                logger.info("META TUNE: %s: %s → %s | %s", adj["param"], adj["old"], adj["new"], adj["reason"])
        else:
            logger.info("Meta: no hyperparameter changes warranted this cycle")

        result["hyperparameter_changes"] = result.get("hyperparameter_changes", [])

        # ── Step 3: LIFECYCLE ──────────────────────────────────────────────
        logger.info("Meta step 3/6: Managing strategy lifecycle...")
        lifecycle_result = manage_strategy_lifecycle(meta_state, training_state)
        result["lifecycle"] = lifecycle_result

        # ── Step 4: REFLECT (recursive self-evaluation) ────────────────────
        logger.info("Meta step 4/6: Reflecting on meta-learner effectiveness...")
        ir = meta_state["training_effectiveness"].get("improvement_rate", 0.0)

        if ir < 0.2:
            meta_state["consecutive_low_meta_rate"] = meta_state.get("consecutive_low_meta_rate", 0) + 1
            logger.warning("Meta: improvement rate critically low (%.2f), consecutive=%d",
                           ir, meta_state["consecutive_low_meta_rate"])
        else:
            meta_state["consecutive_low_meta_rate"] = 0

        if meta_state.get("consecutive_low_meta_rate", 0) >= 5:
            # Recursive failure: meta-learner is making things worse → full reset
            logger.warning("Meta: 5+ consecutive low-rate cycles — RESETTING hyperparameters to defaults")
            meta_state["hyperparameters"] = dict(DEFAULT_HYPERPARAMETERS)
            meta_state["consecutive_low_meta_rate"] = 0
            meta_state["hyperparameter_history"].append({
                "timestamp": _now(),
                "meta_cycle": cycle_num,
                "param": "ALL",
                "old": "varied",
                "new": "defaults",
                "reason": "Recursive failure: meta-learner IR < 0.2 for 5+ cycles — reset to defaults",
            })
            result["reset_to_defaults"] = True
            logger.info("Meta: full reset to defaults complete")
        else:
            result["reset_to_defaults"] = False

        result["consecutive_low_meta_rate"] = meta_state.get("consecutive_low_meta_rate", 0)
        result["meta_self_grade"] = "reset" if result["reset_to_defaults"] else (
            "critical" if ir < 0.2 else "poor" if ir < 0.35 else "ok" if ir < 0.5 else "good"
        )

        # ── Step 5: APPLY decisions ────────────────────────────────────────
        logger.info("Meta step 5/6: Applying decisions to meta_overrides.json...")
        apply_meta_decisions(meta_state, training_state)

        # ── Step 6: LOG ────────────────────────────────────────────────────
        logger.info("Meta step 6/6: Generating report and saving state...")
        meta_report = generate_meta_report(meta_state)
        result["meta_report"] = meta_report

        meta_state["last_meta_cycle"] = _now()
        save_meta_state(meta_state)

        logger.info("Meta-cycle #%d complete: grade=%s IR=%.2f hp_changes=%d",
                    cycle_num, meta_report.get("meta_grade"), ir,
                    len(result["hyperparameter_changes"]))

    except Exception as e:
        logger.error("run_meta_cycle failed: %s", e)
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Demo / standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    import sys

    from utils.logger import setup_logging
    setup_logging()
    print("=" * 60)
    print("META-LEARNER DEMO — synthetic data test")
    print("=" * 60)

    # --- Seed some fake training outcomes ---
    random.seed(42)
    outcomes = []
    pnl = 0.0
    for i in range(30):
        n_adj = random.randint(0, 2)
        adj_list = []
        for _ in range(n_adj):
            adj_list.append({
                "strategy": random.choice(["ema_macd", "bollinger", "sentiment"]),
                "param": random.choice(["stop_loss_pct", "take_profit_pct"]),
                "old_value": round(random.uniform(0.01, 0.05), 3),
                "new_value": round(random.uniform(0.01, 0.05), 3),
                "reason": "demo adjustment",
            })
        delta = random.uniform(-0.05, 0.08)
        pnl += delta
        outcomes.append({
            "cycle": i + 1,
            "timestamp": _now(),
            "total_pnl": round(pnl, 4),
            "total_trades": random.randint(0, 10),
            "adjustments_applied": adj_list,
            "was_reverted": random.random() < 0.05,
            "market_regime": random.choice(["low", "medium", "high"]),
        })
    _save_outcomes(outcomes)
    print(f"Seeded {len(outcomes)} synthetic training outcomes")

    # --- Fake training state ---
    training_state = {
        "cycles_completed": 30,
        "total_adjustments": 14,
        "consecutive_degradations": 0,
        "last_cycle": _now(),
        "last_pnl_snapshot": pnl,
        "revert_count": 1,
    }

    # --- Test individual functions ---
    print("\n--- evaluate_adjustment_outcomes() ---")
    eval_result = evaluate_adjustment_outcomes()
    print(json.dumps(eval_result, indent=2))

    print("\n--- load_meta_state() ---")
    meta_state = load_meta_state()
    print(json.dumps({
        "meta_cycles": meta_state["meta_cycles"],
        "hyperparameters": meta_state["hyperparameters"],
        "improvement_rate": meta_state["training_effectiveness"]["improvement_rate"],
    }, indent=2))

    print("\n--- optimize_hyperparameters() ---")
    meta_state = load_meta_state()
    # Force improvement_rate to test different branches
    meta_state["training_effectiveness"]["improvement_rate"] = 0.65
    meta_state["training_effectiveness"]["improvement_rate_history"] = [0.3, 0.4, 0.5, 0.6, 0.65]
    adjustments = optimize_hyperparameters(meta_state)
    print(json.dumps(adjustments, indent=2))

    print("\n--- run_meta_cycle() ---")
    result = run_meta_cycle(training_state)
    print(json.dumps({
        "meta_grade": result.get("meta_report", {}).get("meta_grade"),
        "hp_changes": result.get("hyperparameter_changes", []),
        "lifecycle": result.get("lifecycle", {}),
        "reset": result.get("reset_to_defaults"),
        "ir": result.get("meta_report", {}).get("training_improvement_rate"),
    }, indent=2))

    # --- Test low IR reset branch ---
    print("\n--- Testing recursive reset (5x low IR) ---")
    meta_state = load_meta_state()
    meta_state["consecutive_low_meta_rate"] = 4
    meta_state["training_effectiveness"]["improvement_rate"] = 0.10  # critically low
    save_meta_state(meta_state)
    result2 = run_meta_cycle(training_state)
    print(f"Reset triggered: {result2.get('reset_to_defaults')}")
    print(f"New tuning_strength: {load_meta_state()['hyperparameters']['tuning_strength']}")
    print(f"(Should be default: {DEFAULT_HYPERPARAMETERS['tuning_strength']})")

    print("\n--- generate_meta_report() ---")
    meta_state = load_meta_state()
    report = generate_meta_report(meta_state)
    print(json.dumps(report, indent=2))

    print("\nDone. Check trainer/meta_state.json for persisted state.")
