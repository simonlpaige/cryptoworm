"""
Correlation Discovery Engine — finds NEW patterns in market data.

Unlike the trainer (which tunes existing strategy params), this module:
  1. Records every market snapshot to a rolling JSON database
  2. Scans signal pairs for predictive correlations (Pearson)
  3. Mines compound conditions for high-confidence patterns
  4. Proposes new strategy ideas when patterns are statistically sound

No numpy/pandas/scipy — pure stdlib math only.

Files:
  trainer/signal_history.json    — rolling 2000-entry time-series store
  trainer/strategy_proposals.json — discovered strategy proposals
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

logger = logging.getLogger("cryptobot.trainer.discovery")

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
SIGNAL_HISTORY_FILE = os.path.join(_HERE, "signal_history.json")
PROPOSALS_FILE = os.path.join(_HERE, "strategy_proposals.json")

MAX_ENTRIES = 2000          # ~7 days at 5-min ticks
MAX_FILE_BYTES = 4_900_000  # stay under 5MB


# ── Utilities ──────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: str, default) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
    return default


def _save_json(path: str, data) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error("Failed to save %s: %s", path, e)
        return False


def _safe_float(value, default=None):
    """Convert value to float, return default on failure."""
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


# ── Pearson Correlation ────────────────────────────────────────────────────

def pearson(x: list, y: list) -> float:
    """Pearson correlation coefficient (no numpy). Returns 0 on insufficient data."""
    # Filter out paired None/non-numeric values
    pairs = [(xi, yi) for xi, yi in zip(x, y)
             if xi is not None and yi is not None
             and not (isinstance(xi, float) and math.isnan(xi))
             and not (isinstance(yi, float) and math.isnan(yi))]
    n = len(pairs)
    if n < 10:
        return 0.0
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in pairs)
    den_x = sum((xi - mean_x) ** 2 for xi in xs) ** 0.5
    den_y = sum((yi - mean_y) ** 2 for yi in ys) ** 0.5
    if den_x * den_y == 0:
        return 0.0
    return num / (den_x * den_y)


# ── Signal History Database ────────────────────────────────────────────────

def _compute_rsi(closes: list, period: int = 14) -> Optional[float]:
    """Compute RSI from a list of closes. Returns None if insufficient data."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    # Wilder smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _compute_price_changes(ohlc: list) -> dict:
    """
    Compute approximate price changes over 1h/4h/24h from OHLC data.
    OHLC assumed to be 60-min candles (as fetched in engine.py).
    """
    closes = [c["close"] for c in ohlc]
    volumes = [c.get("volume", 0) for c in ohlc]
    if not closes:
        return {}

    current = closes[-1]
    result = {}

    def pct_change(n_bars):
        if len(closes) > n_bars:
            old = closes[-(n_bars + 1)]
            if old and old != 0:
                return round((current - old) / old * 100, 3)
        return None

    result["price_change_1h"] = pct_change(1)
    result["price_change_4h"] = pct_change(4)
    result["price_change_24h"] = pct_change(24)

    # Volume change vs prior bar
    if len(volumes) >= 2 and volumes[-2] and volumes[-2] != 0:
        result["volume_change"] = round((volumes[-1] - volumes[-2]) / volumes[-2] * 100, 2)
    else:
        result["volume_change"] = None

    # RSI
    result["rsi_14"] = _compute_rsi(closes)

    return result


def record_snapshot(market_context: dict, analysis: dict, ohlc: Optional[list] = None) -> Optional[dict]:
    """
    Record a market snapshot to signal_history.json.

    Args:
        market_context: Output from researcher.build_market_context()
        analysis: Output from analyzer.full_analysis()
        ohlc: Optional OHLC list (60-min candles) for computing price changes/RSI

    Returns:
        The snapshot dict added, or None on error.
    """
    try:
        entry = {"timestamp": _now_iso()}

        # ── Price & Volatility ─────────────────────────────────────────────
        vol = (market_context or {}).get("volatility") or {}
        entry["price"] = _safe_float(vol.get("current_price"))
        entry["atr_pct"] = _safe_float(vol.get("atr_pct"))
        entry["bb_width"] = _safe_float(vol.get("bb_width_pct"))

        # ── Price changes / RSI / Volume from OHLC ─────────────────────────
        if ohlc:
            computed = _compute_price_changes(ohlc)
            entry.update(computed)
        else:
            entry["price_change_1h"] = None
            entry["price_change_4h"] = None
            entry["price_change_24h"] = None
            entry["volume_change"] = None
            entry["rsi_14"] = None

        # ── Fear & Greed ───────────────────────────────────────────────────
        fg = (market_context or {}).get("fear_greed") or {}
        entry["fear_greed"] = _safe_float(fg.get("current"))

        # ── Funding Rate ───────────────────────────────────────────────────
        funding = (market_context or {}).get("funding_rate") or {}
        entry["funding_rate"] = _safe_float(funding.get("current_rate"))

        # ── Derivatives Sentiment ──────────────────────────────────────────
        deriv = (market_context or {}).get("derivatives_sentiment") or {}
        ls_data = deriv.get("long_short_ratio") or {}
        entry["long_short_ratio"] = _safe_float(ls_data.get("current"))

        oi_data = deriv.get("open_interest") or {}
        oi_value = _safe_float(oi_data.get("value"))
        # Store OI change as None (we'll compute it from history diffs)
        entry["open_interest_raw"] = oi_value
        entry["open_interest_change"] = None  # populated retroactively

        # ── Reddit / Social (if available) ────────────────────────────────
        reddit = (market_context or {}).get("reddit") or {}
        entry["reddit_btc_mentions"] = _safe_float(reddit.get("btc_mentions"))
        entry["reddit_btc_momentum"] = reddit.get("btc_momentum")

        # ── Congress PTR (if available) ────────────────────────────────────
        entry["congress_ptr_count_30d"] = _safe_float(
            (market_context or {}).get("congress_ptr_count_30d")
        )

        # ── Load existing history ──────────────────────────────────────────
        history = _load_json(SIGNAL_HISTORY_FILE, {"entries": []})
        entries = history.get("entries", [])

        # ── Retroactively compute OI change vs prior entry ─────────────────
        if entries and oi_value is not None:
            prev_oi = _safe_float(entries[-1].get("open_interest_raw"))
            if prev_oi and prev_oi != 0:
                entry["open_interest_change"] = round((oi_value - prev_oi) / prev_oi * 100, 3)

        # ── Append and prune ───────────────────────────────────────────────
        entries.append(entry)
        if len(entries) > MAX_ENTRIES:
            entries = entries[-MAX_ENTRIES:]
        history["entries"] = entries
        history["last_updated"] = _now_iso()
        history["entry_count"] = len(entries)

        # Size guard: if file would exceed limit, prune harder
        serialized = json.dumps(history)
        if len(serialized.encode("utf-8")) > MAX_FILE_BYTES:
            prune_to = int(MAX_ENTRIES * 0.75)
            history["entries"] = history["entries"][-prune_to:]
            history["entry_count"] = len(history["entries"])
            logger.warning("Signal history exceeded size limit — pruned to %d entries", prune_to)

        _save_json(SIGNAL_HISTORY_FILE, history)
        logger.debug("Discovery snapshot recorded (total entries: %d)", history["entry_count"])
        return entry

    except Exception as e:
        logger.error("record_snapshot failed: %s", e)
        return None


# ── Correlation Scanner ────────────────────────────────────────────────────

# Numeric signal fields we can correlate
NUMERIC_SIGNALS = [
    "fear_greed", "funding_rate", "bb_width", "atr_pct",
    "rsi_14", "volume_change", "long_short_ratio", "open_interest_change",
    "reddit_btc_mentions",
]

TARGET_FIELDS = ["price_change_1h", "price_change_4h", "price_change_24h"]

# Approximate bars per hour (assuming 60-min candles in ohlc → 1 entry per cycle)
# Signal history is written every training cycle (~60 min) → 1 entry/hour
BARS_PER_HOUR = 1


def _shift_targets(entries: list, lag_entries: int, target_field: str) -> list:
    """
    Shift target field forward by `lag_entries` positions.
    i.e. for entry i, return the value from entry i+lag_entries.
    This gives us "what happened N entries later".
    """
    n = len(entries)
    result = []
    for i in range(n):
        future_idx = i + lag_entries
        if future_idx < n:
            result.append(_safe_float(entries[future_idx].get(target_field)))
        else:
            result.append(None)
    return result


def scan_correlations(lookback_entries: int = 500) -> list:
    """
    Compute Pearson correlations between all signal pairs and forward price changes.

    Returns a sorted list of correlation findings (strongest first).
    """
    try:
        history = _load_json(SIGNAL_HISTORY_FILE, {"entries": []})
        all_entries = history.get("entries", [])
        if len(all_entries) < 20:
            logger.info("Not enough history for correlation scan (%d entries)", len(all_entries))
            return []

        entries = all_entries[-lookback_entries:]
        n = len(entries)
        results = []

        # Lags: 1h, 4h, 24h in terms of entries
        # Since we record one snapshot per training cycle (~60 min), 1 entry ≈ 1h
        lags = {
            "1h_forward": 1,
            "4h_forward": 4,
            "24h_forward": 24,
        }

        for signal in NUMERIC_SIGNALS:
            signal_vals = [_safe_float(e.get(signal)) for e in entries]

            for lag_name, lag_n in lags.items():
                # Correlate signal[i] with price_change at [i + lag_n]
                # Use stored price_change values (already represent 1h/4h/24h change)
                # Choose matching price_change field
                if "1h" in lag_name:
                    target_field = "price_change_1h"
                elif "4h" in lag_name:
                    target_field = "price_change_4h"
                else:
                    target_field = "price_change_24h"

                # Shift: get future price change for each entry
                future_changes = _shift_targets(entries, lag_n, target_field)

                # Filter to entries where both values are available
                paired_sig = []
                paired_tgt = []
                for sv, tv in zip(signal_vals, future_changes):
                    if sv is not None and tv is not None:
                        paired_sig.append(sv)
                        paired_tgt.append(tv)

                if len(paired_sig) < 20:
                    continue

                r = pearson(paired_sig, paired_tgt)
                abs_r = abs(r)

                if abs_r >= 0.15:  # only report meaningful correlations
                    direction = "positive" if r > 0 else "negative"
                    results.append({
                        "signal": signal,
                        "target": target_field,
                        "lag": lag_name,
                        "correlation": round(r, 4),
                        "abs_correlation": round(abs_r, 4),
                        "direction": direction,
                        "sample_size": len(paired_sig),
                        "interpretation": (
                            f"{signal} {'rises with' if r > 0 else 'falls before'} "
                            f"{target_field} (r={r:.3f}, n={len(paired_sig)})"
                        ),
                    })

        results.sort(key=lambda x: x["abs_correlation"], reverse=True)
        logger.info("Correlation scan found %d significant correlations from %d entries",
                    len(results), n)
        return results

    except Exception as e:
        logger.error("scan_correlations failed: %s", e)
        return []


# ── Pattern Miner ──────────────────────────────────────────────────────────

# Signal conditions to test (field, operator, threshold, label)
_CONDITIONS = [
    ("fear_greed",           "<",  20,   "fear_extreme_low"),
    ("fear_greed",           ">",  75,   "fear_extreme_high"),
    ("funding_rate",         ">",  0.02, "funding_high"),
    ("funding_rate",         "<", -0.005,"funding_negative"),
    ("bb_width",             "<",  2.5,  "bb_squeeze"),
    ("bb_width",             ">",  5.0,  "bb_wide"),
    ("rsi_14",               "<",  30,   "rsi_oversold"),
    ("rsi_14",               ">",  70,   "rsi_overbought"),
    ("volume_change",        ">",  30,   "volume_spike"),
    ("volume_change",        "<", -30,   "volume_drop"),
    ("atr_pct",              ">",  3.0,  "atr_high"),
    ("atr_pct",              "<",  1.5,  "atr_low"),
    ("long_short_ratio",     ">",  2.0,  "ls_ratio_high"),
    ("long_short_ratio",     "<",  0.7,  "ls_ratio_low"),
    ("open_interest_change", ">",  10,   "oi_rising_fast"),
    ("open_interest_change", "<", -10,   "oi_falling_fast"),
    ("reddit_btc_momentum",  "==", "rising",  "reddit_momentum_up"),
    ("reddit_btc_momentum",  "==", "falling", "reddit_momentum_down"),
]


def _check_condition(entry: dict, field: str, op: str, threshold) -> bool:
    """Check if an entry satisfies a condition."""
    val = entry.get(field)
    if val is None:
        return False
    try:
        if op == "<":
            return float(val) < float(threshold)
        elif op == ">":
            return float(val) > float(threshold)
        elif op == "==":
            return str(val) == str(threshold)
        return False
    except (TypeError, ValueError):
        return False


def _backtest_condition_combo(
    entries: list,
    conditions: list,
    horizon_bars: int,
    price_target_field: str,
    min_move_pct: float = 1.0,
) -> dict:
    """
    For a set of conditions, find entries where all conditions are met
    and check what happened `horizon_bars` later.

    Returns:
        {
            occurrences: int,
            wins: int,
            win_rate: float,
            avg_return: float,
            avg_loss: float,
            max_drawdown: float,
            max_gain: float,
        }
    """
    hits = []
    n = len(entries)

    for i in range(n - horizon_bars):
        if all(_check_condition(entries[i], f, op, thr) for f, op, thr, _ in conditions):
            future_idx = i + horizon_bars
            future_change = _safe_float(entries[future_idx].get(price_target_field))
            if future_change is not None:
                hits.append(future_change)

    if not hits:
        return {"occurrences": 0, "wins": 0, "win_rate": 0.0, "avg_return": 0.0,
                "avg_loss": 0.0, "max_drawdown": 0.0, "max_gain": 0.0}

    wins = [h for h in hits if h > min_move_pct]
    losses = [h for h in hits if h <= 0]
    win_rate = len(wins) / len(hits)
    avg_return = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    max_gain = max(hits)
    max_drawdown = min(hits)

    return {
        "occurrences": len(hits),
        "wins": len(wins),
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_return, 3),
        "avg_loss": round(avg_loss, 3),
        "max_drawdown": round(max_drawdown, 3),
        "max_gain": round(max_gain, 3),
    }


def mine_patterns(
    entries: Optional[list] = None,
    min_occurrences: int = 5,
    min_win_rate: float = 0.6,
    max_combo_size: int = 3,
) -> list:
    """
    Find compound patterns in signal history.

    Systematically tests 2- and 3-condition combinations against future price moves.
    Returns patterns sorted by confidence * frequency.

    Args:
        entries: Optional list of signal history entries (loads from file if None)
        min_occurrences: Minimum times a pattern must have occurred
        min_win_rate: Minimum win rate to report
        max_combo_size: Test pairs (2) and/or triples (3)
    """
    try:
        if entries is None:
            history = _load_json(SIGNAL_HISTORY_FILE, {"entries": []})
            entries = history.get("entries", [])

        if len(entries) < 30:
            logger.info("Not enough history for pattern mining (%d entries)", len(entries))
            return []

        # Horizons: (horizon_bars, target_field, label)
        horizons = [
            (1,  "price_change_1h",  "1h"),
            (4,  "price_change_4h",  "4h"),
            (24, "price_change_24h", "24h"),
        ]

        found_patterns = []

        for combo_size in range(2, max_combo_size + 1):
            for cond_combo in combinations(_CONDITIONS, combo_size):
                # Quick cardinality check: how many entries meet all conditions?
                hit_count = sum(
                    1 for e in entries
                    if all(_check_condition(e, f, op, thr) for f, op, thr, _ in cond_combo)
                )
                if hit_count < min_occurrences:
                    continue  # Skip expensive backtest if not enough hits

                for horizon_bars, target_field, horizon_label in horizons:
                    if horizon_bars >= len(entries) // 2:
                        continue  # Not enough future data

                    stats = _backtest_condition_combo(
                        entries, list(cond_combo), horizon_bars, target_field
                    )

                    if (stats["occurrences"] >= min_occurrences
                            and stats["win_rate"] >= min_win_rate):

                        condition_labels = [label for _, _, _, label in cond_combo]
                        condition_descs = [
                            f"{f} {op} {thr}" for f, op, thr, _ in cond_combo
                        ]
                        pattern_name = "_".join(condition_labels) + f"_{horizon_label}"

                        confidence = stats["win_rate"] * min(
                            1.0, stats["occurrences"] / 20
                        )  # Penalize small samples

                        found_patterns.append({
                            "name": pattern_name,
                            "conditions": condition_descs,
                            "condition_labels": condition_labels,
                            "horizon": horizon_label,
                            "target_field": target_field,
                            "occurrences": stats["occurrences"],
                            "wins": stats["wins"],
                            "win_rate": stats["win_rate"],
                            "avg_return_pct": stats["avg_return"],
                            "avg_loss_pct": stats["avg_loss"],
                            "max_gain_pct": stats["max_gain"],
                            "max_drawdown_pct": stats["max_drawdown"],
                            "confidence": round(confidence, 4),
                            "description": (
                                f"When {' AND '.join(condition_descs)}, "
                                f"price moves >{1.0:.0f}% within {horizon_label} "
                                f"({stats['wins']}/{stats['occurrences']} times, "
                                f"{stats['win_rate']*100:.0f}% win rate)"
                            ),
                        })

        # Sort by confidence * sample size (prefer reliable + frequent patterns)
        found_patterns.sort(
            key=lambda p: p["confidence"] * min(p["occurrences"] / 10, 1.0),
            reverse=True,
        )

        # Deduplicate: remove patterns dominated by a superset with similar stats
        seen_labels = set()
        deduped = []
        for p in found_patterns:
            key = frozenset(p["condition_labels"]) | {p["horizon"]}
            if key not in seen_labels:
                seen_labels.add(key)
                deduped.append(p)

        logger.info("Pattern mining found %d patterns from %d entries", len(deduped), len(entries))
        return deduped

    except Exception as e:
        logger.error("mine_patterns failed: %s", e)
        return []


# ── Strategy Proposals ─────────────────────────────────────────────────────

def _confidence_label(win_rate: float, sample_size: int) -> str:
    if sample_size < 10:
        return "low"
    if win_rate >= 0.75 and sample_size >= 20:
        return "high"
    if win_rate >= 0.65:
        return "medium"
    return "low"


def propose_strategy(pattern: dict) -> Optional[dict]:
    """
    Convert a discovered pattern into a strategy proposal and save it.

    Proposals are NOT automatically activated — they're surfaced for review.
    The training engine can promote them to 'testing' → 'active'.

    Returns the proposal dict, or None on error.
    """
    try:
        proposals_data = _load_json(PROPOSALS_FILE, {"proposals": []})
        proposals = proposals_data.get("proposals", [])

        # Build a slug name from conditions
        slug = "discovery_" + "_".join(
            label[:8] for label in pattern.get("condition_labels", [])
        )[:50]
        slug = slug.replace(" ", "_").replace(">", "gt").replace("<", "lt")

        # Check if we already have a proposal for this pattern
        for existing in proposals:
            if existing.get("name") == slug:
                # Update stats but keep status
                existing["expected_win_rate"] = pattern["win_rate"]
                existing["expected_avg_return"] = pattern.get("avg_return_pct", 0)
                existing["sample_size"] = pattern["occurrences"]
                existing["confidence"] = _confidence_label(
                    pattern["win_rate"], pattern["occurrences"]
                )
                existing["last_updated"] = _now_iso()
                _save_json(PROPOSALS_FILE, proposals_data)
                logger.info("Updated existing proposal: %s", slug)
                return existing

        # ── Derive SL/TP from observed stats ──────────────────────────────
        # SL: slightly worse than worst observed loss, min 1.0%
        max_drawdown = abs(pattern.get("max_drawdown_pct", -2.0))
        proposed_sl = round(max(max_drawdown * 1.2, 1.0), 2)

        # TP: slightly under best observed gain (conservative)
        max_gain = pattern.get("max_gain_pct", 3.0)
        proposed_tp = round(max(max_gain * 0.7, proposed_sl * 1.5), 2)

        proposal = {
            "name": slug,
            "description": pattern.get("description", ""),
            "entry_conditions": pattern.get("conditions", []),
            "horizon": pattern.get("horizon", "unknown"),
            "target_field": pattern.get("target_field", ""),
            "expected_win_rate": pattern["win_rate"],
            "expected_avg_return": pattern.get("avg_return_pct", 0),
            "sample_size": pattern["occurrences"],
            "confidence": _confidence_label(pattern["win_rate"], pattern["occurrences"]),
            "proposed_sl": proposed_sl,
            "proposed_tp": proposed_tp,
            "status": "proposed",  # proposed → testing → active → retired
            "created_at": _now_iso(),
            "last_updated": _now_iso(),
            "promoted_at": None,
        }

        proposals.append(proposal)
        proposals_data["proposals"] = proposals
        proposals_data["last_updated"] = _now_iso()
        _save_json(PROPOSALS_FILE, proposals_data)

        logger.info(
            "New strategy proposed: %s (win_rate=%.0f%%, n=%d, SL=%.1f%%, TP=%.1f%%)",
            slug, pattern["win_rate"] * 100, pattern["occurrences"],
            proposed_sl, proposed_tp,
        )
        return proposal

    except Exception as e:
        logger.error("propose_strategy failed: %s", e)
        return None


# ── Proposal Validation & Promotion ───────────────────────────────────────

def validate_proposals(min_confidence: str = "medium", max_promote: int = 2) -> list:
    """Validate and promote high-confidence proposals from 'proposed' to 'testing'.

    Criteria for promotion:
    - confidence >= min_confidence ('medium' or 'high')
    - sample_size >= 10
    - expected_win_rate >= 0.65

    Returns list of promoted proposal names.
    """
    try:
        proposals_data = _load_json(PROPOSALS_FILE, {"proposals": []})
        proposals = proposals_data.get("proposals", [])

        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        min_rank = confidence_rank.get(min_confidence, 1)

        promoted = []
        for p in proposals:
            if p.get("status") != "proposed":
                continue
            if len(promoted) >= max_promote:
                break

            conf_rank = confidence_rank.get(p.get("confidence", "low"), 0)
            if (conf_rank >= min_rank
                    and p.get("sample_size", 0) >= 10
                    and p.get("expected_win_rate", 0) >= 0.65):
                p["status"] = "testing"
                p["promoted_at"] = _now_iso()
                promoted.append(p["name"])
                logger.info("PROMOTED proposal '%s' to testing (WR=%.0f%%, n=%d, conf=%s)",
                            p["name"], p["expected_win_rate"] * 100,
                            p["sample_size"], p["confidence"])

        if promoted:
            proposals_data["last_updated"] = _now_iso()
            _save_json(PROPOSALS_FILE, proposals_data)

        return promoted
    except Exception as e:
        logger.error("validate_proposals failed: %s", e)
        return []


# ── Discovery Summary ──────────────────────────────────────────────────────

def get_discovery_summary() -> dict:
    """Return a summary of current discovery state for reporting."""
    try:
        history = _load_json(SIGNAL_HISTORY_FILE, {"entries": []})
        proposals_data = _load_json(PROPOSALS_FILE, {"proposals": []})

        entries = history.get("entries", [])
        proposals = proposals_data.get("proposals", [])

        return {
            "signal_history_entries": len(entries),
            "oldest_entry": entries[0]["timestamp"] if entries else None,
            "newest_entry": entries[-1]["timestamp"] if entries else None,
            "total_proposals": len(proposals),
            "proposals_by_status": {
                status: sum(1 for p in proposals if p.get("status") == status)
                for status in ["proposed", "testing", "active", "retired"]
            },
            "high_confidence_proposals": [
                p["name"] for p in proposals
                if p.get("confidence") == "high" and p.get("status") in ("proposed", "testing")
            ],
        }
    except Exception as e:
        logger.error("get_discovery_summary failed: %s", e)
        return {}


# ── Demo / Self-Test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("=== Correlation Discovery Engine — Demo ===\n")

    # ── Generate synthetic signal history ─────────────────────────────────
    logger.info("Generating 200 synthetic signal snapshots...")
    random.seed(42)

    base_price = 65000.0
    demo_entries = []
    for i in range(200):
        price = base_price * (1 + random.gauss(0, 0.005))
        fng = random.randint(10, 90)
        funding = random.uniform(-0.01, 0.05)
        rsi = random.uniform(20, 80)
        bb_width = random.uniform(1.0, 7.0)
        atr_pct = random.uniform(0.5, 5.0)
        volume_chg = random.uniform(-50, 80)
        ls_ratio = random.uniform(0.5, 2.5)
        oi_chg = random.uniform(-15, 20)

        # Create synthetic signal: fear < 20 + funding negative → +2% in 4h
        price_1h = random.gauss(0, 1.5)
        price_4h = random.gauss(0, 2.5)
        price_24h = random.gauss(0, 4.0)

        # Inject a detectable pattern: extreme fear + negative funding → +3% in 24h
        if fng < 20 and funding < -0.005:
            price_24h = abs(random.gauss(3.5, 1.0))  # strong upside
        # Another pattern: RSI overbought + high funding → -2% in 4h
        if rsi > 70 and funding > 0.02:
            price_4h = -abs(random.gauss(2.0, 0.8))

        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        demo_entries.append({
            "timestamp": ts,
            "price": round(price, 2),
            "price_change_1h": round(price_1h, 3),
            "price_change_4h": round(price_4h, 3),
            "price_change_24h": round(price_24h, 3),
            "fear_greed": fng,
            "funding_rate": round(funding, 5),
            "rsi_14": round(rsi, 1),
            "bb_width": round(bb_width, 2),
            "atr_pct": round(atr_pct, 2),
            "volume_change": round(volume_chg, 2),
            "long_short_ratio": round(ls_ratio, 3),
            "open_interest_change": round(oi_chg, 2),
            "reddit_btc_mentions": random.randint(20, 200),
            "reddit_btc_momentum": random.choice(["rising", "falling", "flat"]),
            "congress_ptr_count_30d": random.randint(0, 30),
        })

    # Save demo history
    demo_history = {
        "entries": demo_entries,
        "last_updated": _now_iso(),
        "entry_count": len(demo_entries),
    }
    _save_json(SIGNAL_HISTORY_FILE, demo_history)
    logger.info("Saved %d entries to %s\n", len(demo_entries), SIGNAL_HISTORY_FILE)

    # ── Pearson test ───────────────────────────────────────────────────────
    logger.info("--- Pearson Correlation Test ---")
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    r = pearson(x, y)
    logger.info("Perfect linear: pearson=%f (expected ~1.0)", r)
    y_neg = [-v for v in y]
    r2 = pearson(x, y_neg)
    logger.info("Perfect inverse: pearson=%f (expected ~-1.0)\n", r2)

    # ── Correlation scan ───────────────────────────────────────────────────
    logger.info("--- Correlation Scan ---")
    correlations = scan_correlations(lookback_entries=200)
    if correlations:
        logger.info("Top 5 correlations:")
        for c in correlations[:5]:
            logger.info("  %s → %s [%s]: r=%.3f (n=%d)",
                        c["signal"], c["target"], c["lag"],
                        c["correlation"], c["sample_size"])
    else:
        logger.info("  No significant correlations found")

    # ── Pattern mining ─────────────────────────────────────────────────────
    logger.info("\n--- Pattern Miner ---")
    patterns = mine_patterns(entries=demo_entries, min_occurrences=3, min_win_rate=0.55)
    logger.info("Found %d patterns", len(patterns))
    for p in patterns[:5]:
        logger.info("  PATTERN: %s", p["name"])
        logger.info("    Conditions: %s", " AND ".join(p["conditions"]))
        logger.info("    Win rate: %.0f%% (%d/%d) over %s",
                    p["win_rate"] * 100, p["wins"], p["occurrences"], p["horizon"])
        logger.info("    Confidence: %.2f", p["confidence"])

    # ── Strategy proposals ─────────────────────────────────────────────────
    logger.info("\n--- Strategy Proposals ---")
    for p in patterns[:3]:
        if p.get("confidence", 0) > 0.5:
            proposal = propose_strategy(p)
            if proposal:
                logger.info("  Proposed: %s (SL=%.1f%%, TP=%.1f%%)",
                            proposal["name"], proposal["proposed_sl"], proposal["proposed_tp"])

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("\n--- Discovery Summary ---")
    summary = get_discovery_summary()
    logger.info("  Signal history: %d entries (%s → %s)",
                summary.get("signal_history_entries", 0),
                summary.get("oldest_entry", "?"),
                summary.get("newest_entry", "?"))
    logger.info("  Total proposals: %d", summary.get("total_proposals", 0))
    logger.info("  High confidence: %s", summary.get("high_confidence_proposals", []))

    logger.info("\n=== Demo complete. Files written: ===")
    logger.info("  %s", SIGNAL_HISTORY_FILE)
    logger.info("  %s", PROPOSALS_FILE)
