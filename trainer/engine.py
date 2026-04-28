"""
Training Engine — the recursive self-improvement loop.

Cycle:
  1. ANALYZE  — score each strategy's recent performance
  2. RESEARCH — fetch market context (volatility regime, sentiment)
  3. DIAGNOSE — map issues to root causes
  4. TUNE     — adjust one parameter per strategy within safe bounds
  5. LOG      — record everything
  6. WAIT     — sleep, then repeat

Safety:
  - Parameters never leave research-backed bounds
  - Max one change per strategy per cycle
  - Max 20% range movement per cycle
  - All changes logged with full before/after
  - Reverts if performance degrades over 3 consecutive cycles
"""
import json
import logging
import math
import os
import time
import signal
from datetime import datetime

import config
from trainer.analyzer import full_analysis
from trainer.researcher import build_market_context, RESEARCH_PARAMS
from trainer.tuner import generate_adjustments, apply_adjustments, load_overrides
from utils.kraken_client import KrakenClient

logger = logging.getLogger("cryptobot.trainer.engine")

TRAINING_STATE_FILE = os.path.join(config.BOT_DIR, "trainer", "training_state.json")
TRAINING_REPORT_DIR = os.path.join(config.BOT_DIR, "trainer", "reports")

_running = True

def _shutdown(sig, frame):
    global _running
    logger.info("Training engine shutdown signal received")
    _running = False

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def load_training_state() -> dict:
    if os.path.exists(TRAINING_STATE_FILE):
        with open(TRAINING_STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "cycles_completed": 0,
        "total_adjustments": 0,
        "consecutive_degradations": 0,
        "last_cycle": None,
        "last_pnl_snapshot": None,
        "revert_count": 0,
    }


def save_training_state(state: dict):
    os.makedirs(os.path.dirname(TRAINING_STATE_FILE), exist_ok=True)
    with open(TRAINING_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def save_report(cycle: int, report: dict):
    """Save a per-cycle report as JSON."""
    os.makedirs(TRAINING_REPORT_DIR, exist_ok=True)
    path = os.path.join(TRAINING_REPORT_DIR, f"cycle_{cycle:04d}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", path)


def check_for_revert(training_state: dict, current_pnl: float) -> bool:
    """
    If PnL has degraded for 3 consecutive cycles, revert all overrides
    to defaults and reset.
    """
    last_pnl = training_state.get("last_pnl_snapshot")
    if last_pnl is not None and current_pnl < last_pnl:
        training_state["consecutive_degradations"] += 1
        logger.warning("Performance degraded: $%.4f → $%.4f (streak: %d)",
                       last_pnl, current_pnl, training_state["consecutive_degradations"])
    else:
        training_state["consecutive_degradations"] = 0

    if training_state["consecutive_degradations"] >= 3:
        logger.warning("3 consecutive degradations — REVERTING all overrides to defaults")
        # Clear overrides file
        overrides_file = os.path.join(config.BOT_DIR, "trainer", "param_overrides.json")
        if os.path.exists(overrides_file):
            with open(overrides_file, "w") as f:
                json.dump({}, f)
        training_state["consecutive_degradations"] = 0
        training_state["revert_count"] = training_state.get("revert_count", 0) + 1
        return True
    return False


def simulate_backtest_trades(ohlc: list, strategies: list) -> list:
    """Simulate what-if trades from recent OHLC data for zero-trade periods.

    When no real trades have closed, this generates synthetic outcomes so the
    analyzer has something to learn from. Rough but better than silence.

    Logic per strategy:
    - ema_macd / rsi_divergence: enter at open, exit at close each bar
    - sentiment: skip (no meaningful intra-bar signal)
    - bollinger: enter when bar touches lower band, exit at SMA
    - grid: not simulated (grid requires state)

    Returns list of synthetic position-like dicts compatible with analyzer.
    """
    if not ohlc or len(ohlc) < 10:
        return []

    synthetic = []
    now = datetime.utcnow().isoformat()

    # ── EMA/MACD: simple momentum simulation ─────────────────────────────
    if "ema_macd" in strategies:
        # Use last 20 bars; enter long if close > open (green candle), short otherwise
        for i, bar in enumerate(ohlc[-20:]):
            side = "buy" if bar["close"] > bar["open"] else "sell"
            entry = bar["open"]
            exit_price = bar["close"]
            pnl_pct = (exit_price - entry) / entry if side == "buy" else (entry - exit_price) / entry
            pnl = pnl_pct * 10.0  # small notional ($10 per sim trade)
            synthetic.append({
                "strategy": "ema_macd",
                "side": side,
                "entry_price": entry,
                "exit_price": exit_price,
                "pnl": round(pnl, 4),
                "opened_at": now,
                "closed_at": now,
                "simulated": True,
            })

    # ── RSI Divergence: similar bar-by-bar logic ──────────────────────────
    if "rsi_divergence" in strategies:
        closes = [c["close"] for c in ohlc]
        # Compute a rough 14-period RSI for each bar
        for i in range(14, len(ohlc[-20:]) + 14):
            if i >= len(closes):
                break
            gains = [max(closes[j] - closes[j - 1], 0) for j in range(i - 13, i + 1)]
            losses = [max(closes[j - 1] - closes[j], 0) for j in range(i - 13, i + 1)]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))

            bar = ohlc[i] if i < len(ohlc) else ohlc[-1]
            if rsi < 40:  # oversold → simulate long
                side = "buy"
                entry = bar["open"]
                # Simulate: exit after 2 bars or at low (whatever comes first)
                exit_bar = ohlc[min(i + 2, len(ohlc) - 1)]
                exit_price = exit_bar["close"]
            elif rsi > 60:  # overbought → simulate short
                side = "sell"
                entry = bar["open"]
                exit_bar = ohlc[min(i + 2, len(ohlc) - 1)]
                exit_price = exit_bar["close"]
            else:
                continue

            pnl_pct = (exit_price - entry) / entry if side == "buy" else (entry - exit_price) / entry
            pnl = pnl_pct * 10.0
            synthetic.append({
                "strategy": "rsi_divergence",
                "side": side,
                "entry_price": entry,
                "exit_price": exit_price,
                "pnl": round(pnl, 4),
                "opened_at": now,
                "closed_at": now,
                "simulated": True,
            })

    # ── Bollinger: enter near lower band, exit at SMA ─────────────────────
    if "bollinger" in strategies:
        closes = [c["close"] for c in ohlc]
        for i in range(20, len(ohlc)):
            window = closes[i - 20:i]
            sma = sum(window) / 20
            std = math.sqrt(sum((x - sma) ** 2 for x in window) / 20)
            lower_band = sma - 2 * std
            upper_band = sma + 2 * std

            bar = ohlc[i]
            if bar["low"] <= lower_band:  # touched lower band → long
                entry = lower_band
                exit_price = sma
                pnl_pct = (exit_price - entry) / entry
                pnl = pnl_pct * 10.0
                synthetic.append({
                    "strategy": "bollinger",
                    "side": "buy",
                    "entry_price": round(entry, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 4),
                    "opened_at": now,
                    "closed_at": now,
                    "simulated": True,
                })
            elif bar["high"] >= upper_band:  # touched upper band → short
                entry = upper_band
                exit_price = sma
                pnl_pct = (entry - exit_price) / entry
                pnl = pnl_pct * 10.0
                synthetic.append({
                    "strategy": "bollinger",
                    "side": "sell",
                    "entry_price": round(entry, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 4),
                    "opened_at": now,
                    "closed_at": now,
                    "simulated": True,
                })

    logger.info("Simulated %d backtest trades across %d strategies",
                len(synthetic), len(strategies))
    return synthetic


def inject_backtest_results(analysis: dict) -> dict:
    """Replace synthetic simulation with real backtest results for strategies
    with no closed-trade data.

    Runs the actual backtester against cached historical data (fast if cache
    exists, ~30s otherwise). Uses backtest win_rate, Sharpe, and profit_factor
    as the evaluation signal for the trainer.
    """
    no_data_strats = [
        name for name, strat_analysis in analysis.get("strategies", {}).items()
        if strat_analysis.get("status") == "no_data"
    ]

    if not no_data_strats:
        return analysis

    logger.info("No closed trades for: %s — running real backtests", no_data_strats)

    # Check for cached backtest results (run full backtest at most once per 6 hours)
    cache_path = os.path.join(config.BOT_DIR, "trainer", "backtest_cache.json")
    cache_valid = False
    cached = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            cached_at = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
            if (datetime.utcnow() - cached_at).total_seconds() < 21600:  # 6 hours
                cache_valid = True
                logger.info("Using cached backtest results (age: %.1fh)",
                            (datetime.utcnow() - cached_at).total_seconds() / 3600)
        except Exception:
            pass

    if not cache_valid:
        try:
            from trainer.backtester import fetch_historical_data, run_strategy_backtest
            candles = fetch_historical_data(months=3)  # 3 months for speed
            if candles and len(candles) > 100:
                all_results = {}
                for strat in ["grid", "sentiment", "ema_macd", "bollinger",
                              "rsi_divergence", "congress_frontrun"]:
                    try:
                        result = run_strategy_backtest(strat, candles)
                        all_results[strat] = result
                        logger.info("Backtest %s: %d trades, %.1f%% WR, PF=%.2f, Sharpe=%.2f",
                                    strat, result["total_trades"], result["win_rate"],
                                    result["profit_factor"], result["sharpe_ratio"])
                    except Exception as e:
                        logger.warning("Backtest %s failed: %s", strat, e)

                cached = {"cached_at": datetime.utcnow().isoformat(), "results": all_results}
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cached, f, indent=2)
            else:
                logger.warning("Insufficient historical data for backtests (%d candles)",
                               len(candles) if candles else 0)
                return analysis
        except Exception as e:
            logger.error("Backtest run failed: %s", e)
            return analysis

    # Inject backtest results into analysis
    bt_results = cached.get("results", {})
    for strat in no_data_strats:
        bt = bt_results.get(strat)
        if not bt or bt.get("total_trades", 0) == 0:
            continue

        issues = []
        if bt["win_rate"] < 50:
            issues.append(f"low_win_rate:{bt['win_rate']:.0f}%")
        pf = bt.get("profit_factor", 0)
        if 0 < pf < 1.3:
            issues.append(f"bad_risk_reward:{pf:.2f}")
        if bt.get("max_drawdown_pct", 0) > 10:
            issues.append("excessive_drawdown")

        analysis["strategies"][strat] = {
            "strategy": strat,
            "trade_count": bt["total_trades"],
            "wins": bt.get("wins", 0),
            "losses": bt.get("losses", 0),
            "win_rate": bt["win_rate"],
            "total_pnl": bt.get("total_pnl", 0),
            "avg_win": round(bt.get("gross_profit", 0) / max(bt.get("wins", 1), 1), 4),
            "avg_loss": round(-bt.get("gross_loss", 0) / max(bt.get("losses", 1), 1), 4),
            "risk_reward": bt.get("profit_factor", 0),
            "max_drawdown": bt.get("max_drawdown_pct", 0),
            "avg_hold_hours": bt.get("avg_hold_time_hours", 0),
            "sl_hits": 0,
            "tp_hits": 0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
            "issues": issues,
            "status": "backtest" if not issues else "backtest_needs_improvement",
            "note": f"From real backtest: {bt['total_trades']} trades, Sharpe={bt.get('sharpe_ratio', 0):.2f}",
        }

    # Update totals
    analysis["total_trades"] = sum(
        s.get("trade_count", 0) for s in analysis["strategies"].values()
    )
    analysis["total_pnl"] = round(sum(
        s.get("total_pnl", 0) for s in analysis["strategies"].values()
    ), 4)

    return analysis


def run_cycle(kraken: KrakenClient, training_state: dict) -> dict:
    """Execute one training cycle. Returns a report dict."""
    cycle_num = training_state["cycles_completed"] + 1
    logger.info("=" * 50)
    logger.info("TRAINING CYCLE #%d", cycle_num)
    logger.info("=" * 50)

    report = {
        "cycle": cycle_num,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # ── Step 1: ANALYZE ──────────────────────────────────────────────────
    logger.info("Step 1: Analyzing trade performance...")
    # Read lookback_days from meta_overrides (meta-learner can adjust this)
    _meta_lookback = 14
    try:
        _meta_overrides_path = os.path.join(config.BOT_DIR, "trainer", "meta_overrides.json")
        if os.path.exists(_meta_overrides_path):
            with open(_meta_overrides_path, "r") as _f:
                _meta_overrides_path = json.load(_f)
            _meta_lookback = int(_meta_overrides_path.get("lookback_days_analysis", 14))
    except Exception:
        pass
    analysis = full_analysis(lookback_days=_meta_lookback)
    report["analysis"] = analysis
    logger.info("  Total trades: %d | PnL: $%.4f | Issues: %s",
                analysis["total_trades"], analysis["total_pnl"], analysis["all_issues"])

    # ── Step 2: RESEARCH ─────────────────────────────────────────────────
    logger.info("Step 2: Fetching market context...")
    ohlc = kraken.get_ohlc(interval=60, count=100)

    # Inject real backtest results when no live trades have closed yet
    if analysis["total_trades"] == 0:
        logger.info("No closed trades yet — injecting real backtest results")
        analysis = inject_backtest_results(analysis)
        report["analysis"] = analysis
        report["backtest_used"] = True
        logger.info("  After backtest injection: %d trades from backtests", analysis["total_trades"])
    else:
        report["backtest_used"] = False

    market_context = build_market_context(ohlc)
    report["market_context"] = market_context
    fg = market_context.get("fear_greed", {})
    vol = market_context.get("volatility", {})
    logger.info("  Fear & Greed: %s (%s) | Volatility: %s | Recommendations: %s",
                fg.get("current", "?"), fg.get("classification", "?"),
                vol.get("regime", "?"), market_context.get("recommendations", []))

    # ── Always record discovery snapshot ─────────────────────────────────
    try:
        from trainer.discovery import record_snapshot
        record_snapshot(market_context, analysis, ohlc)
    except Exception as de:
        logger.error("Discovery snapshot failed: %s", de)

    # ── Step 3: CHECK FOR REVERT ─────────────────────────────────────────
    reverted = check_for_revert(training_state, analysis["total_pnl"])
    report["reverted"] = reverted
    if reverted:
        logger.warning("Overrides reverted to defaults this cycle")
        report["adjustments"] = {"applied": [], "total": 0, "reason": "reverted"}
    else:
        # ── Step 4: TUNE ─────────────────────────────────────────────────
        logger.info("Step 3: Generating parameter adjustments...")
        adjustments = generate_adjustments(analysis, market_context)
        report["proposed_adjustments"] = [
            {"strategy": a["strategy"], "param": a["param"],
             "old": a["old_value"], "new": a["new_value"], "reason": a["reason"]}
            for a in adjustments
        ]

        if adjustments:
            logger.info("Step 4: Applying %d adjustments...", len(adjustments))
            result = apply_adjustments(adjustments)
            report["adjustments"] = result
            training_state["total_adjustments"] += result["total"]
        else:
            logger.info("No adjustments needed this cycle")
            report["adjustments"] = {"applied": [], "total": 0}

    # ── Step 5: DISCOVER ─────────────────────────────────────────────────
    if cycle_num % 6 == 0:  # every 6th cycle (~6 hours at 60-min intervals)
        logger.info("Step 5: Running correlation discovery...")
        try:
            from trainer.discovery import scan_correlations, mine_patterns, propose_strategy, get_discovery_summary, validate_proposals

            correlations = scan_correlations()
            patterns = mine_patterns()

            if correlations:
                report["top_correlations"] = correlations[:5]
                logger.info("  Top correlation: %s → %s (r=%.3f)",
                            correlations[0]["signal"], correlations[0]["target"],
                            correlations[0]["correlation"])

            if patterns:
                report["discovered_patterns"] = patterns[:3]
                logger.info("  Found %d patterns; top: %s (%.0f%% win rate, n=%d)",
                            len(patterns), patterns[0]["name"],
                            patterns[0]["win_rate"] * 100, patterns[0]["occurrences"])
                for p in patterns:
                    if p.get("confidence", 0) > 0.7:
                        proposal = propose_strategy(p)
                        if proposal:
                            logger.info("  Strategy proposed: %s", proposal["name"])

            # Auto-promote qualifying proposals from 'proposed' → 'testing'
            promoted = validate_proposals(min_confidence="medium", max_promote=2)
            if promoted:
                report["promoted_proposals"] = promoted
                logger.info("  Promoted %d proposals to testing: %s", len(promoted), promoted)

            summary = get_discovery_summary()
            report["discovery_summary"] = summary
            logger.info("  Discovery: %d history entries, %d total proposals (%s)",
                        summary.get("signal_history_entries", 0),
                        summary.get("total_proposals", 0),
                        summary.get("proposals_by_status", {}))
        except Exception as disc_err:
            logger.error("Discovery step failed: %s", disc_err)

    # ── Update state ─────────────────────────────────────────────────────
    training_state["cycles_completed"] = cycle_num
    training_state["last_cycle"] = datetime.utcnow().isoformat()
    training_state["last_pnl_snapshot"] = analysis["total_pnl"]

    save_report(cycle_num, report)
    save_training_state(training_state)

    # ── Step 6: META-LEARNING ─────────────────────────────────────────────
    # Record this cycle's results for meta-evaluation.
    # Run a full meta-cycle every 12th training cycle (~12 hours).
    try:
        from trainer.meta_learner import record_training_outcome, run_meta_cycle, load_meta_state

        # Always record the outcome so meta-learner has data to learn from
        record_training_outcome(report)

        if cycle_num % 12 == 0:
            logger.info("Step 6: Running meta-learning cycle (every 12th cycle)...")
            meta_state = load_meta_state()
            meta_report = run_meta_cycle(training_state)
            report["meta_learning"] = meta_report

            if meta_report.get("hyperparameter_changes"):
                for change in meta_report["hyperparameter_changes"]:
                    logger.info("META: %s: %s → %s (%s)",
                                change["param"], change["old"], change["new"], change["reason"])
            if meta_report.get("reset_to_defaults"):
                logger.warning("META: reset all hyperparameters to defaults (recursive failure recovery)")
        else:
            logger.debug("Step 6: meta-learning outcome recorded (meta-cycle every 12th)")
    except Exception as me:
        logger.error("Meta-learning failed: %s", me)

    # ── Step 7: AUTO-LEARNING (failure pattern encoding) ─────────────
    try:
        # Track API failure patterns for diagnosis
        if hasattr(kraken, 'get_failure_summary'):
            failure_summary = kraken.get_failure_summary()
            if failure_summary.get('consecutive_failures', 0) > 0:
                report['api_failures'] = failure_summary
                logger.warning("API failure summary: %s", failure_summary)

        # Detect strategies with 0 trades over extended period
        zero_trade_strategies = []
        for strat_name, strat_data in analysis.get('strategies', {}).items():
            if strat_data.get('trade_count', 0) == 0 and strat_data.get('status') == 'no_data':
                zero_trade_strategies.append(strat_name)
        if zero_trade_strategies and cycle_num > 12:  # after first 12 hours
            report['auto_learn'] = report.get('auto_learn', [])
            report['auto_learn'].append({
                'type': 'zero_trade_strategies',
                'strategies': zero_trade_strategies,
                'cycles_elapsed': cycle_num,
                'note': 'These strategies have produced 0 trades. '
                        'Check if conditions are unreachable in current market regime.'
            })
            logger.warning("[auto-learn] Zero-trade strategies after %d cycles: %s",
                          cycle_num, zero_trade_strategies)
    except Exception as al_err:
        logger.error("Auto-learning step failed: %s", al_err)

    logger.info("Cycle #%d complete. Adjustments: %d | Balance: $%.2f",
                cycle_num, report["adjustments"].get("total", 0), analysis["balance"])

    return report


def run_forever(interval_minutes: int = 60):
    """
    Main entry point: run the training loop forever.
    
    Default: every 60 minutes.
    On a $500 paper account with 5-min ticks, this means the trainer
    evaluates ~12 data points per cycle and tunes accordingly.
    """
    logger.info("Training Engine starting (interval=%dm)", interval_minutes)
    kraken = KrakenClient()
    training_state = load_training_state()

    while _running:
        try:
            report = run_cycle(kraken, training_state)

            # Log summary
            adj_count = report["adjustments"].get("total", 0)
            if adj_count > 0:
                logger.info("Cycle summary: %d parameters tuned", adj_count)
                for a in report["adjustments"].get("applied", []):
                    logger.info("  → %s.%s: %s → %s", a["strategy"], a["param"],
                                a["old_value"], a["new_value"])

        except Exception as e:
            logger.exception("Training cycle failed: %s", e)

        # Sleep in small increments for graceful shutdown
        for _ in range(interval_minutes * 60):
            if not _running:
                break
            time.sleep(1)

    logger.info("Training Engine stopped. Cycles: %d, Total adjustments: %d",
                training_state["cycles_completed"], training_state["total_adjustments"])


def run_once():
    """Run a single training cycle and exit (for testing)."""
    kraken = KrakenClient()
    training_state = load_training_state()
    report = run_cycle(kraken, training_state)
    return report


if __name__ == "__main__":
    from utils.logger import setup_logging
    setup_logging()
    
    import sys
    if "--once" in sys.argv:
        report = run_once()
        print(json.dumps(report, indent=2))
    else:
        run_forever()
