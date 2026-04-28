"""
Trade Analyzer — dissects performance per strategy and generates a diagnosis.
Runs as step 1 of the recursive improvement loop.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config

logger = logging.getLogger("cryptobot.trainer.analyzer")


def load_state() -> dict:
    """Load bot state from JSON."""
    if os.path.exists(config.STATE_FILE):
        with open(config.STATE_FILE, "r") as f:
            return json.load(f)
    return {"positions": [], "balance": config.INITIAL_BALANCE}


def analyze_strategy(positions: list, strategy: str, lookback_days: int = 14) -> dict:
    """Analyze a single strategy's recent performance."""
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

    trades = [p for p in positions
              if p.get("strategy") == strategy
              and p.get("closed_at") and p["closed_at"] >= cutoff]

    if not trades:
        return {
            "strategy": strategy,
            "trade_count": 0,
            "status": "no_data",
            "diagnosis": "Not enough trades to evaluate",
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Max drawdown within this strategy
    running_pnl = 0
    peak = 0
    max_dd = 0
    for t in sorted(trades, key=lambda x: x["closed_at"]):
        running_pnl += t["pnl"]
        if running_pnl > peak:
            peak = running_pnl
        dd = peak - running_pnl
        if dd > max_dd:
            max_dd = dd

    # Average hold time
    hold_times = []
    for t in trades:
        if t.get("opened_at") and t.get("closed_at"):
            opened = datetime.fromisoformat(t["opened_at"])
            closed = datetime.fromisoformat(t["closed_at"])
            hold_times.append((closed - opened).total_seconds() / 3600)
    avg_hold_hours = sum(hold_times) / len(hold_times) if hold_times else 0

    # SL/TP hit analysis
    sl_hits = sum(1 for t in losses
                  if t.get("exit_price") and t.get("stop_loss")
                  and ((t["side"] == "buy" and t["exit_price"] <= t["stop_loss"] * 1.001)
                       or (t["side"] == "sell" and t["exit_price"] >= t["stop_loss"] * 0.999)))
    tp_hits = sum(1 for t in wins
                  if t.get("exit_price") and t.get("take_profit")
                  and ((t["side"] == "buy" and t["exit_price"] >= t["take_profit"] * 0.999)
                       or (t["side"] == "sell" and t["exit_price"] <= t["take_profit"] * 1.001)))

    # Generate diagnosis
    issues = []
    if win_rate < 50:
        issues.append(f"low_win_rate:{win_rate:.0f}%")
    if risk_reward < 1.3:
        issues.append(f"bad_risk_reward:{risk_reward:.2f}")
    if max_dd > abs(total_pnl) * 2:
        issues.append("excessive_drawdown")
    if avg_hold_hours > 168:  # > 7 days
        issues.append("holding_too_long")
    if avg_hold_hours < 0.5 and strategy != "grid":
        issues.append("exiting_too_fast")
    if sl_hits > len(trades) * 0.5:
        issues.append("stops_too_tight")

    # Long vs short breakdown
    long_trades = [t for t in trades if t["side"] == "buy"]
    short_trades = [t for t in trades if t["side"] == "sell"]
    long_win_rate = len([t for t in long_trades if t["pnl"] > 0]) / len(long_trades) * 100 if long_trades else 0
    short_win_rate = len([t for t in short_trades if t["pnl"] > 0]) / len(short_trades) * 100 if short_trades else 0

    if long_trades and short_trades:
        if long_win_rate > short_win_rate + 20:
            issues.append("shorts_underperforming")
        elif short_win_rate > long_win_rate + 20:
            issues.append("longs_underperforming")

    return {
        "strategy": strategy,
        "trade_count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "risk_reward": round(risk_reward, 2),
        "max_drawdown": round(max_dd, 4),
        "avg_hold_hours": round(avg_hold_hours, 1),
        "sl_hits": sl_hits,
        "tp_hits": tp_hits,
        "long_win_rate": round(long_win_rate, 1),
        "short_win_rate": round(short_win_rate, 1),
        "issues": issues,
        "status": "needs_improvement" if issues else "healthy",
    }


def full_analysis(lookback_days: int = 14) -> dict:
    """Analyze all strategies and return a complete diagnosis."""
    state = load_state()
    positions = state.get("positions", [])

    strategies = ["grid", "sentiment", "ema_macd", "bollinger", "rsi_divergence"]
    results = {}
    for strat in strategies:
        results[strat] = analyze_strategy(positions, strat, lookback_days)

    # Overall health
    total_trades = sum(r["trade_count"] for r in results.values())
    total_pnl = sum(r.get("total_pnl", 0) for r in results.values())
    all_issues = []
    for r in results.values():
        all_issues.extend(r.get("issues", []))

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "lookback_days": lookback_days,
        "balance": state.get("balance", config.INITIAL_BALANCE),
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 4),
        "strategies": results,
        "all_issues": all_issues,
        "overall_status": "healthy" if not all_issues else "needs_improvement",
    }
