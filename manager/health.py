"""
Health Manager — monitors the trading bot and its subsystems.

Checks:
  1. Bot process alive (last tick recency)
  2. Kraken API connectivity
  3. Strategy health (are strategies firing? stuck? erroring?)
  4. Risk rule violations (approaching limits)
  5. Position staleness (open too long?)
  6. Training engine running (cycles progressing?)
  7. Zombie process detection
  8. Disk/log bloat

Outputs a health report dict. Can be called standalone or from cron.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config

logger = logging.getLogger("cryptobot.manager.health")

STATE_FILE = config.STATE_FILE
TRAINING_STATE = os.path.join(config.BOT_DIR, "trainer", "training_state.json")
TUNING_LOG = os.path.join(config.BOT_DIR, "trainer", "tuning_log.json")
BOT_LOG = config.LOG_FILE


def check_bot_alive() -> dict:
    """Check if the bot is producing ticks by looking at log file mtime."""
    result = {"check": "bot_alive", "status": "unknown"}
    if not os.path.exists(BOT_LOG):
        result["status"] = "error"
        result["detail"] = "No bot.log found"
        return result

    mtime = os.path.getmtime(BOT_LOG)
    age_seconds = time.time() - mtime
    result["last_log_age_seconds"] = round(age_seconds, 1)

    if age_seconds < 600:  # within 10 min (2 tick cycles)
        result["status"] = "healthy"
    elif age_seconds < 1800:  # within 30 min
        result["status"] = "warning"
        result["detail"] = f"Last log activity {age_seconds/60:.0f}m ago"
    else:
        result["status"] = "critical"
        result["detail"] = f"Bot appears dead — no log activity for {age_seconds/60:.0f}m"
    return result


def check_kraken_api() -> dict:
    """Quick connectivity check to Kraken."""
    result = {"check": "kraken_api", "status": "unknown"}
    try:
        import urllib.request
        req = urllib.request.Request("https://api.kraken.com/0/public/Time")
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        if data.get("error"):
            result["status"] = "error"
            result["detail"] = str(data["error"])
        else:
            result["status"] = "healthy"
            result["server_time"] = data.get("result", {}).get("rfc1123", "")
    except Exception as e:
        result["status"] = "error"
        result["detail"] = str(e)
    return result


def check_positions() -> dict:
    """Check open positions for staleness and risk proximity."""
    result = {"check": "positions", "status": "healthy", "open": 0, "alerts": []}

    if not os.path.exists(STATE_FILE):
        result["detail"] = "No state file"
        return result

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    balance = state.get("balance", config.INITIAL_BALANCE)
    peak = state.get("peak_balance", config.INITIAL_BALANCE)
    positions = state.get("positions", [])

    open_positions = [p for p in positions if p.get("status") == "open"]
    result["open"] = len(open_positions)
    result["balance"] = round(balance, 2)
    result["peak_balance"] = round(peak, 2)

    # Drawdown check
    if peak > 0:
        drawdown_pct = (peak - balance) / peak * 100
        result["drawdown_pct"] = round(drawdown_pct, 2)
        if drawdown_pct >= config.DRAWDOWN_PAUSE_PCT * 0.8:
            result["alerts"].append(f"Approaching drawdown limit: {drawdown_pct:.1f}% (pause at {config.DRAWDOWN_PAUSE_PCT}%)")

    # Daily loss check
    daily_pnl = state.get("daily_pnl", 0)
    daily_limit = balance * (config.DAILY_MAX_LOSS_PCT / 100)
    if abs(daily_pnl) >= daily_limit * 0.8 and daily_pnl < 0:
        result["alerts"].append(f"Approaching daily loss limit: ${daily_pnl:.2f} (limit: ${daily_limit:.2f})")

    # Stale positions
    for pos in open_positions:
        if pos.get("opened_at"):
            opened = datetime.fromisoformat(pos["opened_at"])
            age_hours = (datetime.utcnow() - opened).total_seconds() / 3600
            if age_hours > 168:  # > 7 days
                result["alerts"].append(f"Stale position {pos['id']}: open {age_hours:.0f}h ({pos['strategy']})")
            # Unrealized P&L warning — would need current price, skip for now

    # Paused check
    if state.get("paused"):
        result["status"] = "warning"
        result["alerts"].append(f"Trading PAUSED: {state.get('pause_reason', 'unknown')}")

    if result["alerts"]:
        result["status"] = "warning"

    return result


def check_strategies() -> dict:
    """Check if strategies are actually firing (not silently broken)."""
    result = {"check": "strategies", "status": "healthy", "details": {}}

    if not os.path.exists(STATE_FILE):
        return result

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    positions = state.get("positions", [])
    strategies = ["grid", "sentiment", "ema_macd", "bollinger", "rsi_divergence"]

    for strat in strategies:
        strat_trades = [p for p in positions if p.get("strategy") == strat]
        last_trade = None
        if strat_trades:
            # Find most recent
            sorted_trades = sorted(strat_trades, key=lambda x: x.get("opened_at", ""), reverse=True)
            last_trade = sorted_trades[0].get("opened_at")

        result["details"][strat] = {
            "total_trades": len(strat_trades),
            "last_trade": last_trade,
        }

    # Flag strategies that have NEVER traded after 24h of runtime
    bot_start = None
    if os.path.exists(BOT_LOG):
        # Approximate from log mtime minus expected runtime
        log_size = os.path.getsize(BOT_LOG)
        if log_size > 50000:  # decent amount of logging = been running a while
            never_traded = [s for s, d in result["details"].items() if d["total_trades"] == 0]
            if len(never_traded) >= 3:
                result["status"] = "info"
                result["detail"] = f"Strategies never fired: {', '.join(never_traded)} (may be normal for current market)"

    return result


def check_training_engine() -> dict:
    """Check if the training engine is running and progressing."""
    result = {"check": "training_engine", "status": "healthy"}

    if not os.path.exists(TRAINING_STATE):
        result["status"] = "warning"
        result["detail"] = "No training state file — engine may not have run"
        return result

    with open(TRAINING_STATE, "r") as f:
        ts = json.load(f)

    result["cycles_completed"] = ts.get("cycles_completed", 0)
    result["total_adjustments"] = ts.get("total_adjustments", 0)
    result["revert_count"] = ts.get("revert_count", 0)

    last_cycle = ts.get("last_cycle")
    if last_cycle:
        last_dt = datetime.fromisoformat(last_cycle)
        age_hours = (datetime.utcnow() - last_dt).total_seconds() / 3600
        result["last_cycle_hours_ago"] = round(age_hours, 1)
        if age_hours > 2:
            result["status"] = "warning"
            result["detail"] = f"Training engine stale — last cycle {age_hours:.1f}h ago"

    if ts.get("revert_count", 0) >= 3:
        result["status"] = "warning"
        result["detail"] = f"Training engine has reverted {ts['revert_count']} times — parameters may be unstable"

    return result


def check_log_size() -> dict:
    """Check if logs are getting too large."""
    result = {"check": "log_size", "status": "healthy"}
    files_to_check = [
        (BOT_LOG, 50),  # 50 MB
        (TUNING_LOG, 10),
    ]
    for path, max_mb in files_to_check:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > max_mb:
                result["status"] = "warning"
                result["alerts"] = result.get("alerts", [])
                result["alerts"].append(f"{os.path.basename(path)}: {size_mb:.1f}MB (max {max_mb}MB)")
    return result


def check_zombie_processes() -> dict:
    """Check for multiple bot.py processes (zombie detection)."""
    result = {"check": "zombies", "status": "healthy", "count": 0}
    try:
        import subprocess
        out = subprocess.check_output(
            ["powershell", "-Command",
             "(Get-Process python* -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match 'bot.py' }).Count"],
            timeout=10, text=True
        ).strip()
        count = int(out) if out.isdigit() else 0
        result["count"] = count
        if count > 1:
            result["status"] = "critical"
            result["detail"] = f"{count} bot.py processes running — zombies detected!"
        elif count == 0:
            result["status"] = "warning"
            result["detail"] = "No bot.py process found — bot may be dead"
    except Exception as e:
        result["status"] = "unknown"
        result["detail"] = str(e)
    return result


def full_health_check() -> dict:
    """Run all health checks and return a consolidated report."""
    checks = [
        check_bot_alive(),
        check_kraken_api(),
        check_positions(),
        check_strategies(),
        check_training_engine(),
        check_log_size(),
        check_zombie_processes(),
    ]

    overall = "healthy"
    for check in checks:
        if check["status"] == "critical":
            overall = "critical"
            break
        elif check["status"] == "warning" and overall != "critical":
            overall = "warning"
        elif check["status"] == "error" and overall not in ("critical", "warning"):
            overall = "error"

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall": overall,
        "checks": {c["check"]: c for c in checks},
    }

    # Save report
    report_path = os.path.join(config.BOT_DIR, "manager", "last_health.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def format_health_report(report: dict) -> str:
    """Format health report as readable text."""
    lines = []
    status_emoji = {"healthy": "✅", "warning": "⚠️", "error": "❌", "critical": "🚨", "info": "ℹ️", "unknown": "❓"}

    lines.append(f"{status_emoji.get(report['overall'], '❓')} CryptoBot Health: {report['overall'].upper()}")
    lines.append(f"Checked: {report['timestamp'][:19]}Z")
    lines.append("")

    for name, check in report["checks"].items():
        emoji = status_emoji.get(check["status"], "❓")
        line = f"  {emoji} {name}: {check['status']}"
        if "detail" in check:
            line += f" — {check['detail']}"
        lines.append(line)

        # Extra info
        if name == "positions":
            lines.append(f"     Open: {check.get('open', 0)} | Balance: ${check.get('balance', 0):.2f} | Drawdown: {check.get('drawdown_pct', 0):.1f}%")
            for alert in check.get("alerts", []):
                lines.append(f"     ⚠️ {alert}")
        elif name == "training_engine":
            lines.append(f"     Cycles: {check.get('cycles_completed', 0)} | Adjustments: {check.get('total_adjustments', 0)} | Reverts: {check.get('revert_count', 0)}")
        elif name == "zombies":
            if check.get("count", 0) > 0:
                lines.append(f"     Processes: {check['count']}")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = full_health_check()
    print(format_health_report(report))
