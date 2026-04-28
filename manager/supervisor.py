"""
Supervisor — watches the bot process, auto-restarts on crash,
kills zombies, and runs health checks on a schedule.

This is the top-level process that should always be running.
It spawns bot.py as a subprocess and monitors it.
"""
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import config
from manager.health import full_health_check, format_health_report

logger = logging.getLogger("cryptobot.manager.supervisor")

HEALTH_INTERVAL = 900  # health check every 15 min
MAX_RESTARTS_PER_HOUR = 5
RESTART_BACKOFF_BASE = 30  # seconds

_running = True

def _shutdown(sig, frame):
    global _running
    logger.info("Supervisor shutdown signal received")
    _running = False

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def kill_zombie_bots(exclude_pid: int = None):
    """Kill any bot.py processes except the one we're managing."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-Process python* -ErrorAction SilentlyContinue | "
             "Where-Object { $_.CommandLine -match 'bot.py' } | "
             "Select-Object -ExpandProperty Id"],
            capture_output=True, text=True, timeout=10
        )
        pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip().isdigit()]
        killed = 0
        for pid in pids:
            if pid != exclude_pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed += 1
                    logger.warning("Killed zombie bot.py process: PID %d", pid)
                except (ProcessLookupError, PermissionError):
                    pass
        if killed:
            logger.info("Killed %d zombie bot processes", killed)
        return killed
    except Exception as e:
        logger.error("Failed to check for zombies: %s", e)
        return 0


def run_bot_supervised():
    """Main supervisor loop — run bot.py, monitor, restart on crash."""
    restart_count = 0
    restart_times = []
    last_health_check = 0

    logger.info("=" * 60)
    logger.info("CryptoBot Supervisor starting")
    logger.info("  Bot script: %s", os.path.join(config.BOT_DIR, "bot.py"))
    logger.info("  Health check interval: %ds", HEALTH_INTERVAL)
    logger.info("=" * 60)

    # Kill any existing zombies before starting
    kill_zombie_bots()

    while _running:
        # Start bot subprocess
        bot_cmd = [sys.executable, os.path.join(config.BOT_DIR, "bot.py")]
        logger.info("Starting bot process: %s", " ".join(bot_cmd))

        try:
            proc = subprocess.Popen(
                bot_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=config.BOT_DIR,
            )
            logger.info("Bot started: PID %d", proc.pid)

            # Monitor loop
            while _running:
                # Check if bot is still running
                retcode = proc.poll()
                if retcode is not None:
                    logger.warning("Bot process exited with code %d", retcode)
                    break

                # Periodic health check
                now = time.time()
                if now - last_health_check >= HEALTH_INTERVAL:
                    try:
                        report = full_health_check()
                        last_health_check = now

                        if report["overall"] in ("critical", "error"):
                            logger.error("HEALTH CHECK FAILED:\n%s", format_health_report(report))
                            # Save alert
                            alert_path = os.path.join(config.BOT_DIR, "manager", "alerts.jsonl")
                            with open(alert_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "level": report["overall"],
                                    "report": report,
                                }) + "\n")
                        elif report["overall"] == "warning":
                            logger.warning("Health check warnings:\n%s", format_health_report(report))
                        else:
                            logger.info("Health check: %s", report["overall"])

                        # Kill zombies (excluding our managed process)
                        kill_zombie_bots(exclude_pid=proc.pid)

                    except Exception as e:
                        logger.error("Health check failed: %s", e)

                time.sleep(10)  # Check every 10s

        except Exception as e:
            logger.exception("Failed to start bot: %s", e)

        if not _running:
            # Clean shutdown — kill the bot
            if proc and proc.poll() is None:
                logger.info("Shutting down bot process...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            break

        # Bot crashed — handle restart
        restart_count += 1
        restart_times.append(time.time())

        # Rate limit restarts
        recent_restarts = [t for t in restart_times if time.time() - t < 3600]
        restart_times = recent_restarts  # prune old ones

        if len(recent_restarts) >= MAX_RESTARTS_PER_HOUR:
            logger.critical("Too many restarts (%d in last hour) — supervisor pausing for 30 min",
                           len(recent_restarts))
            # Save critical alert
            alert_path = os.path.join(config.BOT_DIR, "manager", "alerts.jsonl")
            os.makedirs(os.path.dirname(alert_path), exist_ok=True)
            with open(alert_path, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "critical",
                    "detail": f"Bot crashed {len(recent_restarts)} times in the last hour — pausing restarts",
                }) + "\n")
            time.sleep(1800)
            restart_times.clear()
            continue

        # Exponential backoff
        backoff = min(RESTART_BACKOFF_BASE * (2 ** (len(recent_restarts) - 1)), 600)
        logger.info("Restarting bot in %ds (restart #%d)...", backoff, restart_count)
        for _ in range(int(backoff)):
            if not _running:
                break
            time.sleep(1)

    logger.info("Supervisor stopped. Total restarts: %d", restart_count)


if __name__ == "__main__":
    from utils.logger import setup_logging
    setup_logging()
    run_bot_supervised()
