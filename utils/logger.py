"""
Trade logger — writes to TRADE_LOG.md and Python log file.
"""
import logging
import os
from datetime import datetime

import config

logger = logging.getLogger("cryptobot.logger")


def setup_logging():
    """Configure root logger for the bot."""
    root = logging.getLogger("cryptobot")
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)


def log_trade_to_md(position: dict):
    """Append a trade row to TRADE_LOG.md."""
    path = config.TRADE_LOG_PATH
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    side = position["side"].upper()
    entry = position["entry_price"]
    exit_p = position.get("exit_price", "—")
    size = position["size_btc"]
    strategy = position["strategy"]
    sl = position["stop_loss"]
    tp = position["take_profit"]
    pnl = position.get("pnl", 0)
    status = position["status"]

    if status == "open":
        result = "OPEN"
        notes = f"Entry @ ${entry:,.2f}"
    else:
        result = f"${pnl:+.2f}"
        notes = f"Entry ${entry:,.2f} → Exit ${exit_p:,.2f}"

    row = (f"| {date_str} | {time_str} | {config.PAIR_DISPLAY} | {side} "
           f"| ${entry:,.2f} | {size:.6f} | {strategy} "
           f"| ${sl:,.2f} | ${tp:,.2f} | {result} | {notes} |")

    # Append under the correct section
    if not os.path.exists(path):
        _init_trade_log(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Insert before "## Live Trades" section
    marker = "## Live Trades"
    if marker in content:
        idx = content.index(marker)
        content = content[:idx] + row + "\n" + content[idx:]
    else:
        content += "\n" + row + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Logged trade to TRADE_LOG.md: %s", row.strip())


def log_daily_summary(summary: dict):
    """Append daily summary to TRADE_LOG.md. Skips if already written for this date."""
    path = config.TRADE_LOG_PATH
    date_marker = f"### Daily Summary — {summary['date']}"

    # Check for duplicate: don't write if this date's summary already exists
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            if date_marker in f.read():
                logger.debug("Daily summary for %s already written — skipping", summary["date"])
                return

    lines = [
        "",
        date_marker,
        f"- **Balance:** ${summary['balance']:,.2f}",
        f"- **Daily P&L:** ${summary['daily_pnl']:+,.2f}",
        f"- **Trades opened:** {summary['trades_opened']}",
        f"- **Trades closed:** {summary['trades_closed']}",
        f"- **Win rate:** {summary['win_rate']:.0f}%",
        f"- **Drawdown:** {summary['drawdown_pct']:.1f}%",
        f"- **Open positions:** {summary['open_positions']}",
        f"- **Paused:** {'Yes' if summary['paused'] else 'No'}",
        "",
    ]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Daily summary written for %s", summary["date"])


def _init_trade_log(path: str):
    content = """# CryptoBot — Trade Log

## Format
| Date | Time | Pair | Side | Price | Size | Strategy | Stop-Loss | Take-Profit | Result | Notes |
|---|---|---|---|---|---|---|---|---|---|---|

## Paper Trades (Phase 1)

## Live Trades (Phase 2)
*Not started — awaiting Simon account setup + Phase 1 completion*
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
