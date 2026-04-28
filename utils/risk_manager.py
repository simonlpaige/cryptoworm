"""
Risk Manager — enforces every rule from RISK_RULES.md.
"""
import logging
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows — file locking handled via fallback

import config
from utils.logger import log_trade_to_md

logger = logging.getLogger("cryptobot.risk")

# Maximum age for open positions before forced close (hours)
MAX_POSITION_AGE_HOURS = 168  # 7 days default

# Trailing stop parameters
TRAILING_STOP_ACTIVATION_PCT = 1.0   # Activate trailing stop once 1% in profit
TRAILING_STOP_DISTANCE_PCT = 2.5     # Trail 2.5% below high water mark (buys) / above low water mark (sells)


class RiskManager:
    """Tracks P&L, enforces limits, triggers alerts."""

    def __init__(self, state_file: str = config.STATE_FILE):
        self._state_file = state_file
        self._state = self._load_state()

    # ── State persistence ────────────────────────────────────────────────

    def _default_state(self) -> dict:
        return {
            "balance": config.INITIAL_BALANCE,
            "peak_balance": config.INITIAL_BALANCE,
            "positions": [],  # list of open position dicts
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "daily_reset": datetime.utcnow().strftime("%Y-%m-%d"),
            "weekly_reset": datetime.utcnow().strftime("%Y-%m-%d"),
            "paused": False,
            "pause_reason": "",
            "trades_today": 0,
        }

    def _load_state(self) -> dict:
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r") as f:
                    state = json.load(f)
                # Reset daily/weekly counters if needed
                self._maybe_reset_counters(state)
                return state
            except Exception:
                logger.warning("Corrupt state file, starting fresh")
        return self._default_state()

    def save_state(self):
        """Atomically save state. Uses atomic rename to prevent partial writes."""
        tmp_file = self._state_file + ".tmp"
        try:
            with open(tmp_file, "w") as f:
                if fcntl:
                    fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(self._state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
                if fcntl:
                    fcntl.flock(f, fcntl.LOCK_UN)
            os.replace(tmp_file, self._state_file)
        except Exception:
            with open(self._state_file, "w") as f:
                json.dump(self._state, f, indent=2)

    def reload_state(self):
        """Re-read state from disk. Call before SL/TP checks to pick up
        changes made by other processes (e.g. duplicate bot instances)."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r") as f:
                disk_state = json.load(f)
            # Merge: keep the union of positions from disk and memory.
            # A position closed in either copy stays closed.
            mem_positions = {p["id"]: p for p in self._state.get("positions", [])}
            disk_positions = {p["id"]: p for p in disk_state.get("positions", [])}
            merged = {}
            for pid in set(mem_positions) | set(disk_positions):
                mem_p = mem_positions.get(pid)
                disk_p = disk_positions.get(pid)
                if mem_p and disk_p:
                    # If either copy shows closed, it's closed
                    if mem_p["status"] == "closed" or disk_p["status"] == "closed":
                        merged[pid] = mem_p if mem_p["status"] == "closed" else disk_p
                    else:
                        merged[pid] = disk_p  # prefer disk for open positions
                elif mem_p:
                    merged[pid] = mem_p
                else:
                    merged[pid] = disk_p
            self._state["positions"] = list(merged.values())
            # Take the max balance (don't regress from closed-trade gains)
            self._state["balance"] = max(
                self._state.get("balance", config.INITIAL_BALANCE),
                disk_state.get("balance", config.INITIAL_BALANCE),
            )
            self._state["peak_balance"] = max(
                self._state.get("peak_balance", config.INITIAL_BALANCE),
                disk_state.get("peak_balance", config.INITIAL_BALANCE),
            )
        except Exception as e:
            logger.warning("State reload failed: %s", e)

    def _maybe_reset_counters(self, state: dict):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if state.get("daily_reset") != today:
            state["daily_pnl"] = 0.0
            state["daily_reset"] = today
            state["trades_today"] = 0
        # Weekly reset on Monday
        now = datetime.utcnow()
        last_reset = datetime.strptime(state.get("weekly_reset", today), "%Y-%m-%d")
        if (now - last_reset).days >= 7:
            state["weekly_pnl"] = 0.0
            state["weekly_reset"] = today

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def balance(self) -> float:
        return self._state["balance"]

    @property
    def positions(self) -> list:
        return self._state["positions"]

    @property
    def is_paused(self) -> bool:
        return self._state.get("paused", False)

    @property
    def pause_reason(self) -> str:
        return self._state.get("pause_reason", "")

    def can_open_without_conflict(self, side: str, strategy: str) -> tuple[bool, str]:
        """Portfolio-level position arbiter: prevent opposing positions across strategies.

        If one strategy has a BUY open and another wants to SELL (or vice versa),
        the trades cancel each other out. Block the conflicting trade.
        """
        opposite = "sell" if side == "buy" else "buy"
        open_positions = [p for p in self._state["positions"] if p["status"] == "open"]

        for pos in open_positions:
            if pos["strategy"] == strategy:
                continue  # same strategy manages its own positions
            if pos["side"] == opposite:
                return False, (
                    f"Position conflict: {pos['strategy']} has an open {pos['side'].upper()} "
                    f"— cannot open {side.upper()} from {strategy}"
                )
        return True, "OK"

    def can_open_position(self, price: float, side: str = None, strategy: str = None) -> tuple[bool, str]:
        """Check ALL risk rules before opening a position.

        Args:
            price: Current market price.
            side: 'buy' or 'sell' — used for position conflict check.
            strategy: Strategy name — used for position conflict check.
        """
        if self.is_paused:
            return False, f"Trading paused: {self.pause_reason}"

        # Position arbiter: check for opposing positions across strategies
        if side and strategy:
            can_open, reason = self.can_open_without_conflict(side, strategy)
            if not can_open:
                logger.warning("ARBITER BLOCKED: %s", reason)
                return False, reason

        # Max concurrent positions
        open_count = len([p for p in self._state["positions"] if p["status"] == "open"])
        if open_count >= config.MAX_CONCURRENT_POSITIONS:
            return False, f"Max {config.MAX_CONCURRENT_POSITIONS} concurrent positions"

        # Daily loss limit (only cap losses, not gains)
        if self._state["daily_pnl"] <= -(self.balance * (config.DAILY_MAX_LOSS_PCT / 100)):
            return False, "Daily loss limit reached"

        # Weekly loss limit (only cap losses, not gains)
        if self._state["weekly_pnl"] <= -(self.balance * (config.WEEKLY_MAX_LOSS_PCT / 100)):
            return False, "Weekly loss limit reached"

        # Drawdown check
        drawdown = (self._state["peak_balance"] - self.balance) / self._state["peak_balance"] * 100
        if drawdown >= config.DRAWDOWN_PAUSE_PCT:
            self._state["paused"] = True
            self._state["pause_reason"] = f"Drawdown {drawdown:.1f}% exceeds {config.DRAWDOWN_PAUSE_PCT}%"
            self.save_state()
            return False, self._state["pause_reason"]

        return True, "OK"

    def max_position_size_usd(self) -> float:
        """Max USD per trade = 2% of balance."""
        return self.balance * (config.MAX_RISK_PER_TRADE_PCT / 100)

    def position_size_btc(self, price: float) -> float:
        """How much BTC we can buy with max allowed USD."""
        max_usd = self.max_position_size_usd()
        return max_usd / price

    def open_position(self, side: str, price: float, size_btc: float,
                      strategy: str, stop_loss: float, take_profit: float) -> dict:
        """Record a new paper position."""
        pos = {
            "id": f"{strategy}-{int(time.time())}-{uuid.uuid4().hex[:6]}",
            "side": side,
            "entry_price": price,
            "size_btc": size_btc,
            "size_usd": price * size_btc,
            "strategy": strategy,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "open",
            "opened_at": datetime.utcnow().isoformat(),
            "closed_at": None,
            "exit_price": None,
            "pnl": 0.0,
        }
        self._state["positions"].append(pos)
        self._state["trades_today"] += 1
        self.save_state()
        log_trade_to_md(pos)
        logger.info("OPENED %s %s %.6f BTC @ $%.2f [%s] SL=$%.2f TP=$%.2f",
                     side, config.PAIR_DISPLAY, size_btc, price, strategy, stop_loss, take_profit)
        return pos

    def close_position(self, pos_id: str, exit_price: float) -> Optional[dict]:
        """Close a position, compute P&L, update balance."""
        for pos in self._state["positions"]:
            if pos["id"] == pos_id and pos["status"] == "open":
                pos["status"] = "closed"
                pos["exit_price"] = exit_price
                pos["closed_at"] = datetime.utcnow().isoformat()

                if pos["side"] == "buy":
                    pos["pnl"] = (exit_price - pos["entry_price"]) * pos["size_btc"]
                else:
                    pos["pnl"] = (pos["entry_price"] - exit_price) * pos["size_btc"]

                self._state["balance"] += pos["pnl"]
                self._state["daily_pnl"] += pos["pnl"]
                self._state["weekly_pnl"] += pos["pnl"]

                # Update peak
                if self._state["balance"] > self._state["peak_balance"]:
                    self._state["peak_balance"] = self._state["balance"]

                self.save_state()
                log_trade_to_md(pos)
                logger.info("CLOSED %s @ $%.2f | P&L: $%.2f | Balance: $%.2f",
                            pos_id, exit_price, pos["pnl"], self.balance)
                return pos
        return None

    def _update_trailing_stop(self, pos: dict, current_price: float):
        """Update high water mark and trailing stop for a position.

        Trailing stop activates once position is TRAILING_STOP_ACTIVATION_PCT in profit.
        Once active, stop follows TRAILING_STOP_DISTANCE_PCT behind the high water mark.
        The trailing stop only moves in the favorable direction (ratchets tighter).
        """
        entry = pos["entry_price"]

        if pos["side"] == "buy":
            profit_pct = (current_price - entry) / entry * 100
            # Update high water mark
            hwm = pos.get("high_water_mark", entry)
            if current_price > hwm:
                pos["high_water_mark"] = current_price
                hwm = current_price
            # Activate trailing stop once profit threshold is met
            if profit_pct >= TRAILING_STOP_ACTIVATION_PCT:
                trailing_sl = hwm * (1 - TRAILING_STOP_DISTANCE_PCT / 100)
                # Only ratchet up — never lower the stop
                if trailing_sl > pos["stop_loss"]:
                    old_sl = pos["stop_loss"]
                    pos["stop_loss"] = trailing_sl
                    pos["trailing_stop_active"] = True
                    logger.info("TRAILING STOP updated %s: SL $%.2f → $%.2f (HWM=$%.2f, profit=%.1f%%)",
                                pos["id"], old_sl, trailing_sl, hwm, profit_pct)
                elif not pos.get("trailing_stop_active"):
                    pos["trailing_stop_active"] = True
                    logger.info("TRAILING STOP activated for %s (profit=%.1f%%, HWM=$%.2f)",
                                pos["id"], profit_pct, hwm)
        else:  # sell/short
            profit_pct = (entry - current_price) / entry * 100
            # Low water mark for shorts
            lwm = pos.get("high_water_mark", entry)  # reuse field name for consistency
            if current_price < lwm:
                pos["high_water_mark"] = current_price
                lwm = current_price
            if profit_pct >= TRAILING_STOP_ACTIVATION_PCT:
                trailing_sl = lwm * (1 + TRAILING_STOP_DISTANCE_PCT / 100)
                # Only ratchet down — never raise the stop for shorts
                if trailing_sl < pos["stop_loss"]:
                    old_sl = pos["stop_loss"]
                    pos["stop_loss"] = trailing_sl
                    pos["trailing_stop_active"] = True
                    logger.info("TRAILING STOP updated %s: SL $%.2f → $%.2f (LWM=$%.2f, profit=%.1f%%)",
                                pos["id"], old_sl, trailing_sl, lwm, profit_pct)
                elif not pos.get("trailing_stop_active"):
                    pos["trailing_stop_active"] = True
                    logger.info("TRAILING STOP activated for %s (profit=%.1f%%, LWM=$%.2f)",
                                pos["id"], profit_pct, lwm)

    def check_stop_loss_take_profit(self, current_price: float) -> list:
        """Check all open positions for SL/TP hits, trailing stops, and max-age timeout.
        Reloads state from disk first to handle multi-process scenarios.
        Returns list of closed positions."""
        # Reload from disk to catch changes from other bot instances
        self.reload_state()

        closed = []
        now = datetime.utcnow()
        open_positions = [p for p in self._state["positions"] if p["status"] == "open"]

        for pos in open_positions:
            pos_id = pos["id"]

            # Update trailing stop before checking SL/TP
            self._update_trailing_stop(pos, current_price)

            # Log position status each tick for debugging
            trailing_tag = " [TRAILING]" if pos.get("trailing_stop_active") else ""
            if pos["side"] == "buy":
                dist_tp = (pos["take_profit"] - current_price) / current_price * 100
                dist_sl = (current_price - pos["stop_loss"]) / current_price * 100
            else:
                dist_tp = (current_price - pos["take_profit"]) / current_price * 100
                dist_sl = (pos["stop_loss"] - current_price) / current_price * 100
            logger.debug("POS %s [%s %s]%s entry=$%.2f price=$%.2f TP=$%.2f(%.2f%%) SL=$%.2f(%.2f%%)",
                         pos_id, pos["side"], pos["strategy"], trailing_tag,
                         pos["entry_price"], current_price,
                         pos["take_profit"], dist_tp,
                         pos["stop_loss"], dist_sl)

            # Max-age timeout: force-close stale positions
            if pos.get("opened_at"):
                try:
                    opened = datetime.fromisoformat(pos["opened_at"])
                    age_hours = (now - opened).total_seconds() / 3600
                    if age_hours >= MAX_POSITION_AGE_HOURS:
                        logger.warning("MAX-AGE hit for %s (%.1fh old) @ $%.2f", pos_id, age_hours, current_price)
                        result = self.close_position(pos_id, current_price)
                        if result:
                            closed.append(result)
                        continue
                except (ValueError, TypeError):
                    pass

            if pos["side"] == "buy":
                if current_price <= pos["stop_loss"]:
                    sl_type = "TRAILING STOP" if pos.get("trailing_stop_active") else "STOP-LOSS"
                    logger.warning("%s hit for %s @ $%.2f (SL=$%.2f)", sl_type, pos_id, current_price, pos["stop_loss"])
                    result = self.close_position(pos_id, current_price)
                    if result:
                        closed.append(result)
                elif current_price >= pos["take_profit"]:
                    logger.info("TAKE-PROFIT hit for %s @ $%.2f (TP=$%.2f)", pos_id, current_price, pos["take_profit"])
                    result = self.close_position(pos_id, current_price)
                    if result:
                        closed.append(result)
            else:  # sell/short (paper only)
                if current_price >= pos["stop_loss"]:
                    sl_type = "TRAILING STOP" if pos.get("trailing_stop_active") else "STOP-LOSS"
                    logger.warning("%s hit for %s @ $%.2f (SL=$%.2f)", sl_type, pos_id, current_price, pos["stop_loss"])
                    result = self.close_position(pos_id, current_price)
                    if result:
                        closed.append(result)
                elif current_price <= pos["take_profit"]:
                    logger.info("TAKE-PROFIT hit for %s @ $%.2f (TP=$%.2f)", pos_id, current_price, pos["take_profit"])
                    result = self.close_position(pos_id, current_price)
                    if result:
                        closed.append(result)

        if closed:
            logger.info("Closed %d positions this tick", len(closed))
        # Persist trailing stop state changes
        if open_positions:
            self.save_state()

        return closed

    def get_daily_summary(self) -> dict:
        """Generate daily performance summary."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        today_trades = [p for p in self._state["positions"]
                        if p.get("opened_at", "").startswith(today)]
        closed_today = [p for p in today_trades if p["status"] == "closed"]
        wins = [p for p in closed_today if p["pnl"] > 0]
        losses = [p for p in closed_today if p["pnl"] < 0]

        return {
            "date": today,
            "balance": self.balance,
            "daily_pnl": self._state["daily_pnl"],
            "trades_opened": len(today_trades),
            "trades_closed": len(closed_today),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed_today) * 100 if closed_today else 0,
            "open_positions": len([p for p in self._state["positions"] if p["status"] == "open"]),
            "drawdown_pct": (self._state["peak_balance"] - self.balance) / self._state["peak_balance"] * 100,
            "paused": self.is_paused,
        }
