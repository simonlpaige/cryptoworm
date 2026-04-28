"""
Strategy 1: Grid Bot
- Sets buy/sell grid levels ±10% around a reference price
- Buys at lower grid levels, sells at upper ones
- Passive strategy for sideways markets
- Grid spacing scales with realized volatility (when GRID_VOL_SCALING enabled)
"""
import logging
import math
import time
from typing import Optional, List

import config
from utils.risk_manager import RiskManager
from utils.kraken_client import KrakenClient
logger = logging.getLogger("cryptoworm.grid")

# Baseline daily volatility for BTC (typical ~2%)
_BASELINE_VOL = 0.02


class GridBot:
    """Paper-trading grid bot."""

    def __init__(self, kraken: KrakenClient, risk: RiskManager):
        self.kraken = kraken
        self.risk = risk
        self.grid_levels: List[float] = []
        self.reference_price: Optional[float] = None
        self.filled_buys: set = set()   # grid levels where we bought
        self.filled_sells: set = set()  # grid levels where we sold
        self._initialized = False

    def _calc_realized_vol(self) -> Optional[float]:
        """Calculate realized volatility from last 24h of 1h OHLC data."""
        try:
            ohlc = self.kraken.get_ohlc(interval=60, count=24)
            if not ohlc or len(ohlc) < 10:
                return None
            closes = [c["close"] for c in ohlc]
            log_rets = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0 and closes[i] > 0:
                    log_rets.append(math.log(closes[i] / closes[i - 1]))
            if len(log_rets) < 5:
                return None
            mean = sum(log_rets) / len(log_rets)
            variance = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
            # Annualize from hourly: sqrt(24) to get daily
            hourly_vol = math.sqrt(variance)
            daily_vol = hourly_vol * math.sqrt(24)
            return daily_vol
        except Exception as e:
            logger.debug("Realized vol calculation failed: %s", e)
            return None

    def _vol_scaled_range_pct(self) -> float:
        """Compute grid range % scaled by realized volatility.

        Grid spacing = base_spacing * (realized_vol / baseline_vol)
        In low vol: tighter grids (more trades, smaller profits)
        In high vol: wider grids (fewer trades, bigger profits)
        """
        if not getattr(config, "GRID_VOL_SCALING", False):
            return config.GRID_RANGE_PCT

        realized_vol = self._calc_realized_vol()
        if realized_vol is None or realized_vol <= 0:
            logger.debug("Vol scaling: using default range (no vol data)")
            return config.GRID_RANGE_PCT

        vol_ratio = realized_vol / _BASELINE_VOL
        # Clamp scaling between 0.5x and 3x to avoid extreme grids
        vol_ratio = max(0.5, min(3.0, vol_ratio))
        scaled = config.GRID_RANGE_PCT * vol_ratio

        logger.info("Grid vol scaling: realized_vol=%.3f%%, baseline=%.1f%%, ratio=%.2f, range=%.1f%%",
                     realized_vol * 100, _BASELINE_VOL * 100, vol_ratio, scaled)
        return scaled

    def initialize(self, current_price: float):
        """Set up grid around current price. Uses volatility-scaled spacing if enabled."""
        self.reference_price = current_price
        range_pct = self._vol_scaled_range_pct()
        low = current_price * (1 - range_pct / 100)
        high = current_price * (1 + range_pct / 100)
        step = (high - low) / config.GRID_LEVELS

        self.grid_levels = [round(low + i * step, 2) for i in range(config.GRID_LEVELS + 1)]
        self._initialized = True

        # Separate into buy (below price) and sell (above price) levels
        buy_levels = [l for l in self.grid_levels if l < current_price]
        sell_levels = [l for l in self.grid_levels if l > current_price]

        logger.info("Grid initialized: ref=$%.2f, range=$%.2f-$%.2f (%.1f%%), %d levels",
                     current_price, low, high, range_pct, len(self.grid_levels))
        logger.info("  Buy levels: %s", [f"${l:,.2f}" for l in buy_levels[:5]])
        logger.info("  Sell levels: %s", [f"${l:,.2f}" for l in sell_levels[:5]])

    def evaluate(self, current_price: float) -> list:
        """Check if price has crossed any grid levels. Returns list of actions taken."""
        if not self._initialized:
            self.initialize(current_price)
            return []

        actions = []

        # Check buy levels (price dropped to a grid level below current price)
        for level in self.grid_levels:
            if level <= current_price and level < self.reference_price:
                level_key = f"buy-{level}"
                if level_key not in self.filled_buys:
                    action = self._try_grid_buy(current_price, level)
                    if action:
                        self.filled_buys.add(level_key)
                        actions.append(action)

            # Check sell levels (price rose to a grid level above current price)
            elif level >= current_price and level > self.reference_price:
                level_key = f"sell-{level}"
                if level_key not in self.filled_sells:
                    action = self._try_grid_sell(current_price, level)
                    if action:
                        self.filled_sells.add(level_key)
                        actions.append(action)

        return actions

    def _count_open_grid_positions(self) -> int:
        """Count currently open grid positions."""
        return sum(1 for p in self.risk.positions
                   if p["status"] == "open" and p["strategy"] == "grid")

    def _try_grid_buy(self, price: float, level: float) -> Optional[dict]:
        """Attempt a grid buy. Also closes any open grid shorts at profit."""
        # Close any profitable grid shorts first
        for pos in list(self.risk.positions):
            if pos["status"] == "open" and pos["strategy"] == "grid" and pos["side"] == "sell":
                if price <= pos["take_profit"]:
                    result = self.risk.close_position(pos["id"], price)
                    if result:
                        logger.info("Grid short covered at $%.2f, P&L=$%.2f", price, result["pnl"])

        # Respect max grid positions to leave room for other strategies
        max_grid = getattr(config, "MAX_GRID_POSITIONS", 2)
        if self._count_open_grid_positions() >= max_grid:
            logger.debug("Grid buy blocked: %d/%d grid positions open", self._count_open_grid_positions(), max_grid)
            return None

        can_open, reason = self.risk.can_open_position(price, side="buy", strategy="grid")
        if not can_open:
            logger.info("Grid buy blocked at $%.2f: %s", price, reason)
            return None

        size_btc = self.risk.position_size_btc(price)
        stop_loss = price * (1 - config.SWING_STOP_LOSS_PCT / 100)
        # Take profit at the next grid level up
        tp_candidates = [l for l in self.grid_levels if l > price]
        take_profit = tp_candidates[0] if tp_candidates else price * 1.05

        pos = self.risk.open_position(
            side="buy",
            price=price,
            size_btc=size_btc,
            strategy="grid",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        return pos

    def _try_grid_sell(self, price: float, level: float) -> Optional[dict]:
        """Close any open grid buy positions at profit, then open a short."""
        closed = []
        for pos in self.risk.positions:
            if pos["status"] == "open" and pos["strategy"] == "grid" and pos["side"] == "buy":
                if price >= pos["take_profit"]:
                    result = self.risk.close_position(pos["id"], price)
                    if result:
                        closed.append(result)

        # Also open a short at upper grid levels
        if not closed:  # only short if we didn't just close a long (avoid doubling up)
            action = self._try_grid_short(price, level)
            if action:
                return action

        return closed[0] if closed else None

    def _try_grid_short(self, price: float, level: float) -> Optional[dict]:
        """Open a grid short position at upper grid levels."""
        # Don't short if we already have an open grid short
        open_shorts = [p for p in self.risk.positions
                       if p["status"] == "open" and p["strategy"] == "grid" and p["side"] == "sell"]
        if open_shorts:
            return None

        # Respect max grid positions to leave room for other strategies
        max_grid = getattr(config, "MAX_GRID_POSITIONS", 2)
        if self._count_open_grid_positions() >= max_grid:
            logger.debug("Grid short blocked: %d/%d grid positions open", self._count_open_grid_positions(), max_grid)
            return None

        can_open, reason = self.risk.can_open_position(price, side="sell", strategy="grid")
        if not can_open:
            logger.info("Grid short blocked at $%.2f: %s", price, reason)
            return None

        size_btc = self.risk.position_size_btc(price)
        stop_loss = price * (1 + config.SWING_STOP_LOSS_PCT / 100)  # SL above entry
        # Take profit at the next grid level down
        tp_candidates = [l for l in sorted(self.grid_levels, reverse=True) if l < price]
        take_profit = tp_candidates[0] if tp_candidates else price * 0.95

        pos = self.risk.open_position(
            side="sell",
            price=price,
            size_btc=size_btc,
            strategy="grid",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        logger.info("Grid SHORT: price=$%.2f, level=$%.2f, size=%.6f BTC",
                     price, level, size_btc)
        return pos

    def should_reinitialize(self, current_price: float) -> bool:
        """Re-center grid if price moved >25% from reference.
        Raised from 15% to prevent excessive grid resets in volatile markets.
        """
        if not self.reference_price:
            return True
        pct_move = abs(current_price - self.reference_price) / self.reference_price * 100
        return pct_move > 25
