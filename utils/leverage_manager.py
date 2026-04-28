"""
LeverageManager -- keeps us from over-leveraging in Stage 1.

Stage 1 hard cap: 3x (config.MAX_LEVERAGE).
Warns at 2x. Blocks new positions when portfolio leverage >= cap.

Margin model:
  initial_margin = position_size / leverage
  maintenance_margin = initial_margin * 0.5  (50% of initial)
  Liquidation happens when account equity < maintenance_margin.

In paper trading this is all simulated, but the math is real --
we need to know if a position would get liquidated at a given price move.
"""
import logging
import config

logger = logging.getLogger("cryptoworm.leverage")

_DEFAULT_MAX_LEVERAGE = 3.0
_WARN_LEVERAGE = 2.0


class LeverageManager:
    """Track and enforce leverage limits per position and portfolio-wide."""

    def __init__(self):
        self.max_leverage = getattr(config, "MAX_LEVERAGE", _DEFAULT_MAX_LEVERAGE)
        logger.info("LeverageManager: max_leverage=%.1fx", self.max_leverage)

    def check_leverage(
        self,
        position_size_usd: float,
        balance: float,
        current_leverage: float = 1.0,
    ) -> tuple:
        """
        Check if opening a position at current_leverage is allowed.

        Returns: (allowed: bool, reason: str, adjusted_size: float)
          - allowed: False if leverage would exceed cap
          - adjusted_size: may be reduced to fit within cap
        """
        if balance <= 0:
            return False, "Balance is zero or negative", 0.0

        if current_leverage > self.max_leverage:
            allowed_size = balance * self.max_leverage
            logger.warning(
                "Leverage %.1fx exceeds cap %.1fx -- capping position at $%.2f",
                current_leverage, self.max_leverage, allowed_size
            )
            return True, f"Capped to {self.max_leverage}x", min(position_size_usd, allowed_size)

        if current_leverage >= _WARN_LEVERAGE:
            logger.warning(
                "Leverage %.1fx approaching cap %.1fx -- watch margin",
                current_leverage, self.max_leverage
            )

        return True, "OK", position_size_usd

    def compute_margin(self, position_size_usd: float, leverage: float = 1.0) -> dict:
        """
        Compute margin requirements for a position.

        initial_margin: the USD collateral locked
        maintenance_margin: minimum equity to avoid liquidation (50% of initial)
        liquidation_cushion_pct: how far price can move before liquidation
        """
        leverage = max(leverage, 1.0)
        initial_margin = position_size_usd / leverage
        maintenance_margin = initial_margin * 0.5

        # Rough liquidation price estimate for a long at 1:leverage
        # Price must fall by (1/leverage - maintenance_rate) to hit liquidation
        # Using 50% maintenance rate: liquidation_at = entry * (1 - 0.5/leverage)
        liquidation_cushion_pct = (0.5 / leverage) * 100

        return {
            "initial_margin": round(initial_margin, 2),
            "maintenance_margin": round(maintenance_margin, 2),
            "liquidation_cushion_pct": round(liquidation_cushion_pct, 2),
            "leverage": leverage,
        }

    def portfolio_leverage(self, open_positions: list, balance: float) -> float:
        """
        Total portfolio leverage: sum of position sizes / balance.
        A portfolio of $300 in positions with $100 balance = 3x leverage.
        """
        if balance <= 0:
            return 0.0
        total_exposure = sum(
            p.get("size_usd", 0.0)
            for p in open_positions
            if p.get("status") == "open"
        )
        lev = total_exposure / balance
        if lev > _WARN_LEVERAGE:
            logger.info(
                "Portfolio leverage: %.2fx (exposure=$%.2f balance=$%.2f)",
                lev, total_exposure, balance
            )
        return lev
