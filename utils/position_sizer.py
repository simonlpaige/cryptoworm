"""
PositionSizer -- two models for computing how much to bet.

Model 1: Kelly-ATR
  Kelly criterion tells you the theoretically optimal fraction of
  your bankroll to bet given a win rate and payoff ratio. We then
  scale that down by ATR (average true range) -- when the market is
  swinging hard, we bet smaller.

Model 2: Drawdown scaling
  As your daily P&L falls from the peak, we cut size progressively:
    > 5% drawdown  -> 75% of normal size
    > 10% drawdown -> 50% of normal size
    > 15% drawdown -> 0 (halt signal -- triggers trading pause)

The two models run in series: Kelly-ATR first, then drawdown scale.
This means bad days shrink your bets twice -- once from the math,
once from the drawdown guard.
"""
import logging
import math
import config

logger = logging.getLogger("cryptoworm.position_sizer")


class PositionSizer:
    """Compute position sizes using Kelly-ATR and drawdown scaling."""

    def kelly_atr_size(
        self,
        balance: float,
        atr: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = None,
    ) -> float:
        """
        Kelly-ATR position size in USD.

        Basic Kelly fraction: f* = (W * b - L) / b
          where b = avg_win / avg_loss, W = win_rate, L = 1 - win_rate

        Then we scale by:
          - Kelly fraction (conservative, from config.KELLY_ATR_FRACTION)
          - ATR volatility: tighter bet when ATR/price is large

        Returns USD position size, clamped to balance.
        """
        if fraction is None:
            fraction = getattr(config, "KELLY_ATR_FRACTION", 0.25)

        # Need valid inputs
        if avg_loss <= 0 or price <= 0 or win_rate <= 0 or win_rate >= 1:
            logger.warning("Kelly: invalid inputs, using 1%% of balance")
            return balance * 0.01

        b = avg_win / avg_loss  # payoff ratio
        kelly_f = (win_rate * b - (1 - win_rate)) / b

        if kelly_f <= 0:
            logger.info("Kelly fraction negative (%.4f) -- no edge, skip", kelly_f)
            return 0.0

        # Apply conservative fraction
        kelly_f = kelly_f * fraction

        # ATR volatility adjustment: scale down when market is noisy
        # atr_pct is ATR as a fraction of price; high = more volatile
        atr_pct = (atr / price) if atr and price else 0.02
        # Typical BTC ATR is ~1-3%. Normalize: 0.02 = neutral, higher = smaller bet
        atr_scale = min(1.0, 0.02 / max(atr_pct, 0.001))

        size_usd = balance * kelly_f * atr_scale
        size_usd = max(0.0, min(size_usd, balance))

        logger.debug(
            "Kelly-ATR: balance=%.2f kelly_f=%.4f atr_pct=%.4f atr_scale=%.4f -> $%.2f",
            balance, kelly_f, atr_pct, atr_scale, size_usd
        )
        return size_usd

    def drawdown_scaled_size(
        self,
        base_size_usd: float,
        current_balance: float,
        peak_balance: float,
    ) -> float:
        """
        Scale position size down as drawdown from peak grows.
        At 15%+ drawdown returns 0.0 -- caller should trigger a pause.
        """
        if peak_balance <= 0:
            return base_size_usd

        drawdown_pct = (peak_balance - current_balance) / peak_balance * 100

        if drawdown_pct >= 15.0:
            logger.warning(
                "Drawdown %.1f%% >= 15%% -- hard stop signal, returning size=0",
                drawdown_pct
            )
            return 0.0
        elif drawdown_pct >= 10.0:
            scale = 0.50
        elif drawdown_pct >= 5.0:
            scale = 0.75
        else:
            scale = 1.0

        scaled = base_size_usd * scale
        if scale < 1.0:
            logger.info(
                "Drawdown %.1f%% -> scaling position %.0f%% ($%.2f -> $%.2f)",
                drawdown_pct, scale * 100, base_size_usd, scaled
            )
        return scaled

    def compute(
        self,
        balance: float,
        atr: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        peak_balance: float,
    ) -> float:
        """
        Full sizing pipeline: Kelly-ATR -> drawdown scale.
        Returns final USD position size.
        """
        base = self.kelly_atr_size(balance, atr, price, win_rate, avg_win, avg_loss)
        final = self.drawdown_scaled_size(base, balance, peak_balance)
        return final
