"""
ExchangeManager -- the chokepoint between paper and real money.

All order placement flows through here. The EXCHANGE_MODE config value
controls what happens:

  'paper'   -> log the order, delegate to RiskManager (no real orders)
  'testnet' -> scaffold for future testnet integration (not yet wired)
  'live'    -> raises RuntimeError (needs explicit enable in config)

Why this exists: the old bot had no separation between paper logic and
real exchange calls. This class makes it impossible to accidentally go
live -- you have to change config.EXCHANGE_MODE to 'live' AND the code
raises unless a future integration is wired up.

The Kraken price-data client (KrakenClient) still handles all market
data. This class handles execution only.
"""
import logging
import config

logger = logging.getLogger("cryptoworm.exchange")


class ExchangeManager:
    """Single place for all order logic. Paper / testnet / live modes."""

    def __init__(self, risk_manager=None):
        self.mode = getattr(config, "EXCHANGE_MODE", "paper")
        self.exchange_name = getattr(config, "EXCHANGE_NAME", "kraken")
        self._risk = risk_manager

        if self.mode == "live":
            raise RuntimeError(
                "Live trading is not enabled. EXCHANGE_MODE='live' requires a "
                "wired exchange integration. Set EXCHANGE_MODE='paper' or 'testnet'."
            )

        logger.info(
            "ExchangeManager initialized: mode=%s exchange=%s",
            self.mode, self.exchange_name
        )

    # -- Order placement --------------------------------------------------

    def place_order(
        self,
        side: str,
        size_btc: float,
        price: float,
        order_type: str = "market",
        strategy: str = "unknown",
        stop_loss: float = None,
        take_profit: float = None,
    ) -> dict:
        """
        Place an order. What 'place' means depends on mode:
          paper   -> records a paper position via RiskManager
          testnet -> logs intent, raises NotImplementedError (scaffold)
          live    -> blocked at __init__

        Returns the position dict (paper) or raises.
        """
        if self.mode == "paper":
            return self._paper_order(side, size_btc, price, strategy, stop_loss, take_profit)
        elif self.mode == "testnet":
            logger.info(
                "TESTNET ORDER (not sent): %s %.6f BTC @ $%.2f [%s]",
                side.upper(), size_btc, price, strategy
            )
            raise NotImplementedError(
                "Testnet integration not yet wired. Add ccxt exchange here."
            )
        else:
            raise RuntimeError("Unexpected EXCHANGE_MODE: %s" % self.mode)

    def _paper_order(self, side, size_btc, price, strategy, stop_loss, take_profit) -> dict:
        """Record a paper trade via the attached RiskManager."""
        if not self._risk:
            raise RuntimeError("ExchangeManager has no RiskManager -- cannot paper trade")

        sl = stop_loss or (
            price * (1 - config.STOP_LOSS_PCT / 100) if side == "buy"
            else price * (1 + config.STOP_LOSS_PCT / 100)
        )
        tp = take_profit or (
            price * (1 + config.SWING_TAKE_PROFIT_PCT / 100) if side == "buy"
            else price * (1 - config.SWING_TAKE_PROFIT_PCT / 100)
        )

        logger.info(
            "PAPER ORDER: %s %.6f BTC @ $%.2f SL=$%.2f TP=$%.2f [%s]",
            side.upper(), size_btc, price, sl, tp, strategy
        )
        return self._risk.open_position(side, price, size_btc, strategy, sl, tp)

    # -- Balance / data ---------------------------------------------------

    def get_balance(self) -> float:
        """Return available balance. In paper mode, comes from RiskManager."""
        if self.mode == "paper" and self._risk:
            return self._risk.balance
        return 0.0

    def get_funding_rate(self, pair: str = "XBTUSD") -> float:
        """
        Return current perpetual funding rate for the pair.
        Funding rate carry trade fires when rate > 0.1%.
        Delegates to FundingRateMonitor (imported lazily to avoid circular dep).
        """
        try:
            from utils.funding_rate import FundingRateMonitor
            monitor = FundingRateMonitor()
            result = monitor.update()
            return result.get("current_rate", 0.0) or 0.0
        except Exception as e:
            logger.warning("get_funding_rate failed: %s", e)
            return 0.0
