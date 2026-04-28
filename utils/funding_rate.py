"""
Funding Rate Monitor
====================
Fetches BTC perpetual funding rates from free public APIs.
Tracks history, calculates averages and trends.
Informational for paper trading (feeds into ML model as feature).
"""
import logging
import time
from typing import Optional, Dict, Any, List
from collections import deque

import requests

logger = logging.getLogger("cryptobot.funding_rate")

# Funding rate history: (timestamp, rate) tuples
_MAX_HISTORY = 21 * 3  # 7 days * 3 funding events per day


class FundingRateMonitor:
    """Monitors BTC perpetual funding rates from public APIs."""

    def __init__(self):
        self._history: deque = deque(maxlen=_MAX_HISTORY)
        self._last_fetch_time: float = 0.0
        self._fetch_interval: float = 3600.0  # fetch at most once per hour
        self._current_rate: Optional[float] = None

    def update(self) -> Dict[str, Any]:
        """Fetch latest funding rate and return summary.

        Returns dict with: current_rate, avg_rate, trend, extreme_positive,
        extreme_negative. All values may be None if API unavailable.
        """
        now = time.time()
        if now - self._last_fetch_time >= self._fetch_interval:
            self._fetch_funding_rate()
            self._last_fetch_time = now

        return self.get_summary()

    def _fetch_funding_rate(self):
        """Try multiple public APIs for funding rate data."""
        # Try Binance first (most reliable, public endpoint)
        rate = self._fetch_binance()
        if rate is not None:
            self._current_rate = rate
            self._history.append((time.time(), rate))
            logger.info("Funding rate: %.4f%% (Binance)", rate * 100)
            return

        # Fallback: CoinGlass public
        rate = self._fetch_coinglass()
        if rate is not None:
            self._current_rate = rate
            self._history.append((time.time(), rate))
            logger.info("Funding rate: %.4f%% (CoinGlass)", rate * 100)
            return

        logger.warning("Could not fetch funding rate from any source")

    def _fetch_binance(self) -> Optional[float]:
        """Fetch from Binance Futures public API."""
        try:
            resp = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": "BTCUSDT", "limit": 1},
                timeout=10,
            )
            if resp.status_code == 451:
                logger.debug("Binance funding rate: 451 (geo-blocked)")
                return None
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                return float(data[0]["fundingRate"])
        except Exception as e:
            logger.debug("Binance funding rate fetch failed: %s", e)
        return None

    def _fetch_coinglass(self) -> Optional[float]:
        """Fetch from CoinGlass public API (no key needed for basic data)."""
        try:
            resp = requests.get(
                "https://open-api.coinglass.com/public/v2/funding",
                params={"symbol": "BTC", "time_type": "h8"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                # CoinGlass returns rates per exchange; average them
                rates = []
                for item in data["data"]:
                    r = item.get("rate") or item.get("uMarginRate")
                    if r is not None:
                        rates.append(float(r))
                if rates:
                    return sum(rates) / len(rates)
        except Exception as e:
            logger.debug("CoinGlass funding rate fetch failed: %s", e)
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get current funding rate summary."""
        if not self._history:
            return {
                "current_rate": None,
                "avg_rate": None,
                "trend": None,
                "extreme_positive": False,
                "extreme_negative": False,
            }

        rates = [r for _, r in self._history]
        current = rates[-1] if rates else None
        avg = sum(rates) / len(rates) if rates else None

        # Trend: compare recent avg to older avg
        trend = None
        if len(rates) >= 6:
            recent = sum(rates[-3:]) / 3
            older = sum(rates[-6:-3]) / 3
            trend = recent - older

        # Extreme thresholds: 0.05% = 0.0005
        extreme_pos = current is not None and current > 0.0005
        extreme_neg = current is not None and current < -0.0005

        summary = {
            "current_rate": current,
            "avg_rate": avg,
            "trend": trend,
            "extreme_positive": extreme_pos,
            "extreme_negative": extreme_neg,
        }

        if extreme_pos:
            logger.info("FUNDING RATE EXTREME POSITIVE: %.4f%% — longs paying shorts",
                        current * 100)
        elif extreme_neg:
            logger.info("FUNDING RATE EXTREME NEGATIVE: %.4f%% — shorts paying longs",
                        current * 100)

        return summary

    @property
    def current_rate(self) -> Optional[float]:
        return self._current_rate

    @property
    def avg_rate(self) -> Optional[float]:
        if not self._history:
            return None
        rates = [r for _, r in self._history]
        return sum(rates) / len(rates)

    @property
    def trend(self) -> Optional[float]:
        rates = [r for _, r in self._history]
        if len(rates) < 6:
            return None
        recent = sum(rates[-3:]) / 3
        older = sum(rates[-6:-3]) / 3
        return recent - older
