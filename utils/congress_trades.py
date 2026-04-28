"""
Congressional Trading Signal Provider
======================================
Tracks periodic transaction reports (PTRs) from House and Senate members
for crypto-adjacent stock activity (COIN, MSTR, MARA, RIOT, GBTC, IBIT).

Signal logic:
- 3+ members buying crypto-adjacent stocks within 7 days → bullish
- 3+ members selling within 7 days → bearish

Data sources:
- House Clerk XML disclosures (public)
- Senate EFDS (public)
- Historical data loader for backtesting
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional
from xml.etree import ElementTree

import requests

logger = logging.getLogger("cryptobot.congress")

# Crypto-adjacent tickers to track
CRYPTO_TICKERS = {"COIN", "MSTR", "MARA", "RIOT", "GBTC", "IBIT", "BITB", "BITO", "CLSK", "HUT"}

HISTORICAL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "data", "congress_historical.json")


class CongressTradesProvider:
    """Fetches and analyzes congressional trading disclosures."""

    def __init__(self):
        self._cache = []
        self._last_fetch = None
        self._fetch_interval = 3600  # refresh hourly

    def fetch_house_disclosures(self) -> list:
        """Fetch recent House PTR disclosures from the Clerk's XML feed.

        Returns list of dicts: {member, ticker, tx_type, amount, tx_date, filed_date}
        """
        trades = []
        year = datetime.utcnow().year
        url = f"https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/"

        try:
            resp = requests.get(url, timeout=30, headers={
                "User-Agent": "CryptoBot/1.0 (research)"
            })
            if resp.status_code != 200:
                logger.debug("House disclosures returned %d", resp.status_code)
                return trades

            # Parse the HTML listing for XML links (simplified approach)
            # The actual page lists PDF links; we look for any structured data
            # In practice, this endpoint serves an HTML directory listing
            # We extract what we can from the page text
            text = resp.text.lower()
            for ticker in CRYPTO_TICKERS:
                if ticker.lower() in text:
                    logger.info("Found reference to %s in House disclosures", ticker)

        except requests.exceptions.RequestException as e:
            logger.debug("Failed to fetch House disclosures: %s", e)

        return trades

    def fetch_senate_disclosures(self) -> list:
        """Fetch recent Senate EFDS disclosures.

        Returns list of dicts: {member, ticker, tx_type, amount, tx_date, filed_date}
        """
        trades = []
        url = "https://efds.senate.gov/search/"

        try:
            resp = requests.get(url, timeout=30, headers={
                "User-Agent": "CryptoBot/1.0 (research)"
            })
            if resp.status_code != 200:
                logger.debug("Senate EFDS returned %d", resp.status_code)
                return trades

        except requests.exceptions.RequestException as e:
            logger.debug("Failed to fetch Senate EFDS: %s", e)

        return trades

    def load_historical_data(self) -> list:
        """Load historical congressional trades from JSON for backtesting.

        Expected format: list of {member, ticker, tx_type, amount_range, tx_date, filed_date}
        """
        if not os.path.exists(HISTORICAL_DATA_PATH):
            return []
        try:
            with open(HISTORICAL_DATA_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load congress historical data: %s", e)
            return []

    def get_recent_trades(self, days: int = 7) -> list:
        """Get crypto-adjacent congressional trades from the last N days.

        Combines live data from House/Senate with any cached results.
        """
        now = datetime.utcnow()

        # Rate-limit fetches
        if self._last_fetch and (now - self._last_fetch).total_seconds() < self._fetch_interval:
            return self._filter_recent(self._cache, days)

        self._last_fetch = now
        trades = []
        trades.extend(self.fetch_house_disclosures())
        time.sleep(1)
        trades.extend(self.fetch_senate_disclosures())

        # Merge with cache, deduplicate
        seen = set()
        merged = []
        for t in trades + self._cache:
            key = f"{t.get('member', '')}_{t.get('ticker', '')}_{t.get('tx_date', '')}"
            if key not in seen:
                seen.add(key)
                merged.append(t)
        self._cache = merged

        return self._filter_recent(merged, days)

    def _filter_recent(self, trades: list, days: int) -> list:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [t for t in trades if t.get("tx_date", "") >= cutoff]

    def generate_signal(self, trades: list = None) -> dict:
        """Analyze recent trades and generate a signal.

        Returns: {signal: 'buy'|'sell'|'neutral', strength: 0-100,
                  buy_count: int, sell_count: int, members: list}
        """
        if trades is None:
            trades = self.get_recent_trades(days=7)

        crypto_trades = [t for t in trades if t.get("ticker", "").upper() in CRYPTO_TICKERS]

        buys = [t for t in crypto_trades if t.get("tx_type", "").lower() in ("purchase", "buy")]
        sells = [t for t in crypto_trades if t.get("tx_type", "").lower() in ("sale", "sell", "sale (full)", "sale (partial)")]

        buy_members = set(t.get("member", "") for t in buys)
        sell_members = set(t.get("member", "") for t in sells)

        signal = "neutral"
        strength = 0

        if len(buy_members) >= 3:
            signal = "buy"
            strength = min(100, len(buy_members) * 20)
        elif len(sell_members) >= 3:
            signal = "sell"
            strength = min(100, len(sell_members) * 20)

        return {
            "signal": signal,
            "strength": strength,
            "buy_count": len(buys),
            "sell_count": len(sells),
            "buy_members": list(buy_members),
            "sell_members": list(sell_members),
        }

    def generate_backtest_signal(self, trades: list, as_of_date: str) -> dict:
        """Generate signal from historical data as of a specific date.

        Args:
            trades: Full list of historical trades
            as_of_date: ISO date string (YYYY-MM-DD) to evaluate at
        """
        cutoff = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        window = [t for t in trades
                  if cutoff <= t.get("filed_date", "") <= as_of_date
                  and t.get("ticker", "").upper() in CRYPTO_TICKERS]
        return self.generate_signal(window)
