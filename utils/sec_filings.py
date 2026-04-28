"""
SEC EDGAR Institutional Filings Signal Provider
=================================================
Tracks 13F filings for Bitcoin/crypto holdings from major institutions.
Monitors whale institutional moves (BlackRock, Fidelity, etc. Bitcoin ETF flows).

Signal logic:
- Large institutional buys → bullish confirmation
- Multiple institutions increasing crypto exposure → strong bullish

Data source: EDGAR full-text search API (free, no key needed)
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger("cryptobot.sec")

# Major institutions to track for crypto/Bitcoin holdings
TRACKED_INSTITUTIONS = {
    "BLACKROCK": "BlackRock",
    "FIDELITY": "Fidelity",
    "ARK INVEST": "ARK Invest",
    "GRAYSCALE": "Grayscale",
    "BITWISE": "Bitwise",
    "VANECK": "VanEck",
    "INVESCO": "Invesco",
    "WISDOMTREE": "WisdomTree",
    "FRANKLIN TEMPLETON": "Franklin Templeton",
    "VALKYRIE": "Valkyrie",
    "GALAXY DIGITAL": "Galaxy Digital",
    "MICROSTRATEGY": "MicroStrategy",
}

# Keywords indicating crypto-related filings
CRYPTO_KEYWORDS = [
    "bitcoin", "BTC", "crypto", "digital asset",
    "IBIT", "GBTC", "FBTC", "ARKB", "BITB",
    "bitcoin trust", "bitcoin etf", "spot bitcoin",
]

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"

HISTORICAL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "data", "sec_historical.json")


class SecFilingsProvider:
    """Tracks SEC EDGAR 13F filings for institutional crypto exposure."""

    def __init__(self):
        self._cache = []
        self._last_fetch = None
        self._fetch_interval = 7200  # refresh every 2 hours

    def search_edgar(self, query: str, form_type: str = "13F",
                     date_range: str = None) -> list:
        """Search EDGAR full-text search for filings matching query.

        Args:
            query: Search terms (e.g., "bitcoin" or "IBIT")
            form_type: SEC form type filter (default: 13F)
            date_range: Date range filter (e.g., "2025-01-01,2025-12-31")

        Returns list of {filer, form_type, filed_date, description}
        """
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": query,
            "forms": form_type,
            "dateRange": "custom",
        }

        if date_range:
            start, end = date_range.split(",")
            params["startdt"] = start.strip()
            params["enddt"] = end.strip()
        else:
            # Default: last 90 days
            end = datetime.utcnow().strftime("%Y-%m-%d")
            start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
            params["startdt"] = start
            params["enddt"] = end

        filings = []
        try:
            resp = requests.get(url, params=params, timeout=30, headers={
                "User-Agent": "CryptoBot/1.0 research@example.com",
                "Accept": "application/json",
            })

            if resp.status_code != 200:
                logger.debug("EDGAR search returned %d for query '%s'", resp.status_code, query)
                return filings

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            for hit in hits[:20]:  # limit to 20 results
                source = hit.get("_source", {})
                filings.append({
                    "filer": source.get("display_names", ["Unknown"])[0] if source.get("display_names") else "Unknown",
                    "form_type": source.get("form_type", form_type),
                    "filed_date": source.get("file_date", ""),
                    "description": source.get("display_date_filed", ""),
                    "file_num": source.get("file_num", ""),
                })

        except requests.exceptions.RequestException as e:
            logger.debug("EDGAR search failed: %s", e)
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("EDGAR response parse error: %s", e)

        return filings

    def fetch_crypto_filings(self) -> list:
        """Fetch recent 13F filings mentioning Bitcoin/crypto from tracked institutions.

        Returns list of {institution, filing_type, date, direction, details}
        """
        now = datetime.utcnow()
        if self._last_fetch and (now - self._last_fetch).total_seconds() < self._fetch_interval:
            return self._cache

        self._last_fetch = now
        all_filings = []

        for keyword in ["bitcoin ETF", "IBIT", "GBTC", "spot bitcoin"]:
            filings = self.search_edgar(keyword, form_type="13F")
            time.sleep(0.5)  # Be respectful to EDGAR

            for f in filings:
                filer_upper = f["filer"].upper()
                institution = None
                for key, name in TRACKED_INSTITUTIONS.items():
                    if key in filer_upper:
                        institution = name
                        break

                if institution:
                    all_filings.append({
                        "institution": institution,
                        "filing_type": f["form_type"],
                        "date": f["filed_date"],
                        "direction": "buy",  # 13F reports holdings; existence = holding
                        "details": f.get("description", ""),
                    })

        # Deduplicate
        seen = set()
        unique = []
        for f in all_filings:
            key = f"{f['institution']}_{f['date']}"
            if key not in seen:
                seen.add(key)
                unique.append(f)

        self._cache = unique
        return unique

    def load_historical_data(self) -> list:
        """Load historical SEC filing data for backtesting."""
        if not os.path.exists(HISTORICAL_DATA_PATH):
            return []
        try:
            with open(HISTORICAL_DATA_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load SEC historical data: %s", e)
            return []

    def generate_signal(self) -> dict:
        """Generate signal from recent institutional filings.

        Returns: {signal: 'buy'|'sell'|'neutral', strength: 0-100,
                  institutions: list, filing_count: int}
        """
        filings = self.fetch_crypto_filings()

        # Filter to last 30 days
        cutoff = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent = [f for f in filings if f.get("date", "") >= cutoff]

        institutions = list(set(f["institution"] for f in recent))
        buy_count = len([f for f in recent if f["direction"] == "buy"])

        signal = "neutral"
        strength = 0

        if buy_count >= 3 or len(institutions) >= 3:
            signal = "buy"
            strength = min(100, len(institutions) * 15 + buy_count * 10)
        elif buy_count >= 1:
            signal = "buy"
            strength = min(40, buy_count * 15)

        return {
            "signal": signal,
            "strength": strength,
            "institutions": institutions,
            "filing_count": len(recent),
        }

    def generate_backtest_signal(self, filings: list, as_of_date: str) -> dict:
        """Generate signal from historical filing data as of a specific date."""
        cutoff = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        recent = [f for f in filings
                  if cutoff <= f.get("date", "") <= as_of_date]

        institutions = list(set(f.get("institution", "") for f in recent))
        buy_count = len([f for f in recent if f.get("direction", "") == "buy"])

        signal = "neutral"
        strength = 0
        if buy_count >= 3 or len(institutions) >= 3:
            signal = "buy"
            strength = min(100, len(institutions) * 15 + buy_count * 10)
        elif buy_count >= 1:
            signal = "buy"
            strength = min(40, buy_count * 15)

        return {
            "signal": signal,
            "strength": strength,
            "institutions": institutions,
            "filing_count": len(recent),
        }
