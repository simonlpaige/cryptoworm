"""
Federal Reserve / Macro Calendar Signal Provider
==================================================
Tracks FOMC meetings, CPI releases, jobs reports, GDP releases.

Signal logic:
- 24-48h before major announcements → reduce position sizes (vol spike expected)
- After dovish FOMC → bullish for BTC
- After hawkish FOMC → bearish for BTC
- Uses FRED API (free, no key needed) for CPI/unemployment data

Data sources:
- Static FOMC calendar (updated periodically)
- FRED API for economic data series
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger("cryptobot.fed")

# FRED API base (no key required for basic access)
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Key FRED series IDs
FRED_SERIES = {
    "CPI": "CPIAUCSL",           # Consumer Price Index
    "UNEMPLOYMENT": "UNRATE",     # Unemployment Rate
    "FED_FUNDS": "FEDFUNDS",     # Federal Funds Rate
    "GDP": "GDP",                 # Gross Domestic Product
}

# Static macro calendar for 2025-2026 (major events)
# Format: (date, event_type, description)
MACRO_CALENDAR = [
    # 2025 FOMC meetings
    ("2025-01-29", "FOMC", "FOMC Meeting"),
    ("2025-03-19", "FOMC", "FOMC Meeting"),
    ("2025-05-07", "FOMC", "FOMC Meeting"),
    ("2025-06-18", "FOMC", "FOMC Meeting"),
    ("2025-07-30", "FOMC", "FOMC Meeting"),
    ("2025-09-17", "FOMC", "FOMC Meeting"),
    ("2025-10-29", "FOMC", "FOMC Meeting"),
    ("2025-12-10", "FOMC", "FOMC Meeting"),
    # 2026 FOMC meetings (projected)
    ("2026-01-28", "FOMC", "FOMC Meeting"),
    ("2026-03-18", "FOMC", "FOMC Meeting"),
    ("2026-04-29", "FOMC", "FOMC Meeting"),
    ("2026-06-10", "FOMC", "FOMC Meeting"),
    ("2026-07-29", "FOMC", "FOMC Meeting"),
    ("2026-09-16", "FOMC", "FOMC Meeting"),
    ("2026-10-28", "FOMC", "FOMC Meeting"),
    ("2026-12-09", "FOMC", "FOMC Meeting"),
    # 2025 CPI releases (BLS schedule)
    ("2025-01-15", "CPI", "CPI Release"),
    ("2025-02-12", "CPI", "CPI Release"),
    ("2025-03-12", "CPI", "CPI Release"),
    ("2025-04-10", "CPI", "CPI Release"),
    ("2025-05-13", "CPI", "CPI Release"),
    ("2025-06-11", "CPI", "CPI Release"),
    ("2025-07-11", "CPI", "CPI Release"),
    ("2025-08-12", "CPI", "CPI Release"),
    ("2025-09-10", "CPI", "CPI Release"),
    ("2025-10-14", "CPI", "CPI Release"),
    ("2025-11-12", "CPI", "CPI Release"),
    ("2025-12-10", "CPI", "CPI Release"),
    # 2025 Jobs reports (first Friday of month, approx)
    ("2025-01-10", "JOBS", "Non-Farm Payrolls"),
    ("2025-02-07", "JOBS", "Non-Farm Payrolls"),
    ("2025-03-07", "JOBS", "Non-Farm Payrolls"),
    ("2025-04-04", "JOBS", "Non-Farm Payrolls"),
    ("2025-05-02", "JOBS", "Non-Farm Payrolls"),
    ("2025-06-06", "JOBS", "Non-Farm Payrolls"),
    ("2025-07-03", "JOBS", "Non-Farm Payrolls"),
    ("2025-08-01", "JOBS", "Non-Farm Payrolls"),
    ("2025-09-05", "JOBS", "Non-Farm Payrolls"),
    ("2025-10-03", "JOBS", "Non-Farm Payrolls"),
    ("2025-11-07", "JOBS", "Non-Farm Payrolls"),
    ("2025-12-05", "JOBS", "Non-Farm Payrolls"),
    # 2025 GDP releases (BEA schedule, advance estimates)
    ("2025-01-30", "GDP", "GDP Advance Estimate Q4 2024"),
    ("2025-04-30", "GDP", "GDP Advance Estimate Q1 2025"),
    ("2025-07-30", "GDP", "GDP Advance Estimate Q2 2025"),
    ("2025-10-29", "GDP", "GDP Advance Estimate Q3 2025"),
]

HISTORICAL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "data", "fed_historical.json")


class FedSignalProvider:
    """Tracks Federal Reserve and macro calendar events for trading signals."""

    def __init__(self):
        self._fred_cache = {}
        self._last_fred_fetch = {}

    def get_upcoming_events(self, days_ahead: int = 2) -> list:
        """Get macro events within the next N days.

        Returns list of {date, event_type, description, hours_until}
        """
        now = datetime.utcnow()
        cutoff = now + timedelta(days=days_ahead)
        events = []

        for date_str, event_type, desc in MACRO_CALENDAR:
            event_date = datetime.strptime(date_str, "%Y-%m-%d")
            if now <= event_date <= cutoff:
                hours_until = (event_date - now).total_seconds() / 3600
                events.append({
                    "date": date_str,
                    "event_type": event_type,
                    "description": desc,
                    "hours_until": round(hours_until, 1),
                })

        return events

    def get_events_at_date(self, date_str: str, days_ahead: int = 2) -> list:
        """Get macro events within N days of a specific date (for backtesting)."""
        ref = datetime.strptime(date_str, "%Y-%m-%d")
        cutoff = ref + timedelta(days=days_ahead)
        events = []

        for evt_date_str, event_type, desc in MACRO_CALENDAR:
            event_date = datetime.strptime(evt_date_str, "%Y-%m-%d")
            if ref <= event_date <= cutoff:
                hours_until = (event_date - ref).total_seconds() / 3600
                events.append({
                    "date": evt_date_str,
                    "event_type": event_type,
                    "description": desc,
                    "hours_until": round(hours_until, 1),
                })

        return events

    def fetch_fred_series(self, series_id: str, limit: int = 12) -> list:
        """Fetch recent observations from FRED API (no key needed for basic).

        Returns list of {date, value} dicts.
        """
        # Rate-limit FRED fetches (cache for 6 hours)
        now = datetime.utcnow()
        cache_key = series_id
        if cache_key in self._fred_cache:
            cached_at = self._last_fred_fetch.get(cache_key, datetime.min)
            if (now - cached_at).total_seconds() < 21600:
                return self._fred_cache[cache_key]

        url = f"{FRED_BASE_URL}/series/observations"
        try:
            resp = requests.get(url, params={
                "series_id": series_id,
                "sort_order": "desc",
                "limit": limit,
                "file_type": "json",
                "api_key": "DEMO_KEY",  # FRED provides limited access with demo key
            }, timeout=15)

            if resp.status_code != 200:
                logger.debug("FRED API returned %d for %s", resp.status_code, series_id)
                return self._fred_cache.get(cache_key, [])

            data = resp.json()
            observations = []
            for obs in data.get("observations", []):
                try:
                    observations.append({
                        "date": obs["date"],
                        "value": float(obs["value"]) if obs["value"] != "." else None,
                    })
                except (ValueError, KeyError):
                    continue

            self._fred_cache[cache_key] = observations
            self._last_fred_fetch[cache_key] = now
            return observations

        except requests.exceptions.RequestException as e:
            logger.debug("Failed to fetch FRED series %s: %s", series_id, e)
            return self._fred_cache.get(cache_key, [])

    def get_cpi_trend(self) -> Optional[str]:
        """Determine CPI trend from recent data.

        Returns: 'rising', 'falling', 'stable', or None
        """
        observations = self.fetch_fred_series(FRED_SERIES["CPI"], limit=6)
        if len(observations) < 3:
            return None

        values = [o["value"] for o in observations if o["value"] is not None][:3]
        if len(values) < 3:
            return None

        # Most recent first from FRED
        if values[0] > values[1] > values[2]:
            return "rising"
        elif values[0] < values[1] < values[2]:
            return "falling"
        return "stable"

    def get_fed_funds_rate(self) -> Optional[float]:
        """Get current federal funds rate."""
        observations = self.fetch_fred_series(FRED_SERIES["FED_FUNDS"], limit=1)
        if observations and observations[0]["value"] is not None:
            return observations[0]["value"]
        return None

    def generate_signal(self) -> dict:
        """Generate macro signal based on upcoming events and economic data.

        Returns: {signal: str, strength: 0-100, reason: str,
                  upcoming_events: list, reduce_size: bool}
        """
        events = self.get_upcoming_events(days_ahead=2)
        cpi_trend = self.get_cpi_trend()

        signal = "neutral"
        strength = 0
        reason = "No significant macro events"
        reduce_size = False

        # If major event within 48h, recommend reducing position sizes
        if events:
            reduce_size = True
            event_types = [e["event_type"] for e in events]
            reason = f"Upcoming: {', '.join(e['description'] for e in events)}"

            if "FOMC" in event_types:
                strength = 60
                # Lean based on CPI trend
                if cpi_trend == "falling":
                    signal = "buy"  # dovish expectation
                    reason += " | CPI falling → dovish lean"
                elif cpi_trend == "rising":
                    signal = "sell"  # hawkish expectation
                    reason += " | CPI rising → hawkish lean"
                else:
                    signal = "neutral"
                    strength = 30
            else:
                strength = 30

        return {
            "signal": signal,
            "strength": strength,
            "reason": reason,
            "upcoming_events": events,
            "reduce_size": reduce_size,
            "cpi_trend": cpi_trend,
        }

    def generate_backtest_signal(self, date_str: str) -> dict:
        """Generate macro signal for a specific date (backtesting)."""
        events = self.get_events_at_date(date_str, days_ahead=2)

        signal = "neutral"
        strength = 0
        reason = "No significant macro events"
        reduce_size = False

        if events:
            reduce_size = True
            event_types = [e["event_type"] for e in events]
            reason = f"Upcoming: {', '.join(e['description'] for e in events)}"

            if "FOMC" in event_types:
                strength = 60
                signal = "neutral"  # can't know direction in backtest without data
            else:
                strength = 30

        return {
            "signal": signal,
            "strength": strength,
            "reason": reason,
            "upcoming_events": events,
            "reduce_size": reduce_size,
        }
