"""
Kraken API client — used ONLY for fetching price data.
No order placement ever happens through this client.

Auto-learning applied:
- SSL errors (UNEXPECTED_EOF): retry with exponential backoff + fresh session
- Rate limits: respect Kraken's limits with adaptive throttling
- Connection resets: rebuild krakenex session on repeated failures
- All failure patterns logged for post-mortem analysis
"""
import logging
import time
import requests
from typing import Optional, Dict, Any

import krakenex
from pykrakenapi import KrakenAPI

import config

logger = logging.getLogger("cryptobot.kraken")

# Auto-learning: track failure patterns for diagnosis
_failure_counts: Dict[str, int] = {}
_MAX_CONSECUTIVE_FAILURES = 10
_SESSION_REBUILD_THRESHOLD = 5  # rebuild session after this many SSL errors


class KrakenClient:
    """Thin wrapper around krakenex for price data with resilient retry logic."""

    def __init__(self):
        self._build_session()
        self._last_call = 0.0
        self._min_interval = 2.0  # Kraken rate-limit courtesy
        self._consecutive_failures = 0
        self._ssl_error_count = 0

    def _build_session(self):
        """Build or rebuild the Kraken API session.
        
        Auto-learning: SSL EOF errors accumulate when the underlying
        requests.Session gets stale. Rebuilding the session (fresh TCP
        connection + TLS handshake) resolves this.
        """
        self._api = krakenex.API()
        self._api.key = config.KRAKEN_API_KEY
        self._api.secret = config.KRAKEN_PRIVATE_KEY
        # Force a fresh requests session
        self._api.session = requests.Session()
        self._k = KrakenAPI(self._api)
        self._ssl_error_count = 0
        logger.info("Kraken API session (re)built")

    def _throttle(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _retry_with_backoff(self, func, max_retries: int = 3, base_delay: float = 5.0):
        """
        Retry a Kraken API call with exponential backoff.
        
        Auto-learning patterns encoded:
        - SSLError / ConnectionError: rebuild session after threshold
        - Timeout: increase backoff, log for rate-limit calibration
        - Other errors: standard retry with logging
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                self._throttle()
                result = func()
                # Success: reset failure counters
                self._consecutive_failures = 0
                return result
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                self._ssl_error_count += 1
                self._consecutive_failures += 1
                error_type = type(e).__name__
                _failure_counts[error_type] = _failure_counts.get(error_type, 0) + 1
                logger.warning(
                    "Kraken %s (attempt %d/%d, total SSL errors: %d): %s",
                    error_type, attempt + 1, max_retries, self._ssl_error_count, e
                )
                # Auto-learning: rebuild session when SSL errors accumulate
                if self._ssl_error_count >= _SESSION_REBUILD_THRESHOLD:
                    logger.info("SSL error threshold hit (%d) - rebuilding session",
                                self._ssl_error_count)
                    self._build_session()
                last_error = e
            except Exception as e:
                self._consecutive_failures += 1
                error_type = type(e).__name__
                _failure_counts[error_type] = _failure_counts.get(error_type, 0) + 1
                logger.warning(
                    "Kraken error (attempt %d/%d): %s: %s",
                    attempt + 1, max_retries, error_type, e
                )
                last_error = e

            # Exponential backoff: 5s, 10s, 20s
            delay = base_delay * (2 ** attempt)
            logger.debug("Backing off %.1fs before retry", delay)
            time.sleep(delay)

        # All retries exhausted
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            logger.error(
                "CRITICAL: %d consecutive Kraken failures. Failure pattern: %s. "
                "Rebuilding session as last resort.",
                self._consecutive_failures, _failure_counts
            )
            self._build_session()
            self._consecutive_failures = 0

        raise last_error if last_error else RuntimeError("Kraken API call failed")

    def get_ticker(self, pair: str = config.PAIR) -> Optional[Dict[str, Any]]:
        """Return ticker dict with 'ask', 'bid', 'last' as floats."""
        try:
            def _fetch():
                resp = self._api.query_public("Ticker", {"pair": pair})
                if resp.get("error"):
                    logger.error("Kraken Ticker error: %s", resp["error"])
                    return None
                data = list(resp["result"].values())[0]
                return {
                    "ask": float(data["a"][0]),
                    "bid": float(data["b"][0]),
                    "last": float(data["c"][0]),
                    "volume_24h": float(data["v"][1]),
                    "high_24h": float(data["h"][1]),
                    "low_24h": float(data["l"][1]),
                }
            return self._retry_with_backoff(_fetch)
        except Exception as e:
            logger.error("Failed to fetch ticker after retries: %s", e)
            return None

    def get_ohlc(self, pair: str = config.PAIR, interval: int = 60, count: int = 100):
        """Return OHLC as list of dicts with open/high/low/close/volume keys.
        
        Uses raw Kraken API to avoid pykrakenapi pandas freq bug.
        """
        try:
            def _fetch():
                resp = self._api.query_public("OHLC", {"pair": pair, "interval": interval})
                if resp.get("error"):
                    logger.error("Kraken OHLC error: %s", resp["error"])
                    return None
                result = resp.get("result", {})
                pair_data = None
                for k, v in result.items():
                    if k != "last" and isinstance(v, list):
                        pair_data = v
                        break
                if not pair_data:
                    return None
                records = []
                for candle in pair_data:
                    records.append({
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[6]),
                    })
                return records[-count:] if len(records) > count else records
            return self._retry_with_backoff(_fetch)
        except Exception as e:
            logger.error("Failed to fetch OHLC after retries: %s", e)
            return None

    def get_failure_summary(self) -> Dict[str, Any]:
        """Return failure pattern summary for auto-learning diagnosis."""
        return {
            "consecutive_failures": self._consecutive_failures,
            "ssl_errors_since_rebuild": self._ssl_error_count,
            "failure_counts_by_type": dict(_failure_counts),
        }
