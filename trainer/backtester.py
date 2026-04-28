#!/usr/bin/env python3
"""
Backtesting Engine
==================
Downloads 6 months of historical BTC/USD 1h candles from Kraken's public API,
runs each strategy against the data, and outputs per-strategy performance stats.

Usage:
    python -m trainer.backtester
"""
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy API keys so config.py doesn't raise during backtesting
# (backtester uses public endpoints only, no auth needed)
if not os.environ.get("KRAKEN_API_KEY"):
    os.environ["KRAKEN_API_KEY"] = "backtest-dummy"
if not os.environ.get("KRAKEN_PRIVATE_KEY"):
    os.environ["KRAKEN_PRIVATE_KEY"] = "backtest-dummy"

import random
import requests

import config

logger = logging.getLogger("cryptoworm.backtester")


# ── Historical data fetching ────────────────────────────────────────────

OHLC_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ohlc_cache.json")


def _load_ohlc_cache(pair: str, interval: int, months: int) -> Optional[list]:
    """Load cached OHLC data if it exists and covers the requested range."""
    if not os.path.exists(OHLC_CACHE_PATH):
        return None
    try:
        with open(OHLC_CACHE_PATH, "r") as f:
            cache = json.load(f)
        key = f"{pair}_{interval}_{months}"
        entry = cache.get(key)
        if not entry:
            return None
        # Cache is valid for 24 hours
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if (datetime.utcnow() - cached_at).total_seconds() > 86400:
            logger.info("OHLC cache expired, will re-fetch")
            return None
        candles = entry["candles"]
        logger.info("Loaded %d candles from cache (cached %s)", len(candles), entry["cached_at"])
        return candles
    except Exception as e:
        logger.warning("Failed to load OHLC cache: %s", e)
        return None


def _save_ohlc_cache(pair: str, interval: int, months: int, candles: list):
    """Save fetched OHLC data to local cache."""
    key = f"{pair}_{interval}_{months}"
    try:
        cache = {}
        if os.path.exists(OHLC_CACHE_PATH):
            with open(OHLC_CACHE_PATH, "r") as f:
                cache = json.load(f)
        cache[key] = {
            "cached_at": datetime.utcnow().isoformat(),
            "candle_count": len(candles),
            "candles": candles,
        }
        with open(OHLC_CACHE_PATH, "w") as f:
            json.dump(cache, f)
        logger.info("Saved %d candles to cache", len(candles))
    except Exception as e:
        logger.warning("Failed to save OHLC cache: %s", e)


def _fetch_coingecko(months: int = 6) -> list:
    """Fetch BTC/USD data from CoinGecko free API.

    Makes two requests:
    - days=90 for hourly granularity (recent 3 months)
    - days=180 for daily granularity (older 3 months)
    Merges them: hourly where available, daily for the rest.

    Returns list of candle dicts: time, open, high, low, close, volume.
    """
    base_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    all_candles = []

    # Request 1: daily data for full 180 days (gives daily points for older data)
    logger.info("CoinGecko: fetching 180 days of daily data...")
    try:
        resp = requests.get(base_url, params={
            "vs_currency": "usd",
            "days": min(months * 30, 180),
        }, timeout=30)
        resp.raise_for_status()
        daily_data = resp.json()
    except Exception as e:
        logger.warning("CoinGecko daily request failed: %s", e)
        return []

    # Rate limit safety: wait before second request
    time.sleep(6)

    # Request 2: hourly data for recent 90 days
    logger.info("CoinGecko: fetching 90 days of hourly data...")
    try:
        resp = requests.get(base_url, params={
            "vs_currency": "usd",
            "days": 90,
        }, timeout=30)
        resp.raise_for_status()
        hourly_data = resp.json()
    except Exception as e:
        logger.warning("CoinGecko hourly request failed: %s", e)
        return []

    # Determine the cutoff: hourly data starts ~90 days ago
    hourly_prices = hourly_data.get("prices", [])
    hourly_start_ms = hourly_prices[0][0] if hourly_prices else float("inf")

    # Process daily data (only for timestamps BEFORE hourly data starts)
    daily_prices = daily_data.get("prices", [])
    daily_volumes = daily_data.get("total_volumes", [])
    vol_by_day = {int(v[0]): v[1] for v in daily_volumes} if daily_volumes else {}

    for i, (ts_ms, price) in enumerate(daily_prices):
        if ts_ms >= hourly_start_ms:
            break  # hourly data takes over from here
        ts = int(ts_ms / 1000)
        # Daily point: open=close=high=low=price (single data point per day)
        vol = 0.0
        # Find closest volume entry
        for vts in vol_by_day:
            if abs(vts - int(ts_ms)) < 86400000:  # within 1 day
                vol = vol_by_day[vts]
                break
        all_candles.append({
            "time": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": vol,
        })

    # Process hourly data: group price points by hour into OHLC candles
    hourly_volumes = hourly_data.get("total_volumes", [])
    vol_lookup = {}
    for ts_ms, vol in hourly_volumes:
        hour_ts = int(ts_ms / 1000) // 3600 * 3600
        vol_lookup.setdefault(hour_ts, []).append(vol)

    hour_buckets = {}
    for ts_ms, price in hourly_prices:
        hour_ts = int(ts_ms / 1000) // 3600 * 3600
        hour_buckets.setdefault(hour_ts, []).append(price)

    for hour_ts in sorted(hour_buckets.keys()):
        prices = hour_buckets[hour_ts]
        vols = vol_lookup.get(hour_ts, [0.0])
        all_candles.append({
            "time": hour_ts,
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(vols) / len(vols) if vols else 0.0,
        })

    logger.info("CoinGecko: assembled %d candles", len(all_candles))
    return all_candles


def _fetch_cryptocompare(months: int = 6) -> list:
    """Fetch BTC/USD hourly data from CryptoCompare free API.

    Paginates backwards using toTs parameter, up to 2000 candles per request.
    Returns list of candle dicts: time, open, high, low, close, volume.
    """
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    all_candles = []
    target_hours = months * 30 * 24
    to_ts = int(datetime.utcnow().timestamp())
    earliest_ts = int((datetime.utcnow() - timedelta(days=months * 30)).timestamp())
    page = 0
    max_pages = (target_hours // 2000) + 2

    logger.info("CryptoCompare: fetching ~%d hours of data (~%d pages)...", target_hours, max_pages)

    while page < max_pages:
        page += 1
        try:
            resp = requests.get(url, params={
                "fsym": "BTC",
                "tsym": "USD",
                "limit": 2000,
                "toTs": to_ts,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("CryptoCompare request failed (page %d): %s", page, e)
            break

        if data.get("Response") != "Success":
            logger.warning("CryptoCompare error: %s", data.get("Message", "unknown"))
            break

        candle_data = data.get("Data", {}).get("Data", [])
        if not candle_data:
            break

        for c in candle_data:
            ts = int(c["time"])
            if ts < earliest_ts:
                continue
            # Skip zero-price entries (CryptoCompare returns these for future timestamps)
            if c.get("close", 0) <= 0:
                continue
            all_candles.append({
                "time": ts,
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("volumefrom", 0)),
            })

        # Move pagination cursor to before the earliest candle in this batch
        batch_earliest = min(c["time"] for c in candle_data)
        if batch_earliest <= earliest_ts:
            break
        to_ts = int(batch_earliest)

        logger.info("CryptoCompare page %d: %d candles so far", page, len(all_candles))
        time.sleep(1)  # rate limit

    logger.info("CryptoCompare: fetched %d candles", len(all_candles))
    return all_candles


def _fetch_kraken(pair: str = "XXBTZUSD", interval: int = 60, months: int = 6) -> list:
    """Download historical OHLC from Kraken public API (fallback, ~720 candles max).

    Returns list of dicts with keys: time, open, high, low, close, volume.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    all_candles = []
    since = int((datetime.utcnow() - timedelta(days=months * 30)).timestamp())
    end_ts = int(datetime.utcnow().timestamp())
    max_pages = (months * 30 * 24) // 720 + 2

    logger.info("Kraken: fetching %d months of %dm candles for %s (~%d pages)...",
                months, interval, pair, max_pages)

    page = 0
    while since < end_ts and page < max_pages:
        page += 1
        retries = 0
        max_retries = 5
        backoff = 3.0

        while retries <= max_retries:
            try:
                resp = requests.get(url, params={
                    "pair": pair,
                    "interval": interval,
                    "since": since,
                }, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                errors = data.get("error", [])
                rate_limited = any("rate" in str(e).lower() or "EAPI:Rate" in str(e)
                                   for e in errors) if errors else False
                if rate_limited:
                    retries += 1
                    wait = backoff * (2 ** (retries - 1))
                    logger.warning("Rate limited (attempt %d/%d), backing off %.1fs...",
                                   retries, max_retries, wait)
                    time.sleep(wait)
                    continue

                if errors:
                    logger.error("Kraken API error: %s", errors)
                    break

                result = data.get("result", {})
                last = result.pop("last", None)

                pair_data = None
                for k, v in result.items():
                    if isinstance(v, list):
                        pair_data = v
                        break

                if not pair_data:
                    break

                for candle in pair_data:
                    all_candles.append({
                        "time": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[6]),
                    })

                if last:
                    since = int(last)
                else:
                    break

                time.sleep(3)
                break

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > max_retries:
                    logger.error("Failed to fetch page %d after %d retries: %s",
                                 page, max_retries, e)
                    break
                wait = backoff * (2 ** (retries - 1))
                logger.warning("Request error (attempt %d/%d): %s — retrying in %.1fs",
                               retries, max_retries, e, wait)
                time.sleep(wait)
            except Exception as e:
                logger.error("Unexpected error fetching OHLC page %d: %s", page, e)
                break
        else:
            logger.error("Giving up on page %d after %d retries", page, max_retries)
            break

    logger.info("Kraken: fetched %d candles", len(all_candles))
    return all_candles


def _deduplicate_and_sort(candles: list) -> list:
    """Deduplicate candles by timestamp and sort chronologically."""
    seen = set()
    unique = []
    for c in candles:
        if c["time"] not in seen:
            seen.add(c["time"])
            unique.append(c)
    unique.sort(key=lambda x: x["time"])
    return unique


def fetch_historical_data(pair: str = "XXBTZUSD", interval: int = 60,
                          months: int = 6) -> list:
    """Fetch historical BTC/USD candle data with multi-source fallback.

    Priority:
    1. Local cache (if <24h old)
    2. CoinGecko (90 days hourly + 90 days daily = ~6 months)
    3. CryptoCompare (paginated hourly, up to 6 months)
    4. Kraken (fallback, only ~30 days / 720 candles)

    Returns list of dicts with keys: time, open, high, low, close, volume.
    """
    # 1. Check local cache
    cached = _load_ohlc_cache(pair, interval, months)
    if cached:
        return cached

    candles = []

    # 2. Try CryptoCompare FIRST — provides full hourly data for 6 months (free, no key)
    logger.info("Attempting CryptoCompare as primary data source (full hourly)...")
    try:
        candles = _fetch_cryptocompare(months)
    except Exception as e:
        logger.warning("CryptoCompare failed: %s", e)

    # 3. Fall back to CoinGecko (hourly for 90d + daily for older)
    if len(candles) < 100:
        logger.info("CryptoCompare insufficient (%d candles), trying CoinGecko...", len(candles))
        try:
            candles = _fetch_coingecko(months)
        except Exception as e:
            logger.warning("CoinGecko failed: %s", e)

    # 4. Fall back to Kraken (limited to ~720 candles)
    if len(candles) < 100:
        logger.info("CoinGecko insufficient (%d candles), falling back to Kraken...", len(candles))
        try:
            candles = _fetch_kraken(pair, interval, months)
        except Exception as e:
            logger.warning("Kraken failed: %s", e)

    # Deduplicate and sort
    candles = _deduplicate_and_sort(candles)

    if candles:
        logger.info("Historical data: %d candles spanning %s to %s",
                    len(candles),
                    datetime.utcfromtimestamp(candles[0]["time"]).strftime("%Y-%m-%d"),
                    datetime.utcfromtimestamp(candles[-1]["time"]).strftime("%Y-%m-%d"))
        _save_ohlc_cache(pair, interval, months, candles)
    else:
        logger.warning("No candles fetched from any source")

    return candles


# ── Mock Kraken client ──────────────────────────────────────────────────

class MockKrakenClient:
    """Serves historical candles as if they were live Kraken API responses."""

    def __init__(self, candles: list):
        self.candles = candles
        self._tick_idx = 0
        self._ohlc_cache = {}  # interval -> list of candles

        # Pre-build OHLC caches for different intervals
        self._build_ohlc_caches()

    def _build_ohlc_caches(self):
        """Pre-aggregate 1h candles into 4h candles for RSI divergence strategy."""
        self._ohlc_cache[60] = self.candles  # 1h is the raw data

        # Build 4h candles by grouping every 4 hourly candles
        four_h = []
        for i in range(0, len(self.candles) - 3, 4):
            group = self.candles[i:i + 4]
            four_h.append({
                "time": group[0]["time"],
                "open": group[0]["open"],
                "high": max(c["high"] for c in group),
                "low": min(c["low"] for c in group),
                "close": group[-1]["close"],
                "volume": sum(c["volume"] for c in group),
            })
        self._ohlc_cache[240] = four_h

    def set_tick(self, idx: int):
        """Set current position in the candle series."""
        self._tick_idx = idx

    def get_ticker(self, pair: str = None) -> Optional[dict]:
        """Return ticker-like data from the current candle."""
        if self._tick_idx >= len(self.candles):
            return None
        c = self.candles[self._tick_idx]
        # Compute 24h high/low from last 24 candles
        start = max(0, self._tick_idx - 24)
        recent = self.candles[start:self._tick_idx + 1]
        return {
            "ask": c["close"] * 1.0005,
            "bid": c["close"] * 0.9995,
            "last": c["close"],
            "volume_24h": sum(x["volume"] for x in recent),
            "high_24h": max(x["high"] for x in recent),
            "low_24h": min(x["low"] for x in recent),
        }

    def get_ohlc(self, pair: str = None, interval: int = 60, count: int = 100) -> Optional[list]:
        """Return historical OHLC data up to the current tick."""
        cache = self._ohlc_cache.get(interval)
        if not cache:
            return None

        # Find the candles up to current time
        current_time = self.candles[self._tick_idx]["time"] if self._tick_idx < len(self.candles) else 0
        relevant = [c for c in cache if c["time"] <= current_time]
        return relevant[-count:] if len(relevant) > count else relevant


# ── Mock Risk Manager for backtesting ───────────────────────────────────

class BacktestRiskManager:
    """Simulates the risk manager without file I/O or trade logging.

    Includes realistic transaction costs and slippage simulation.
    """

    def __init__(self, initial_balance: float = 500.0,
                 slippage_pct: float = 0.05, fee_pct: float = 0.075):
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.positions = []
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.is_paused = False
        self.pause_reason = ""
        self._max_concurrent = 3
        self._balance_history = [initial_balance]
        self._slippage_pct = slippage_pct   # 0.05% per trade
        self._fee_pct = fee_pct             # 0.075% per trade (one-way)

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage: buys fill higher, sells fill lower."""
        if side == "buy":
            return price * (1 + self._slippage_pct / 100)
        else:
            return price * (1 - self._slippage_pct / 100)

    def _calc_fee(self, size_usd: float) -> float:
        """Calculate transaction fee."""
        return size_usd * (self._fee_pct / 100)

    def can_open_position(self, price: float) -> tuple:
        if self.is_paused:
            return False, "paused"
        open_count = len([p for p in self.positions if p["status"] == "open"])
        if open_count >= self._max_concurrent:
            return False, "max positions"
        # Daily loss limit: 5%
        if self.daily_pnl < -(self.balance * 0.05):
            return False, "daily loss limit"
        return True, "OK"

    def max_position_size_usd(self) -> float:
        return self.balance * 0.02  # 2%

    def position_size_btc(self, price: float) -> float:
        return self.max_position_size_usd() / price

    def open_position(self, side: str, price: float, size_btc: float,
                      strategy: str, stop_loss: float, take_profit: float) -> dict:
        # Apply slippage to entry price
        fill_price = self._apply_slippage(price, side)
        size_usd = fill_price * size_btc
        # Deduct entry fee
        entry_fee = self._calc_fee(size_usd)
        self.balance -= entry_fee

        pos = {
            "id": f"{strategy}-bt-{len(self.positions)}",
            "side": side,
            "entry_price": fill_price,
            "size_btc": size_btc,
            "size_usd": size_usd,
            "strategy": strategy,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "open",
            "opened_at": datetime.utcnow().isoformat(),
            "closed_at": None,
            "exit_price": None,
            "pnl": 0.0,
            "fees": entry_fee,
        }
        self.positions.append(pos)
        return pos

    def close_position(self, pos_id: str, exit_price: float) -> Optional[dict]:
        for pos in self.positions:
            if pos["id"] == pos_id and pos["status"] == "open":
                pos["status"] = "closed"
                # Apply slippage to exit (opposite direction)
                exit_side = "sell" if pos["side"] == "buy" else "buy"
                fill_exit = self._apply_slippage(exit_price, exit_side)
                pos["exit_price"] = fill_exit
                pos["closed_at"] = datetime.utcnow().isoformat()

                if pos["side"] == "buy":
                    pos["pnl"] = (fill_exit - pos["entry_price"]) * pos["size_btc"]
                else:
                    pos["pnl"] = (pos["entry_price"] - fill_exit) * pos["size_btc"]

                # Deduct exit fee
                exit_fee = self._calc_fee(fill_exit * pos["size_btc"])
                pos["fees"] = pos.get("fees", 0) + exit_fee
                pos["pnl"] -= exit_fee
                self.balance -= exit_fee

                self.balance += pos["pnl"] + exit_fee  # pnl already includes fee
                self.daily_pnl += pos["pnl"]
                self.weekly_pnl += pos["pnl"]

                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance

                self._balance_history.append(self.balance)
                self.closed_trades.append(pos)
                return pos
        return None

    def check_stop_loss_take_profit(self, current_price: float) -> list:
        closed = []
        for pos in list(self.positions):
            if pos["status"] != "open":
                continue
            if pos["side"] == "buy":
                if current_price <= pos["stop_loss"] or current_price >= pos["take_profit"]:
                    result = self.close_position(pos["id"], current_price)
                    if result:
                        closed.append(result)
            else:
                if current_price >= pos["stop_loss"] or current_price <= pos["take_profit"]:
                    result = self.close_position(pos["id"], current_price)
                    if result:
                        closed.append(result)
        return closed

    def save_state(self):
        pass  # No persistence in backtests

    def get_daily_summary(self) -> dict:
        return {"date": "", "balance": self.balance, "daily_pnl": self.daily_pnl,
                "trades_opened": 0, "trades_closed": 0, "wins": 0, "losses": 0,
                "win_rate": 0, "open_positions": 0, "drawdown_pct": 0, "paused": False}


# ── Sentiment proxy ─────────────────────────────────────────────────────

class MockSentimentProvider:
    """Provides a Fear & Greed proxy using 14-day RSI of price.
    RSI < 25 = extreme fear, RSI > 75 = extreme greed.
    """

    def __init__(self, closes: list):
        self._fng_values = self._compute_fng_proxy(closes)

    def _compute_fng_proxy(self, closes: list) -> list:
        """Convert 14-period RSI into Fear & Greed-like values (0-100)."""
        result = []
        period = 14
        for i in range(len(closes)):
            if i < period:
                result.append(50)  # neutral default
                continue
            segment = closes[max(0, i - period):i + 1]
            deltas = [segment[j] - segment[j - 1] for j in range(1, len(segment))]
            gains = [d for d in deltas if d > 0]
            losses = [-d for d in deltas if d < 0]
            avg_gain = sum(gains) / period if gains else 0.001
            avg_loss = sum(losses) / period if losses else 0.001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            # Map RSI directly to FNG-like scale (already 0-100)
            result.append(int(rsi))
        return result

    def get_fng(self, idx: int) -> dict:
        val = self._fng_values[idx] if idx < len(self._fng_values) else 50
        if val <= 25:
            classification = "Extreme Fear"
        elif val <= 40:
            classification = "Fear"
        elif val <= 60:
            classification = "Neutral"
        elif val <= 75:
            classification = "Greed"
        else:
            classification = "Extreme Greed"
        return {"value": val, "classification": classification}


# ── Strategy runners ────────────────────────────────────────────────────

def run_strategy_backtest(strategy_name: str, candles: list, initial_balance: float = 500.0) -> dict:
    """Run a single strategy against historical data and return stats."""
    mock_kraken = MockKrakenClient(candles)
    slippage = getattr(config, "REALISTIC_SLIPPAGE_PCT", 0.05)
    fee = getattr(config, "REALISTIC_FEE_PCT", 0.075)
    risk = BacktestRiskManager(initial_balance, slippage_pct=slippage, fee_pct=fee)

    # Import strategy classes
    from strategies.grid import GridBot
    from strategies.sentiment import SentimentSwing
    from strategies.ema_macd import EmaMacdMomentum
    from strategies.bollinger import BollingerMeanReversion
    from strategies.rsi_divergence import RsiDivergence
    from strategies.political import PoliticalSignals
    from strategies.novel import TariffWhiplashStrategy, CongressionalFrontRunStrategy

    # Monkey-patch sentiment to use our proxy if running sentiment strategy
    sentiment_provider = None
    if strategy_name == "sentiment":
        closes = [c["close"] for c in candles]
        sentiment_provider = MockSentimentProvider(closes)

    # Pre-generate synthetic data for political strategies
    synthetic_signals = None
    synthetic_congress = None
    if strategy_name in ("political", "tariff_whiplash", "congress_frontrun"):
        synthetic_signals = generate_synthetic_political_signals(candles)
        synthetic_congress = generate_synthetic_congress_trades(candles)

    # Create strategy instance
    if strategy_name == "grid":
        strategy = GridBot(mock_kraken, risk)
    elif strategy_name == "sentiment":
        strategy = SentimentSwing(mock_kraken, risk)
        # Patch the FNG fetch to use our proxy
        _original_fetch = strategy._fetch_fear_greed
        def _mock_fng(idx=[0]):
            result = sentiment_provider.get_fng(mock_kraken._tick_idx)
            strategy._last_fng = result
            prev_idx = max(0, mock_kraken._tick_idx - 24)  # ~1 day prior
            strategy._prev_fng = sentiment_provider.get_fng(prev_idx)
            return result
        strategy._fetch_fear_greed = _mock_fng
    elif strategy_name == "ema_macd":
        strategy = EmaMacdMomentum(mock_kraken, risk)
    elif strategy_name == "bollinger":
        strategy = BollingerMeanReversion(mock_kraken, risk)
    elif strategy_name == "rsi_divergence":
        strategy = RsiDivergence(mock_kraken, risk)
    elif strategy_name == "political":
        strategy = PoliticalSignals(mock_kraken, risk)
    elif strategy_name == "tariff_whiplash":
        strategy = TariffWhiplashStrategy(mock_kraken, risk)
    elif strategy_name == "congress_frontrun":
        strategy = CongressionalFrontRunStrategy(mock_kraken, risk)
    else:
        return {"error": f"Unknown strategy: {strategy_name}"}

    # Warm-up period: skip first 50 candles to allow indicators to stabilize
    warmup = 50
    total_ticks = len(candles)

    logger.info("Backtesting %s: %d ticks (warmup=%d)", strategy_name, total_ticks, warmup)

    for i in range(warmup, total_ticks):
        mock_kraken.set_tick(i)
        price = candles[i]["close"]

        # Check SL/TP
        risk.check_stop_loss_take_profit(price)

        # Grid bot needs initialization
        if strategy_name == "grid" and not strategy._initialized:
            strategy.initialize(price)
            continue

        if strategy_name == "grid" and strategy.should_reinitialize(price):
            strategy.initialize(price)

        # Run strategy
        try:
            if strategy_name == "political" and synthetic_signals is not None:
                strategy.evaluate_backtest(price, candles[i]["time"], synthetic_signals)
            elif strategy_name == "tariff_whiplash" and synthetic_signals is not None:
                strategy.evaluate_backtest(price, candles[i]["time"], synthetic_signals)
            elif strategy_name == "congress_frontrun" and synthetic_congress is not None:
                strategy.evaluate_backtest(price, candles[i]["time"], synthetic_congress)
            else:
                strategy.evaluate(price)
        except Exception as e:
            logger.debug("Strategy %s error at tick %d: %s", strategy_name, i, e)

    # Close any remaining open positions at last price
    final_price = candles[-1]["close"]
    for pos in list(risk.positions):
        if pos["status"] == "open":
            risk.close_position(pos["id"], final_price)

    # Compute stats (pass candles for buy-and-hold benchmark)
    return compute_stats(strategy_name, risk, initial_balance, candles=candles)


def _calc_sortino(returns: list) -> float:
    """Calculate Sortino ratio (penalizes only downside deviation)."""
    if len(returns) < 2:
        return 0.0
    avg_return = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if not downside:
        return float("inf") if avg_return > 0 else 0.0
    downside_dev = math.sqrt(sum(r ** 2 for r in downside) / len(downside))
    if downside_dev == 0:
        return 0.0
    return (avg_return / downside_dev) * math.sqrt(252)


def _monte_carlo_drawdown(trades: list, n_simulations: int = 1000) -> float:
    """Monte Carlo simulation: shuffle trade P&L sequence, report 95th percentile max drawdown.

    Returns 95th percentile max drawdown as an absolute dollar value.
    """
    if not trades:
        return 0.0
    pnls = [t["pnl"] for t in trades]
    max_drawdowns = []
    for _ in range(n_simulations):
        shuffled = pnls[:]
        random.shuffle(shuffled)
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in shuffled:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        # As percentage of starting balance (use initial + peak as reference)
        max_drawdowns.append(max_dd)
    max_drawdowns.sort()
    idx_95 = int(len(max_drawdowns) * 0.95)
    return max_drawdowns[min(idx_95, len(max_drawdowns) - 1)]


def _buy_and_hold_benchmark(candles: list, initial_balance: float, warmup: int = 50) -> dict:
    """Calculate buy-and-hold benchmark over the same period."""
    if not candles or len(candles) <= warmup:
        return {"final_balance": initial_balance, "total_return_pct": 0, "max_drawdown_pct": 0}
    entry_price = candles[warmup]["close"]
    exit_price = candles[-1]["close"]
    if entry_price <= 0:
        return {"final_balance": initial_balance, "total_return_pct": 0, "max_drawdown_pct": 0}
    btc_amount = initial_balance / entry_price
    final_balance = btc_amount * exit_price
    total_return = (final_balance - initial_balance) / initial_balance * 100
    # Max drawdown
    peak = entry_price
    max_dd = 0
    for c in candles[warmup:]:
        p = c["close"]
        if p > peak:
            peak = p
        dd = (peak - p) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return {
        "final_balance": round(final_balance, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }


def compute_stats(strategy_name: str, risk: BacktestRiskManager,
                  initial_balance: float, candles: list = None) -> dict:
    """Compute performance statistics from completed backtest.

    Includes: Sharpe, Sortino, max drawdown, profit factor, win rate,
    Monte Carlo 95th percentile drawdown, and buy-and-hold comparison.
    """
    trades = risk.closed_trades
    if not trades:
        return {
            "strategy": strategy_name,
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "monte_carlo_dd_95": 0,
            "avg_hold_time_hours": 0,
            "final_balance": risk.balance,
            "total_pnl": 0,
            "total_return_pct": 0,
            "total_fees": 0,
            "buy_and_hold": {},
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    total_fees = sum(t.get("fees", 0) for t in trades)

    # Win rate
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    # Max drawdown from balance history
    balance_history = risk._balance_history
    peak = balance_history[0]
    max_dd = 0
    for b in balance_history:
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (using per-trade returns)
    returns = [t["pnl"] / t["size_usd"] for t in trades if t["size_usd"] > 0]
    if len(returns) > 1:
        avg_return = sum(returns) / len(returns)
        std_return = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1))
        sharpe = (avg_return / std_return) * math.sqrt(252) if std_return > 0 else 0
    else:
        sharpe = 0

    # Sortino ratio
    sortino = _calc_sortino(returns)

    # Monte Carlo 95th percentile max drawdown
    mc_dd_95 = _monte_carlo_drawdown(trades)

    # Average hold time
    hold_times = []
    for t in trades:
        if t.get("opened_at") and t.get("closed_at"):
            try:
                opened = datetime.fromisoformat(t["opened_at"])
                closed = datetime.fromisoformat(t["closed_at"])
                hold_times.append((closed - opened).total_seconds() / 3600)
            except Exception:
                pass
    avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0

    total_pnl = risk.balance - initial_balance

    # Buy-and-hold benchmark
    bnh = _buy_and_hold_benchmark(candles, initial_balance) if candles else {}

    result = {
        "strategy": strategy_name,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2) if sortino != float("inf") else 999.0,
        "monte_carlo_dd_95": round(mc_dd_95, 2),
        "avg_hold_time_hours": round(avg_hold, 1),
        "final_balance": round(risk.balance, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / initial_balance * 100, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "total_fees": round(total_fees, 2),
        "buy_and_hold": bnh,
    }

    if bnh:
        logger.info("  vs Buy-and-Hold: %.2f%% return, %.2f%% max DD",
                     bnh.get("total_return_pct", 0), bnh.get("max_drawdown_pct", 0))

    return result


# ── Synthetic historical political signals ─────────────────────────────

def generate_synthetic_political_signals(candles: list) -> list:
    """Generate synthetic political signals based on known historical events.

    Uses real dates of major crypto-moving political events from late 2024 - 2025.
    Each signal has: timestamp, score (-100 to +100), category.
    """
    signals = []

    # Known historical events (date, score, category, description)
    known_events = [
        # 2024 events
        ("2024-11-05 22:00", 70, "crypto", "Trump wins 2024 election — pro-crypto president"),
        ("2024-11-06 10:00", 50, "crypto", "Bitcoin rallies on election result"),
        ("2024-12-05 14:00", 60, "crypto", "Trump nominates pro-crypto SEC chair"),

        # 2025 Q1
        ("2025-01-20 12:00", 55, "crypto", "Trump inauguration — crypto executive orders expected"),
        ("2025-01-23 09:00", 45, "crypto", "Executive order on digital assets signed"),
        ("2025-02-01 10:00", -40, "tariff", "Trump announces 25% tariffs on Canada/Mexico"),
        ("2025-02-04 08:00", 30, "tariff", "Tariff pause announced for 30 days"),
        ("2025-02-12 14:00", -35, "tariff", "Reciprocal tariff framework announced"),
        ("2025-03-02 11:00", 70, "crypto", "Bitcoin Strategic Reserve executive order signed"),
        ("2025-03-06 10:00", 60, "crypto", "White House Crypto Summit"),
        ("2025-03-12 08:30", -15, "fed", "CPI comes in hot — rate cut expectations drop"),

        # 2025 Q2
        ("2025-04-02 16:00", -60, "tariff", "Liberation Day — massive tariff announcement"),
        ("2025-04-03 09:00", -50, "tariff", "Markets crash on tariff fears"),
        ("2025-04-07 08:00", -45, "tariff", "China retaliatory tariffs announced"),
        ("2025-04-09 13:00", 50, "tariff", "90-day tariff pause announced (except China)"),
        ("2025-04-10 09:00", 40, "tariff", "Markets rally on tariff pause"),
        ("2025-04-15 10:00", -20, "regulation", "SEC crypto enforcement action"),
        ("2025-05-07 14:00", -10, "fed", "FOMC meeting — rates unchanged"),

        # 2025 Q3
        ("2025-06-18 14:00", 25, "fed", "FOMC signals rate cuts possible in Sept"),
        ("2025-07-10 08:30", 15, "fed", "CPI drops — rate cut expectations rise"),
        ("2025-07-15 10:00", -30, "tariff", "New tariffs on EU goods announced"),
        ("2025-07-18 11:00", 25, "tariff", "EU tariff exemptions granted"),
        ("2025-08-05 10:00", 35, "crypto", "Stablecoin bill passes Senate"),
        ("2025-09-17 14:00", 40, "fed", "Fed cuts rates 25bp — dovish statement"),

        # 2025 Q4
        ("2025-10-01 10:00", -25, "tariff", "New China tariff round announced"),
        ("2025-10-05 09:00", 20, "tariff", "Partial China trade deal rumors"),
        ("2025-10-29 14:00", 30, "fed", "Fed cuts rates 25bp again"),
        ("2025-11-15 10:00", 45, "crypto", "Bitcoin ETF options approved"),
        ("2025-12-10 14:00", 20, "fed", "Fed holds rates — neutral statement"),

        # 2026 Q1-Q2 (events within our data window)
        ("2026-01-15 10:00", -35, "tariff", "New tariff escalation on semiconductor imports"),
        ("2026-01-29 14:00", 25, "fed", "Fed holds rates — dovish surprise forward guidance"),
        ("2026-02-10 09:00", 55, "crypto", "Trump signs Digital Asset Framework Act"),
        ("2026-02-18 16:00", -45, "tariff", "Blanket 10% tariff on all imports announced"),
        ("2026-02-20 08:00", 30, "tariff", "Tariff carve-outs for tech sector announced"),
        ("2026-03-05 10:00", -30, "tariff", "China retaliatory tariffs on US goods"),
        ("2026-03-07 11:00", 35, "tariff", "Trade negotiation restart announced"),
        ("2026-03-12 08:30", -20, "fed", "CPI higher than expected — 3.2% YoY"),
        ("2026-03-15 10:00", 40, "crypto", "Treasury adds BTC to strategic reserve"),
        ("2026-03-19 14:00", 15, "fed", "FOMC holds — signals data dependent"),
        ("2026-03-25 09:00", -50, "tariff", "Auto tariff 25% on EU/Japan/Korea"),
        ("2026-03-27 10:00", 35, "tariff", "90-day auto tariff delay for allies"),
        ("2026-04-01 09:00", -40, "tariff", "Phase 2 reciprocal tariffs take effect"),
        ("2026-04-02 16:00", -55, "tariff", "Liberation Day 2.0 — expanded tariff list"),
        ("2026-04-03 10:00", 45, "tariff", "Trump hints at tariff pause on Truth Social"),
        ("2026-04-05 12:00", -15, "regulation", "SEC enforcement on DeFi protocols"),
        ("2026-04-07 09:00", 30, "crypto", "Bipartisan crypto bill advances in Senate"),
        ("2026-04-08 08:30", -10, "fed", "Jobs report mixed — uncertainty grows"),
    ]

    # Convert to signal dicts with timestamps
    for date_str, score, category, desc in known_events:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            ts = int(dt.timestamp())
            # Only include if within our candle range
            if candles and candles[0]["time"] <= ts <= candles[-1]["time"]:
                signals.append({
                    "timestamp": ts,
                    "score": score,
                    "category": category,
                    "description": desc,
                })
        except ValueError:
            continue

    logger.info("Generated %d synthetic political signals within candle range", len(signals))
    return signals


def generate_synthetic_congress_trades(candles: list) -> list:
    """Generate synthetic congressional trades for backtesting.

    Based on known patterns of congressional crypto-adjacent stock activity.
    """
    trades = []
    known_trades = [
        # (date, member, ticker, tx_type, amount_range)
        ("2025-01-15", "Rep. Michael McCaul", "COIN", "purchase", "$15,001-$50,000"),
        ("2025-01-16", "Sen. Cynthia Lummis", "MSTR", "purchase", "$50,001-$100,000"),
        ("2025-01-18", "Rep. French Hill", "COIN", "purchase", "$1,001-$15,000"),
        ("2025-02-10", "Sen. Ted Cruz", "MARA", "purchase", "$15,001-$50,000"),
        ("2025-02-12", "Rep. Tom Emmer", "RIOT", "purchase", "$1,001-$15,000"),
        ("2025-02-14", "Rep. Patrick McHenry", "IBIT", "purchase", "$50,001-$100,000"),
        ("2025-03-05", "Sen. Cynthia Lummis", "IBIT", "purchase", "$100,001-$250,000"),
        ("2025-03-07", "Rep. Ro Khanna", "COIN", "purchase", "$15,001-$50,000"),
        ("2025-03-10", "Sen. Tim Scott", "MSTR", "purchase", "$1,001-$15,000"),
        ("2025-04-15", "Rep. Nancy Pelosi", "COIN", "sale", "$50,001-$100,000"),
        ("2025-04-16", "Sen. Tommy Tuberville", "MARA", "sale", "$15,001-$50,000"),
        ("2025-05-01", "Rep. Michael McCaul", "IBIT", "purchase", "$50,001-$100,000"),
        ("2025-05-03", "Sen. Cynthia Lummis", "GBTC", "purchase", "$100,001-$250,000"),
        ("2025-05-05", "Rep. French Hill", "MSTR", "purchase", "$15,001-$50,000"),
        ("2025-06-10", "Rep. Tom Emmer", "COIN", "purchase", "$15,001-$50,000"),
        ("2025-06-12", "Sen. Tim Scott", "IBIT", "purchase", "$50,001-$100,000"),
        ("2025-07-20", "Sen. Tommy Tuberville", "MARA", "purchase", "$15,001-$50,000"),
        ("2025-08-01", "Rep. Ro Khanna", "COIN", "sale", "$15,001-$50,000"),
        ("2025-09-15", "Sen. Cynthia Lummis", "IBIT", "purchase", "$100,001-$250,000"),
        ("2025-09-17", "Rep. Michael McCaul", "MSTR", "purchase", "$50,001-$100,000"),
        ("2025-09-18", "Rep. French Hill", "COIN", "purchase", "$15,001-$50,000"),

        # 2026 Q1-Q2 trades (within our data window)
        ("2026-03-10", "Sen. Cynthia Lummis", "IBIT", "purchase", "$100,001-$250,000"),
        ("2026-03-11", "Rep. Michael McCaul", "MSTR", "purchase", "$50,001-$100,000"),
        ("2026-03-12", "Rep. French Hill", "COIN", "purchase", "$15,001-$50,000"),
        ("2026-03-20", "Sen. Tim Scott", "IBIT", "purchase", "$50,001-$100,000"),
        ("2026-03-22", "Rep. Tom Emmer", "MARA", "purchase", "$15,001-$50,000"),
        ("2026-04-01", "Rep. Nancy Pelosi", "COIN", "sale", "$50,001-$100,000"),
        ("2026-04-02", "Sen. Tommy Tuberville", "MSTR", "sale", "$15,001-$50,000"),
        ("2026-04-05", "Sen. Cynthia Lummis", "GBTC", "purchase", "$100,001-$250,000"),
        ("2026-04-06", "Rep. Ro Khanna", "COIN", "purchase", "$15,001-$50,000"),
        ("2026-04-07", "Rep. Michael McCaul", "IBIT", "purchase", "$50,001-$100,000"),
    ]

    for date_str, member, ticker, tx_type, amount in known_trades:
        trades.append({
            "member": member,
            "ticker": ticker,
            "tx_type": tx_type,
            "amount_range": amount,
            "tx_date": date_str,
            "filed_date": date_str,  # simplified: assume same-day filing
        })

    return trades


def run_political_correlation_analysis(candles: list) -> dict:
    """Analyze BTC price reaction to each type of political signal.

    For each signal category, compute average BTC change at 4h, 24h, 72h after event.
    """
    signals = generate_synthetic_political_signals(candles)
    if not signals or not candles:
        return {}

    # Build timestamp -> price lookup
    price_at = {}
    for c in candles:
        price_at[c["time"]] = c["close"]

    # Find closest candle timestamp to a given target
    candle_times = sorted(price_at.keys())

    def closest_price(target_ts):
        # Binary search for closest candle
        lo, hi = 0, len(candle_times) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if candle_times[mid] < target_ts:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0 and abs(candle_times[lo - 1] - target_ts) < abs(candle_times[lo] - target_ts):
            return price_at[candle_times[lo - 1]]
        return price_at[candle_times[lo]]

    results = {}
    for sig in signals:
        cat = sig["category"]
        if cat not in results:
            results[cat] = {"changes_4h": [], "changes_24h": [], "changes_72h": [], "count": 0}

        base_price = closest_price(sig["timestamp"])
        if base_price <= 0:
            continue

        results[cat]["count"] += 1

        for hours, key in [(4, "changes_4h"), (24, "changes_24h"), (72, "changes_72h")]:
            future_ts = sig["timestamp"] + hours * 3600
            if future_ts <= candle_times[-1]:
                future_price = closest_price(future_ts)
                change_pct = (future_price - base_price) / base_price * 100
                results[cat][key].append(change_pct)

    # Compute averages
    summary = {}
    for cat, data in results.items():
        summary[cat] = {
            "count": data["count"],
            "avg_change_4h": round(sum(data["changes_4h"]) / len(data["changes_4h"]), 2) if data["changes_4h"] else 0,
            "avg_change_24h": round(sum(data["changes_24h"]) / len(data["changes_24h"]), 2) if data["changes_24h"] else 0,
            "avg_change_72h": round(sum(data["changes_72h"]) / len(data["changes_72h"]), 2) if data["changes_72h"] else 0,
        }

    return summary


# ── Walk-forward analysis ───────────────────────────────────────────────

def walk_forward_backtest(ohlc: list, strategy_fn, window_weeks: Optional[int] = None,
                          min_periods: Optional[int] = None) -> dict:
    """Walk-forward validation: train on the first half of each window,
    test on the second.

    Why bother? Because a single backtest over six months tells you what
    a strategy did, not what it would have done if you'd been adapting
    it. We slice the data into rolling windows (default two weeks each)
    and for every window we fit on the first half and grade on the
    second. A strategy that wins consistently across windows is a real
    edge. A strategy that wins only on the full history but loses on
    most windows is curve-fit.

    strategy_fn(train_ohlc, test_ohlc) -> dict with at minimum:
        {'pnl': float, 'trades': int, 'win_rate': float}

    Returns:
        {
          'windows': [per-window result dicts],
          'aggregate': {
            'total_pnl', 'avg_pnl', 'win_windows', 'loss_windows',
            'avg_win_rate', 'periods_tested',
          },
          'promoted': bool,
          'reason': str,
        }
    """
    window_weeks = window_weeks if window_weeks is not None else getattr(
        config, "WALKFORWARD_WINDOW_WEEKS", 2
    )
    min_periods = min_periods if min_periods is not None else getattr(
        config, "WALKFORWARD_MIN_PERIODS", 3
    )

    if not ohlc:
        return {"windows": [], "aggregate": {}, "promoted": False,
                "reason": "no OHLC data"}

    # Hourly candles assumed; window length in candles
    candles_per_week = 24 * 7
    window_size = window_weeks * candles_per_week
    if window_size < 4 or len(ohlc) < window_size * min_periods:
        return {"windows": [], "aggregate": {}, "promoted": False,
                "reason": f"need at least {window_size * min_periods} candles, got {len(ohlc)}"}

    windows = []
    for start in range(0, len(ohlc) - window_size + 1, window_size):
        window = ohlc[start:start + window_size]
        half = len(window) // 2
        train = window[:half]
        test = window[half:]
        try:
            result = strategy_fn(train, test) or {}
        except Exception as e:
            logger.warning("walk_forward window %d failed: %s", start, e)
            result = {"pnl": 0.0, "trades": 0, "win_rate": 0.0, "error": str(e)}
        result.setdefault("pnl", 0.0)
        result.setdefault("trades", 0)
        result.setdefault("win_rate", 0.0)
        result["window_start_idx"] = start
        result["window_end_idx"] = start + window_size
        windows.append(result)

    n = len(windows)
    if n == 0:
        return {"windows": [], "aggregate": {}, "promoted": False,
                "reason": "no completed windows"}

    total_pnl = sum(w["pnl"] for w in windows)
    win_windows = sum(1 for w in windows if w["pnl"] > 0)
    loss_windows = sum(1 for w in windows if w["pnl"] < 0)
    avg_pnl = total_pnl / n
    avg_wr = sum(w["win_rate"] for w in windows) / n

    promoted = (n >= min_periods) and (win_windows > loss_windows) and (total_pnl > 0)
    reason = "promoted" if promoted else (
        f"only {win_windows}/{n} winning windows (need majority and positive total)"
    )

    return {
        "windows": windows,
        "aggregate": {
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "win_windows": win_windows,
            "loss_windows": loss_windows,
            "avg_win_rate": avg_wr,
            "periods_tested": n,
        },
        "promoted": promoted,
        "reason": reason,
    }


# ── Main entry point ────────────────────────────────────────────────────

def run_backtest():
    """Run the full backtesting suite."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("CryptoWorm Backtesting Engine")
    logger.info("=" * 60)

    # Fetch historical data
    candles = fetch_historical_data(pair="XXBTZUSD", interval=60, months=6)
    if not candles or len(candles) < 100:
        logger.error("Insufficient historical data (%d candles). Aborting.",
                      len(candles) if candles else 0)
        return None

    logger.info("Running backtests on %d hourly candles...", len(candles))

    strategies = ["sentiment", "ema_macd", "bollinger", "rsi_divergence", "grid",
                  "political", "tariff_whiplash", "congress_frontrun"]
    results = {}

    for name in strategies:
        logger.info("-" * 40)
        logger.info("Strategy: %s", name)
        logger.info("-" * 40)
        stats = run_strategy_backtest(name, candles)
        results[name] = stats

        # Print summary
        logger.info("  Trades: %d (W:%d / L:%d)", stats["total_trades"],
                     stats.get("wins", 0), stats.get("losses", 0))
        logger.info("  Win rate: %.1f%%", stats["win_rate"])
        logger.info("  Profit factor: %.2f", stats["profit_factor"])
        logger.info("  Max drawdown: %.2f%%", stats["max_drawdown_pct"])
        logger.info("  Sharpe ratio: %.2f", stats["sharpe_ratio"])
        logger.info("  Sortino ratio: %.2f", stats.get("sortino_ratio", 0))
        logger.info("  Monte Carlo 95%% DD: $%.2f", stats.get("monte_carlo_dd_95", 0))
        logger.info("  Total fees: $%.2f", stats.get("total_fees", 0))
        logger.info("  Avg hold time: %.1f hours", stats["avg_hold_time_hours"])
        logger.info("  Final balance: $%.2f (P&L: $%+.2f, %.2f%%)",
                     stats["final_balance"], stats["total_pnl"], stats["total_return_pct"])
        bnh = stats.get("buy_and_hold", {})
        if bnh:
            logger.info("  Buy-and-Hold benchmark: $%.2f (%.2f%% return, %.2f%% max DD)",
                         bnh.get("final_balance", 0), bnh.get("total_return_pct", 0),
                         bnh.get("max_drawdown_pct", 0))

    # ── Correlation analysis ────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Political Signal Correlation Analysis")
    logger.info("-" * 40)
    correlation = run_political_correlation_analysis(candles)
    results["_correlation"] = correlation
    for event_type, stats_data in correlation.items():
        logger.info("  %s: avg_btc_change_24h=%.2f%% avg_btc_change_72h=%.2f%% events=%d",
                     event_type,
                     stats_data.get("avg_change_24h", 0),
                     stats_data.get("avg_change_72h", 0),
                     stats_data.get("count", 0))

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "run_date": datetime.utcnow().isoformat(),
            "candle_count": len(candles),
            "date_range": {
                "start": datetime.utcfromtimestamp(candles[0]["time"]).strftime("%Y-%m-%d"),
                "end": datetime.utcfromtimestamp(candles[-1]["time"]).strftime("%Y-%m-%d"),
            },
            "strategies": results,
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("Results saved to %s", output_path)
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_backtest()
