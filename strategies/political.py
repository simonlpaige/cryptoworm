"""
Strategy 6: Political / Macro Signals
======================================
Aggregates political and macroeconomic signals to generate trading decisions:

A. Trump Truth Social / Presidential Announcements
   - Keyword scanning for tariff, crypto, fed, regulation themes
   - Scored -100 to +100 with 4-hour decay window
B. Congressional Trading (PTRs)
   - Crypto-adjacent stock purchases/sales by Congress members
C. Federal Reserve / Macro Calendar
   - FOMC, CPI, jobs, GDP release proximity
D. SEC EDGAR Institutional Filings
   - 13F filings showing institutional crypto exposure

Each sub-signal contributes to a composite score that drives buy/sell decisions.
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional, List

import requests

import config
from utils.risk_manager import RiskManager
from utils.congress_trades import CongressTradesProvider
from utils.fed_signals import FedSignalProvider
from utils.sec_filings import SecFilingsProvider

logger = logging.getLogger("cryptobot.political")

# ── Signal scoring keywords ──────────────────────────────────────────────

TARIFF_KEYWORDS = {
    "tariff": -40, "trade war": -50, "china": -20, "import": -15,
    "duty": -30, "reciprocal": -35, "trade deficit": -25,
    "trade deal": 20, "trade agreement": 25, "tariff pause": 40,
    "tariff delay": 35, "tariff exemption": 30,
}

CRYPTO_KEYWORDS = {
    "bitcoin": 50, "crypto": 40, "digital asset": 35,
    "blockchain": 30, "strategic reserve": 60, "bitcoin reserve": 70,
    "crypto executive order": 55, "digital gold": 45,
    "ban crypto": -60, "ban bitcoin": -70,
}

FED_KEYWORDS = {
    "federal reserve": -10, "interest rate": -10, "powell": -5,
    "rate cut": 40, "rate hike": -40, "rate pause": 15,
    "quantitative easing": 30, "quantitative tightening": -30,
    "inflation": -15, "deflation": 10,
}

REGULATION_KEYWORDS = {
    "regulation": -10, "sec": -15, "ban": -40,
    "executive order": 10, "deregulation": 30,
    "enforcement action": -35, "crypto regulation": -20,
    "stablecoin bill": 15, "crypto bill": 20,
    "approve": 25, "approval": 25,
}

ALL_KEYWORD_MAPS = {
    "tariff": TARIFF_KEYWORDS,
    "crypto": CRYPTO_KEYWORDS,
    "fed": FED_KEYWORDS,
    "regulation": REGULATION_KEYWORDS,
}

# Historical signals file for backtesting
HISTORICAL_SIGNALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "data", "political_historical_signals.json")

# ── Trump Signal Provider ────────────────────────────────────────────────


class TrumpSignalProvider:
    """Fetches and scores Trump / presidential announcements."""

    def __init__(self):
        self._cache = []
        self._last_fetch = None
        self._fetch_interval = 900  # 15 minutes

    def fetch_recent_posts(self) -> list:
        """Fetch recent posts from Truth Social public profile.

        Returns list of {text, timestamp, source}. Falls back gracefully
        if the endpoint is unavailable.
        """
        now = datetime.utcnow()
        if self._last_fetch and (now - self._last_fetch).total_seconds() < self._fetch_interval:
            return self._cache

        self._last_fetch = now
        posts = []

        # Try Truth Social RSS/public profile
        try:
            resp = requests.get(
                "https://truthsocial.com/@realDonaldTrump.rss",
                timeout=15,
                headers={"User-Agent": "CryptoBot/1.0 (research)"},
            )
            if resp.status_code == 200:
                posts.extend(self._parse_rss(resp.text))
        except requests.exceptions.RequestException as e:
            logger.debug("Truth Social fetch failed: %s", e)

        if posts:
            self._cache = posts

        return self._cache

    def _parse_rss(self, xml_text: str) -> list:
        """Parse RSS/Atom feed for posts."""
        posts = []
        # Simple regex-based XML parsing (avoid lxml dependency)
        items = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)
        if not items:
            items = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)

        for item in items:
            title = re.search(r"<title[^>]*>(.*?)</title>", item, re.DOTALL)
            desc = re.search(r"<description[^>]*>(.*?)</description>", item, re.DOTALL)
            content = re.search(r"<content[^>]*>(.*?)</content>", item, re.DOTALL)
            pub_date = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", item, re.DOTALL)
            updated = re.search(r"<updated[^>]*>(.*?)</updated>", item, re.DOTALL)

            text = ""
            if desc:
                text = re.sub(r"<[^>]+>", "", desc.group(1)).strip()
            elif content:
                text = re.sub(r"<[^>]+>", "", content.group(1)).strip()
            elif title:
                text = re.sub(r"<[^>]+>", "", title.group(1)).strip()

            timestamp = ""
            if pub_date:
                timestamp = pub_date.group(1).strip()
            elif updated:
                timestamp = updated.group(1).strip()

            if text:
                posts.append({
                    "text": text,
                    "timestamp": timestamp,
                    "source": "truth_social",
                })

        return posts

    def load_historical_signals(self) -> list:
        """Load historical political signals from JSON for backtesting.

        Expected format: list of {date, text, source, category}
        """
        if not os.path.exists(HISTORICAL_SIGNALS_PATH):
            return []
        try:
            with open(HISTORICAL_SIGNALS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load historical signals: %s", e)
            return []

    @staticmethod
    def score_text(text: str) -> dict:
        """Score a text for crypto-market-moving sentiment.

        Returns: {score: -100 to +100, category: str, matched_keywords: list}
        """
        text_lower = text.lower()
        total_score = 0
        matched = []
        categories_hit = {}

        for category, keywords in ALL_KEYWORD_MAPS.items():
            for keyword, weight in keywords.items():
                if keyword.lower() in text_lower:
                    total_score += weight
                    matched.append(keyword)
                    categories_hit[category] = categories_hit.get(category, 0) + abs(weight)

        # Clamp to -100..+100
        total_score = max(-100, min(100, total_score))

        # Determine primary category
        primary_category = max(categories_hit, key=categories_hit.get) if categories_hit else "none"

        return {
            "score": total_score,
            "category": primary_category,
            "matched_keywords": matched,
        }


# ── Active Signal Tracker ────────────────────────────────────────────────


class ActiveSignal:
    """Represents a decaying political signal."""

    def __init__(self, score: int, category: str, timestamp: datetime,
                 decay_hours: float = 4.0):
        self.initial_score = score
        self.category = category
        self.timestamp = timestamp
        self.decay_hours = decay_hours

    def current_score(self, now: datetime = None) -> float:
        """Get decayed score. Linear decay over decay_hours window."""
        if now is None:
            now = datetime.utcnow()
        elapsed = (now - self.timestamp).total_seconds() / 3600
        if elapsed >= self.decay_hours:
            return 0.0
        decay_factor = 1.0 - (elapsed / self.decay_hours)
        return self.initial_score * decay_factor

    def is_expired(self, now: datetime = None) -> bool:
        if now is None:
            now = datetime.utcnow()
        return (now - self.timestamp).total_seconds() / 3600 >= self.decay_hours


# ── Main Political Signals Strategy ──────────────────────────────────────


class PoliticalSignals:
    """Strategy 6: Political and macroeconomic signal aggregation.

    Compatible with the existing strategy interface:
    - __init__(kraken_client, risk_manager)
    - evaluate(price) -> list of actions
    """

    def __init__(self, kraken_client, risk_manager: RiskManager):
        self.kraken = kraken_client
        self.risk = risk_manager
        self.trump = TrumpSignalProvider()
        self.congress = CongressTradesProvider()
        self.fed = FedSignalProvider()
        self.sec = SecFilingsProvider()
        self._active_signals = []
        self._last_eval = None
        self._eval_interval = 900  # evaluate every 15 min
        self._cooldown_until = None
        self._position_id = None
        self._signal_threshold = getattr(config, "POLITICAL_SIGNAL_THRESHOLD", 50)
        self._decay_hours = getattr(config, "POLITICAL_DECAY_HOURS", 4.0)
        # Feature toggles
        self._enable_trump = getattr(config, "POLITICAL_ENABLE_TRUMP", True)
        self._enable_congress = getattr(config, "POLITICAL_ENABLE_CONGRESS", True)
        self._enable_fed = getattr(config, "POLITICAL_ENABLE_FED", True)
        self._enable_sec = getattr(config, "POLITICAL_ENABLE_SEC", True)

    def evaluate(self, price: float) -> list:
        """Run political signal evaluation. Returns list of action dicts."""
        now = datetime.utcnow()
        actions = []

        # Cooldown check
        if self._cooldown_until and now < self._cooldown_until:
            return actions

        # Rate-limit evaluations
        if self._last_eval and (now - self._last_eval).total_seconds() < self._eval_interval:
            return actions
        self._last_eval = now

        # Prune expired signals
        self._active_signals = [s for s in self._active_signals if not s.is_expired(now)]

        # ── Gather new signals ───────────────────────────────────────

        # A. Trump / Presidential
        if self._enable_trump:
            try:
                posts = self.trump.fetch_recent_posts()
                for post in posts:
                    scored = TrumpSignalProvider.score_text(post["text"])
                    if abs(scored["score"]) >= 15:  # minimum relevance
                        self._active_signals.append(ActiveSignal(
                            score=scored["score"],
                            category=scored["category"],
                            timestamp=now,
                            decay_hours=self._decay_hours,
                        ))
                        logger.info("Trump signal: score=%d category=%s keywords=%s",
                                    scored["score"], scored["category"], scored["matched_keywords"])
            except Exception as e:
                logger.debug("Trump signal fetch error: %s", e)

        # B. Congressional trades
        if self._enable_congress:
            try:
                congress_sig = self.congress.generate_signal()
                if congress_sig["signal"] != "neutral":
                    score = congress_sig["strength"] if congress_sig["signal"] == "buy" else -congress_sig["strength"]
                    self._active_signals.append(ActiveSignal(
                        score=score, category="congress", timestamp=now,
                        decay_hours=self._decay_hours * 6,  # congressional signals last longer
                    ))
                    logger.info("Congress signal: %s strength=%d buys=%d sells=%d",
                                congress_sig["signal"], congress_sig["strength"],
                                congress_sig["buy_count"], congress_sig["sell_count"])
            except Exception as e:
                logger.debug("Congress signal error: %s", e)

        # C. Fed / Macro
        if self._enable_fed:
            try:
                fed_sig = self.fed.generate_signal()
                if fed_sig["reduce_size"]:
                    # Close any open political position before major macro event
                    if self._position_id:
                        result = self.risk.close_position(self._position_id, price)
                        if result:
                            actions.append({"action": "close", "reason": "macro_event_approaching",
                                            "details": fed_sig["reason"]})
                            self._position_id = None
                    logger.info("Fed signal: reduce size — %s", fed_sig["reason"])
                if fed_sig["signal"] != "neutral":
                    score = fed_sig["strength"] if fed_sig["signal"] == "buy" else -fed_sig["strength"]
                    self._active_signals.append(ActiveSignal(
                        score=score, category="fed", timestamp=now,
                        decay_hours=self._decay_hours * 2,
                    ))
            except Exception as e:
                logger.debug("Fed signal error: %s", e)

        # D. SEC institutional filings
        if self._enable_sec:
            try:
                sec_sig = self.sec.generate_signal()
                if sec_sig["signal"] != "neutral":
                    score = sec_sig["strength"] if sec_sig["signal"] == "buy" else -sec_sig["strength"]
                    self._active_signals.append(ActiveSignal(
                        score=score, category="sec", timestamp=now,
                        decay_hours=self._decay_hours * 12,  # institutional signals persist
                    ))
                    logger.info("SEC signal: %s strength=%d institutions=%s",
                                sec_sig["signal"], sec_sig["strength"], sec_sig["institutions"])
            except Exception as e:
                logger.debug("SEC signal error: %s", e)

        # ── Compute composite score ──────────────────────────────────

        composite = sum(s.current_score(now) for s in self._active_signals)
        logger.info("Political composite score: %.1f (active signals: %d, threshold: ±%d)",
                     composite, len(self._active_signals), self._signal_threshold)

        # ── Trading decisions ────────────────────────────────────────

        if composite >= self._signal_threshold and self._position_id is None:
            can_open, reason = self.risk.can_open_position(price, side="buy", strategy="political")
        elif composite <= -self._signal_threshold and self._position_id is None:
            can_open, reason = self.risk.can_open_position(price, side="sell", strategy="political")
        else:
            can_open, reason = True, "OK"  # no trade planned, skip check

        if composite >= self._signal_threshold and self._position_id is None and can_open:
            # Bullish signal → open long
            # Strong signal (>70): use wider take profit (6% instead of 4%)
            size_btc = self.risk.position_size_btc(price)
            sl = price * (1 - config.STOP_LOSS_PCT / 100)
            tp_pct = 0.06 if composite > 70 else 0.04
            tp = price * (1 + tp_pct)
            pos = self.risk.open_position("buy", price, size_btc, "political", sl, tp)
            self._position_id = pos["id"]
            self._cooldown_until = now + timedelta(hours=2)
            actions.append({"action": "buy", "price": price, "size_btc": size_btc,
                            "reason": f"political_bullish (composite={composite:.0f})"})
            logger.info("POLITICAL BUY at $%.2f | composite=%.0f tp=%.1f%%", price, composite, tp_pct * 100)

        elif composite <= -self._signal_threshold and self._position_id is None and can_open:
            # Bearish signal → open short
            # Strong signal (<-70): use wider take profit (6% instead of 4%)
            size_btc = self.risk.position_size_btc(price)
            sl = price * (1 + config.STOP_LOSS_PCT / 100)
            tp_pct = 0.06 if composite < -70 else 0.04
            tp = price * (1 - tp_pct)
            pos = self.risk.open_position("sell", price, size_btc, "political", sl, tp)
            self._position_id = pos["id"]
            self._cooldown_until = now + timedelta(hours=2)
            actions.append({"action": "sell", "price": price, "size_btc": size_btc,
                            "reason": f"political_bearish (composite={composite:.0f})"})
            logger.info("POLITICAL SELL at $%.2f | composite=%.0f tp=%.1f%%", price, composite, tp_pct * 100)

        elif self._position_id is not None:
            # Check if we should close based on score reversal
            pos = None
            for p in self.risk.positions:
                if p["id"] == self._position_id and p["status"] == "open":
                    pos = p
                    break

            if pos:
                if pos["side"] == "buy" and composite <= 0:
                    result = self.risk.close_position(self._position_id, price)
                    if result:
                        actions.append({"action": "close", "reason": "signal_reversed",
                                        "pnl": result.get("pnl", 0)})
                        self._position_id = None
                elif pos["side"] == "sell" and composite >= 0:
                    result = self.risk.close_position(self._position_id, price)
                    if result:
                        actions.append({"action": "close", "reason": "signal_reversed",
                                        "pnl": result.get("pnl", 0)})
                        self._position_id = None
            else:
                self._position_id = None  # position was closed by SL/TP

        return actions

    def evaluate_backtest(self, price: float, timestamp: int, synthetic_signals: list) -> list:
        """Evaluate with synthetic historical signals for backtesting.

        Args:
            price: Current BTC price
            timestamp: Unix timestamp of current candle
            synthetic_signals: List of {timestamp, score, category} dicts
        """
        now = datetime.utcfromtimestamp(timestamp)
        actions = []

        # Prune expired signals
        self._active_signals = [s for s in self._active_signals if not s.is_expired(now)]

        # Cooldown check
        if self._cooldown_until and now < self._cooldown_until:
            pass
        else:
            # Inject any synthetic signals that fall within the current window
            for sig in synthetic_signals:
                sig_time = datetime.utcfromtimestamp(sig["timestamp"])
                hours_ago = (now - sig_time).total_seconds() / 3600
                if 0 <= hours_ago <= self._decay_hours * 2:
                    # Check if we already have this signal
                    already = any(s.timestamp == sig_time and s.category == sig["category"]
                                  for s in self._active_signals)
                    if not already:
                        self._active_signals.append(ActiveSignal(
                            score=sig["score"],
                            category=sig["category"],
                            timestamp=sig_time,
                            decay_hours=self._decay_hours,
                        ))

        # Compute composite
        composite = sum(s.current_score(now) for s in self._active_signals)

        if composite >= self._signal_threshold and self._position_id is None:
            can_open, reason = self.risk.can_open_position(price, side="buy", strategy="political")
        elif composite <= -self._signal_threshold and self._position_id is None:
            can_open, reason = self.risk.can_open_position(price, side="sell", strategy="political")
        else:
            can_open, reason = True, "OK"

        if composite >= self._signal_threshold and self._position_id is None and can_open:
            size_btc = self.risk.position_size_btc(price)
            sl = price * (1 - config.STOP_LOSS_PCT / 100)
            tp_pct = 0.06 if composite > 70 else 0.04
            tp = price * (1 + tp_pct)
            pos = self.risk.open_position("buy", price, size_btc, "political", sl, tp)
            self._position_id = pos["id"]
            self._cooldown_until = now + timedelta(hours=2)
            actions.append({"action": "buy", "price": price, "composite": composite})

        elif composite <= -self._signal_threshold and self._position_id is None and can_open:
            size_btc = self.risk.position_size_btc(price)
            sl = price * (1 + config.STOP_LOSS_PCT / 100)
            tp_pct = 0.06 if composite < -70 else 0.04
            tp = price * (1 - tp_pct)
            pos = self.risk.open_position("sell", price, size_btc, "political", sl, tp)
            self._position_id = pos["id"]
            self._cooldown_until = now + timedelta(hours=2)
            actions.append({"action": "sell", "price": price, "composite": composite})

        elif self._position_id is not None:
            pos = None
            for p in self.risk.positions:
                if p["id"] == self._position_id and p["status"] == "open":
                    pos = p
                    break
            if pos:
                if (pos["side"] == "buy" and composite <= 0) or \
                   (pos["side"] == "sell" and composite >= 0):
                    result = self.risk.close_position(self._position_id, price)
                    if result:
                        actions.append({"action": "close", "pnl": result.get("pnl", 0)})
                        self._position_id = None
            else:
                self._position_id = None

        return actions
