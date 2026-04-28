#!/usr/bin/env python3
"""
CryptoBot — Paper Trading Bot
==============================
Connects to Kraken for real-time BTC/USD prices.
Runs five strategies:
  1. Grid Bot (passive, ranging markets)
  2. Sentiment Swing (Fear & Greed driven)
  3. EMA/MACD Momentum (trending markets, ADX>25)
  4. Bollinger Mean Reversion (ranging markets, ADX<30)
  5. RSI Divergence (reversal catching, 4h timeframe)
Simulates trades with a virtual $500 balance.
NO real orders are ever placed.

Usage:
    pip install -r requirements.txt
    python bot.py
"""
import sys
import os
import time
import logging
import signal
from datetime import datetime, timezone

# Ensure project root is on path
_bot_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _bot_dir)

# Load .env file if present (for API keys)
_env_file = os.path.join(_bot_dir, ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Handle --backtest before importing config (which requires API keys)
if "--backtest" in sys.argv:
    if not os.environ.get("KRAKEN_API_KEY"):
        os.environ["KRAKEN_API_KEY"] = "backtest-dummy"
    if not os.environ.get("KRAKEN_PRIVATE_KEY"):
        os.environ["KRAKEN_PRIVATE_KEY"] = "backtest-dummy"

import config
from utils.logger import setup_logging, log_daily_summary
from utils.kraken_client import KrakenClient
from utils.risk_manager import RiskManager
from strategies.grid import GridBot
from strategies.sentiment import SentimentSwing
from strategies.ema_macd import EmaMacdMomentum
from strategies.bollinger import BollingerMeanReversion
from strategies.rsi_divergence import RsiDivergence
from strategies.political import PoliticalSignals
from strategies.novel import TariffWhiplashStrategy, CongressionalFrontRunStrategy
from strategies.regime import RegimeDetector
from strategies.ml_signal import MLSignalGenerator
from utils.features import compute_features
from utils.funding_rate import FundingRateMonitor
from trainer.engine import run_cycle, load_training_state, save_training_state
from manager.health import full_health_check, format_health_report
from manager.researcher import run_full_research, format_research_report

logger = logging.getLogger("cryptobot.main")

# ── Graceful shutdown ────────────────────────────────────────────────────
_running = True

def _shutdown(sig, frame):
    global _running
    logger.info("Shutdown signal received — stopping bot...")
    _running = False

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("CryptoBot Paper Trading Bot starting")
    logger.info("  Balance: $%.2f (virtual)", config.INITIAL_BALANCE)
    logger.info("  Pair: %s", config.PAIR_DISPLAY)
    logger.info("  Interval: %ds", config.CHECK_INTERVAL_SECONDS)
    logger.info("  Mode: PAPER TRADING ONLY — no real orders")
    logger.info("=" * 60)

    # Initialize components
    kraken = KrakenClient()
    risk = RiskManager()
    regime_detector = RegimeDetector(kraken)
    grid = GridBot(kraken, risk)
    sentiment = SentimentSwing(kraken, risk)
    ema_macd = EmaMacdMomentum(kraken, risk) if getattr(config, "ENABLE_EMA_MACD", True) else None
    bollinger = BollingerMeanReversion(kraken, risk) if getattr(config, "ENABLE_BOLLINGER", True) else None
    rsi_div = RsiDivergence(kraken, risk) if getattr(config, "ENABLE_RSI_DIVERGENCE", True) else None
    political = PoliticalSignals(kraken, risk)
    tariff_whiplash = TariffWhiplashStrategy(kraken, risk) if getattr(config, "ENABLE_TARIFF_WHIPLASH", True) else None
    congress_frontrun = CongressionalFrontRunStrategy(kraken, risk) if getattr(config, "ENABLE_CONGRESS_FRONTRUN", True) else None

    # ML Signal Generator and Funding Rate Monitor
    ml_signal = None
    if getattr(config, "ENABLE_ML_SIGNAL", False):
        ml_signal = MLSignalGenerator(
            retrain_interval=getattr(config, "ML_RETRAIN_INTERVAL_TICKS", 288),
            min_history=getattr(config, "ML_MIN_HISTORY", 100),
            confidence_threshold=getattr(config, "ML_CONFIDENCE_THRESHOLD", 0.6),
        )

    funding_monitor = None
    if getattr(config, "ENABLE_FUNDING_MONITOR", False):
        funding_monitor = FundingRateMonitor()

    enabled = [s for s, v in [("Grid", True), ("Sentiment", True),
               ("EMA/MACD", ema_macd), ("Bollinger", bollinger),
               ("RSI Div", rsi_div), ("Political", True),
               ("TariffWhiplash", tariff_whiplash), ("CongressFrontrun", congress_frontrun),
               ("ML Signal", ml_signal), ("Funding Monitor", funding_monitor)] if v]
    logger.info("Active strategies: %s", ", ".join(enabled))

    # Track daily summary
    last_summary_date = None
    tick_count = 0

    # Training engine state
    training_state = load_training_state()
    training_interval_ticks = 12  # run trainer every 12 ticks (~60min at 5min ticks)
    last_training_tick = 0

    # Health check & research intervals
    health_interval_ticks = 36   # health check every 36 ticks (~3h)
    last_health_tick = 0
    research_interval_ticks = 72  # full research every 72 ticks (~6h)
    last_research_tick = 0

    while _running:
        try:
            tick_count += 1
            ticker = kraken.get_ticker()
            if not ticker:
                logger.warning("Could not fetch price — retrying in %ds",
                               config.CHECK_INTERVAL_SECONDS)
                time.sleep(config.CHECK_INTERVAL_SECONDS)
                continue

            price = ticker["last"]
            logger.info("Tick #%d | BTC/USD: $%.2f | Bid: $%.2f | Ask: $%.2f | Balance: $%.2f",
                        tick_count, price, ticker["bid"], ticker["ask"], risk.balance)

            # ── Check existing positions for SL/TP ───────────────────────
            closed = risk.check_stop_loss_take_profit(price)

            # ── Funding rate monitor ─────────────────────────────────────
            funding_summary = {}
            if funding_monitor:
                try:
                    funding_summary = funding_monitor.update()
                    if funding_summary.get("current_rate") is not None:
                        logger.info("Funding rate: %.4f%% (avg: %.4f%%)",
                                    funding_summary["current_rate"] * 100,
                                    (funding_summary.get("avg_rate") or 0) * 100)
                except Exception as fe:
                    logger.warning("Funding rate update failed: %s", fe)

            # ── ML Signal Generator ──────────────────────────────────────
            ml_result = {"signal": "HOLD", "confidence": 0.5}
            if ml_signal:
                try:
                    ohlc_data = kraken.get_ohlc(interval=60, count=200)
                    features = compute_features(
                        ohlc_data,
                        funding_rate=funding_summary.get("current_rate"),
                        funding_rate_avg=funding_summary.get("avg_rate"),
                        funding_rate_trend=funding_summary.get("trend"),
                    ) if ohlc_data else {}
                    ml_result = ml_signal.update(ohlc_data or [], features)
                    logger.info("ML Signal: %s (confidence=%.3f) — %s",
                                ml_result["signal"], ml_result["confidence"],
                                ml_result.get("reason", ""))
                except Exception as mle:
                    logger.warning("ML signal update failed: %s", mle)

            # ── Detect market regime ─────────────────────────────────────
            try:
                regime = regime_detector.update()
            except Exception as re_err:
                logger.warning("Regime detection failed (defaulting to neutral): %s", re_err)
                regime = "neutral"

            # ── Run strategies (filtered by regime + ML signal) ──────────
            if not risk.is_paused:
                # ML signal filter: if ML says SELL, block new BUY entries (and vice versa)
                # If HOLD, let other strategies run normally
                ml_sig = ml_result.get("signal", "HOLD") if ml_signal else "HOLD"
                if ml_sig != "HOLD":
                    logger.info("ML filter active: %s — blocking opposite entries", ml_sig)

                # Grid bot: reinitialize if price moved too far
                if grid.should_reinitialize(price):
                    grid.initialize(price)

                # Regime-based strategy selection:
                #   trending (ADX>25): momentum only (sentiment, rsi_divergence, congress_frontrun)
                #   ranging  (ADX<20): grid only
                #   neutral  (20-25):  all strategies
                grid_actions = []
                sentiment_actions = []
                ema_actions = []
                boll_actions = []
                rsi_actions = []
                political_actions = []
                whiplash_actions = []
                frontrun_actions = []

                if regime in ("ranging", "neutral"):
                    grid_actions = grid.evaluate(price)
                if regime in ("trending", "neutral"):
                    sentiment_actions = sentiment.evaluate(price)
                    rsi_actions = rsi_div.evaluate(price) if rsi_div else []
                    frontrun_actions = congress_frontrun.evaluate(price) if congress_frontrun else []
                if regime == "neutral":
                    # These only run in neutral — disabled strategies still gated by config
                    ema_actions = ema_macd.evaluate(price) if ema_macd else []
                    boll_actions = bollinger.evaluate(price) if bollinger else []
                    political_actions = political.evaluate(price)
                    whiplash_actions = tariff_whiplash.evaluate(price) if tariff_whiplash else []

                # Apply ML signal filter: block new entries that conflict with ML
                if ml_sig == "SELL":
                    # Block new BUY entries (keep closes/exits)
                    for action_list in [grid_actions, sentiment_actions, ema_actions,
                                        boll_actions, rsi_actions, political_actions,
                                        whiplash_actions, frontrun_actions]:
                        filtered = [a for a in action_list
                                    if not (a.get("status") == "open" and a.get("side") == "buy")]
                        blocked = len(action_list) - len(filtered)
                        if blocked:
                            logger.info("ML filter blocked %d BUY entries", blocked)
                        action_list[:] = filtered
                elif ml_sig == "BUY":
                    # Block new SELL entries (keep closes/exits)
                    for action_list in [grid_actions, sentiment_actions, ema_actions,
                                        boll_actions, rsi_actions, political_actions,
                                        whiplash_actions, frontrun_actions]:
                        filtered = [a for a in action_list
                                    if not (a.get("status") == "open" and a.get("side") == "sell")]
                        blocked = len(action_list) - len(filtered)
                        if blocked:
                            logger.info("ML filter blocked %d SELL entries", blocked)
                        action_list[:] = filtered

                for name, acts in [("Grid", grid_actions), ("Sentiment", sentiment_actions),
                                   ("EMA/MACD", ema_actions), ("Bollinger", boll_actions),
                                   ("RSI Div", rsi_actions), ("Political", political_actions),
                                   ("TariffWhiplash", whiplash_actions),
                                   ("CongressFrontrun", frontrun_actions)]:
                    if acts:
                        logger.info("%s: %d actions this tick [regime=%s, ml=%s]",
                                    name, len(acts), regime, ml_sig)
            else:
                logger.warning("Trading PAUSED: %s", risk.pause_reason)

            # ── Recursive training engine ───────────────────────────
            if tick_count - last_training_tick >= training_interval_ticks:
                try:
                    logger.info("─" * 40 + " TRAINING CYCLE " + "─" * 40)
                    report = run_cycle(kraken, training_state)
                    adj_count = report["adjustments"].get("total", 0)
                    if adj_count > 0:
                        logger.info("Trainer tuned %d parameters this cycle", adj_count)
                    last_training_tick = tick_count
                except Exception as te:
                    logger.error("Training cycle failed: %s", te)
                    last_training_tick = tick_count

            # ── Health check (manager) ────────────────────────────────────────
            if tick_count - last_health_tick >= health_interval_ticks:
                try:
                    health = full_health_check()
                    if health["overall"] in ("critical", "error"):
                        logger.error("HEALTH ALERT:\n%s", format_health_report(health))
                    elif health["overall"] == "warning":
                        logger.warning("Health warnings:\n%s", format_health_report(health))
                    else:
                        logger.info("Health check: %s", health["overall"])
                    last_health_tick = tick_count
                except Exception as he:
                    logger.error("Health check failed: %s", he)
                    last_health_tick = tick_count

            # ── Research sweep ────────────────────────────────────────────────
            if tick_count - last_research_tick >= research_interval_ticks:
                try:
                    logger.info("─" * 40 + " RESEARCH SWEEP " + "─" * 40)
                    ohlc_data = kraken.get_ohlc(interval=60, count=100)
                    research = run_full_research(ohlc_data)
                    if research["total_findings"] > 0:
                        logger.info("Research: %d findings\n%s",
                                    research["total_findings"],
                                    format_research_report(research))
                    else:
                        logger.info("Research sweep: no actionable findings")
                    last_research_tick = tick_count
                except Exception as re_err:
                    logger.error("Research sweep failed: %s", re_err)
                    last_research_tick = tick_count

            # ── Daily summary ────────────────────────────────────────────
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if last_summary_date != today and datetime.now(timezone.utc).hour >= 23:
                last_summary_date = today  # set immediately to prevent re-trigger
                summary = risk.get_daily_summary()
                log_daily_summary(summary)
                logger.info("Daily summary: P&L=$%+.2f | Balance=$%.2f | Win rate=%.0f%%",
                            summary["daily_pnl"], summary["balance"], summary["win_rate"])

            # ── Sleep until next tick ────────────────────────────────────
            logger.debug("Sleeping %ds until next tick...", config.CHECK_INTERVAL_SECONDS)
            # Sleep in 1s increments so we can catch shutdown signals
            for _ in range(config.CHECK_INTERVAL_SECONDS):
                if not _running:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Unhandled error in main loop: %s", e)
            time.sleep(30)  # Back off on errors

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Bot stopped. Final balance: $%.2f", risk.balance)
    summary = risk.get_daily_summary()
    log_daily_summary(summary)
    risk.save_state()
    logger.info("State saved. Goodbye.")


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        from trainer.backtester import run_backtest
        run_backtest()
    else:
        main()
