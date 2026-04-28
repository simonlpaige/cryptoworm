"""
Strategy Researcher — continuously discovers new trading techniques
and evaluates them for potential integration.

Sources:
  1. Binance Futures — Funding Rate, Open Interest, Long/Short Ratio (free, no auth)
  2. Fear & Greed historical patterns
  3. On-chain signals (future: Glassnode, CryptoQuant)
  4. Volatility regime analysis
  5. Correlation analysis (BTC vs traditional markets)

The researcher doesn't modify the bot directly. It produces
research reports that the training engine can consume, and
flags promising techniques for review.
"""
import json
import logging
import math
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional

import config

logger = logging.getLogger("cryptobot.manager.researcher")

RESEARCH_DIR = os.path.join(config.BOT_DIR, "manager", "research")
FINDINGS_FILE = os.path.join(RESEARCH_DIR, "findings.json")


def _save_finding(finding: dict):
    """Append a research finding."""
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    findings = []
    if os.path.exists(FINDINGS_FILE):
        with open(FINDINGS_FILE, "r") as f:
            findings = json.load(f)
    findings.append(finding)
    # Keep last 200
    if len(findings) > 200:
        findings = findings[-200:]
    with open(FINDINGS_FILE, "w") as f:
        json.dump(findings, f, indent=2)


# ── Research Module 1: Fear & Greed Pattern Analysis ─────────────────────

def research_fng_patterns() -> Optional[dict]:
    """Analyze Fear & Greed Index patterns over the last 30 days.
    
    Looks for:
    - Sustained extreme readings (>7 days = strong mean reversion signal)
    - Rapid sentiment shifts (>30 points in 7 days = momentum signal)
    - Divergence from price action
    """
    try:
        resp = requests.get("https://api.alternative.me/fng/", params={"limit": 30}, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if len(data) < 14:
            return None

        values = [int(d["value"]) for d in data]
        current = values[0]
        avg_7d = sum(values[:7]) / 7
        avg_30d = sum(values) / len(values)

        # Sustained extreme check
        extreme_fear_days = sum(1 for v in values[:14] if v <= 25)
        extreme_greed_days = sum(1 for v in values[:14] if v >= 75)

        # Rapid shift
        shift_7d = values[0] - values[6] if len(values) > 6 else 0

        findings = []

        if extreme_fear_days >= 7:
            findings.append({
                "signal": "sustained_extreme_fear",
                "days": extreme_fear_days,
                "implication": "Strong mean reversion buy signal — historically BTC rebounds after 7+ days of extreme fear",
                "action": "Consider lowering fear_threshold to catch the rebound entry, or widening take_profit",
                "confidence": 0.7,
            })

        if extreme_greed_days >= 7:
            findings.append({
                "signal": "sustained_extreme_greed",
                "days": extreme_greed_days,
                "implication": "Distribution phase likely — shorts become attractive",
                "action": "Consider raising greed_threshold sensitivity, tighten long take_profits",
                "confidence": 0.65,
            })

        if abs(shift_7d) >= 30:
            direction = "fear_to_greed" if shift_7d > 0 else "greed_to_fear"
            findings.append({
                "signal": f"rapid_shift_{direction}",
                "shift": shift_7d,
                "implication": f"Rapid sentiment swing ({shift_7d:+d} in 7d) — momentum strategies should outperform",
                "action": "Favor EMA/MACD over mean reversion during rapid shifts",
                "confidence": 0.6,
            })

        return {
            "source": "fear_and_greed",
            "timestamp": datetime.utcnow().isoformat(),
            "current": current,
            "avg_7d": round(avg_7d, 1),
            "avg_30d": round(avg_30d, 1),
            "extreme_fear_days_14d": extreme_fear_days,
            "extreme_greed_days_14d": extreme_greed_days,
            "shift_7d": shift_7d,
            "findings": findings,
        }
    except Exception as e:
        logger.error("FNG research failed: %s", e)
        return None


# ── Research Module 2: Volatility Regime Transitions ─────────────────────

def research_volatility_transitions(ohlc: list) -> Optional[dict]:
    """Detect volatility regime transitions.
    
    Key insight from research: strategy performance differs dramatically
    across vol regimes. Detecting transitions EARLY is the edge.
    
    Looks for:
    - Bollinger Band squeeze (bandwidth < 3%) → breakout imminent
    - ATR expansion (rising 3 consecutive days) → momentum setup
    - ATR contraction → mean reversion setup
    """
    if not ohlc or len(ohlc) < 30:
        return None

    closes = [c["close"] for c in ohlc]
    highs = [c["high"] for c in ohlc]
    lows = [c["low"] for c in ohlc]

    # Calculate rolling ATR (14-period) for last 7 periods
    atrs = []
    for end in range(14, min(len(closes), 22)):
        trs = []
        for i in range(end - 13, end + 1):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        atrs.append(sum(trs) / 14)

    # Bollinger bandwidth
    recent_20 = closes[-20:]
    sma = sum(recent_20) / 20
    std = math.sqrt(sum((x - sma) ** 2 for x in recent_20) / 20)
    bb_width = (4 * std) / sma * 100  # 2 std * 2 sides

    findings = []

    # BB squeeze
    if bb_width < 3.0:
        findings.append({
            "signal": "bollinger_squeeze",
            "bb_width_pct": round(bb_width, 2),
            "implication": "Tight BB squeeze — breakout expected within 24-48h. Direction uncertain.",
            "action": "Prepare both long and short entries. Widen stops slightly to avoid fakeout stops.",
            "confidence": 0.65,
        })

    # ATR trend
    if len(atrs) >= 3:
        if atrs[-1] > atrs[-2] > atrs[-3]:
            findings.append({
                "signal": "atr_expanding",
                "implication": "Volatility expanding — momentum strategies favored",
                "action": "Widen stops (ATR-based), increase EMA/MACD weighting",
                "confidence": 0.6,
            })
        elif atrs[-1] < atrs[-2] < atrs[-3]:
            findings.append({
                "signal": "atr_contracting",
                "implication": "Volatility contracting — mean reversion strategies favored",
                "action": "Tighten stops, increase Bollinger/grid weighting",
                "confidence": 0.6,
            })

    return {
        "source": "volatility_analysis",
        "timestamp": datetime.utcnow().isoformat(),
        "bb_width_pct": round(bb_width, 2),
        "atr_trend": "expanding" if len(atrs) >= 3 and atrs[-1] > atrs[-2] else "contracting" if len(atrs) >= 3 and atrs[-1] < atrs[-2] else "stable",
        "findings": findings,
    }


# ── Research Module 3: Binance Funding Rate ─────────────────────────────

def research_funding_rates() -> Optional[dict]:
    """Fetch BTC perpetual futures funding rate from Binance Futures.
    
    Free, no authentication required.
    
    Interpretation:
    - Positive (>0.03%): longs paying shorts → market overleveraged long → reversal risk
    - Negative (<-0.01%): shorts paying longs → potential short squeeze
    - Extreme (>0.05% or <-0.03%) → very high volatility, widen stops
    
    Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=10
    """
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": "BTCUSDT", "limit": 10},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None

        rates = [float(d["fundingRate"]) for d in data]
        current_rate = rates[0]
        avg_rate = sum(rates) / len(rates)
        rate_pct = current_rate * 100

        findings = []

        if current_rate > 0.0003:  # >0.03%
            findings.append({
                "signal": "funding_overleveraged_long",
                "rate_pct": round(rate_pct, 5),
                "implication": f"Funding rate {rate_pct:.4f}% — market overleveraged long. Longs paying heavily. Reversal risk elevated.",
                "action": "Tighten long entry filters. Prepare for mean reversion shorts.",
                "confidence": 0.60,
            })
        elif current_rate < -0.0001:  # <-0.01%
            findings.append({
                "signal": "funding_overleveraged_short",
                "rate_pct": round(rate_pct, 5),
                "implication": f"Funding rate {rate_pct:.4f}% — shorts paying longs. Short squeeze potential.",
                "action": "Widen long take-profits. Squeeze conditions favor long momentum.",
                "confidence": 0.58,
            })

        if abs(current_rate) > 0.0005:  # Extreme either direction
            findings.append({
                "signal": "funding_extreme_volatility",
                "rate_pct": round(rate_pct, 5),
                "implication": f"Extreme funding rate ({rate_pct:.4f}%) — market highly unstable.",
                "action": "Widen all stops. Reduce position sizes. High risk of liquidation cascades.",
                "confidence": 0.65,
            })

        return {
            "source": "binance_funding_rate",
            "timestamp": datetime.utcnow().isoformat(),
            "current_rate_pct": round(rate_pct, 5),
            "avg_rate_pct_10": round(avg_rate * 100, 5),
            "regime": "overleveraged_long" if current_rate > 0.0003 else "overleveraged_short" if current_rate < -0.0001 else "neutral",
            "findings": findings,
        }
    except Exception as e:
        logger.error("Funding rate research failed: %s", e)
        return None


# ── Research Module 4: Binance Open Interest ─────────────────────────────

def research_open_interest() -> Optional[dict]:
    """Fetch BTC open interest from Binance Futures. Free, no auth.
    
    Rising OI + rising price = new money entering (trend confirmation)
    Rising OI + falling price = shorts piling in
    Falling OI = positions closing (trend exhaustion)
    
    Endpoint: GET https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT
    """
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": "BTCUSDT"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        current_oi = float(data.get("openInterest", 0))

        findings = []

        # Heuristic thresholds (BTC OI in contracts/BTC — Binance reports in BTC terms)
        # A rough normal range is 50k-300k BTC in OI for BTCUSDT perps
        if current_oi < 50000:
            findings.append({
                "signal": "oi_very_low",
                "oi": current_oi,
                "implication": "Open interest very low — positions closing, trend exhaustion likely.",
                "action": "Tighten stops to protect any open gains. Avoid new trend entries.",
                "confidence": 0.55,
            })
        elif current_oi > 200000:
            findings.append({
                "signal": "oi_very_high",
                "oi": current_oi,
                "implication": "Open interest very high — crowded market. Large moves (up or down) are amplified.",
                "action": "Wider stops warranted. Potential for rapid liquidation cascades.",
                "confidence": 0.55,
            })

        return {
            "source": "binance_open_interest",
            "timestamp": datetime.utcnow().isoformat(),
            "open_interest": round(current_oi, 2),
            "symbol": data.get("symbol", "BTCUSDT"),
            "findings": findings,
        }
    except Exception as e:
        logger.error("Open interest research failed: %s", e)
        return None


# ── Research Module 4b: Binance Long/Short Ratio ──────────────────────────

def research_long_short_ratio() -> Optional[dict]:
    """Fetch BTC global long/short account ratio from Binance Futures. Free, no auth.
    
    Contrarian indicator:
    - Extreme longs (>70% long) → bearish (retail over-positioned)
    - Extreme shorts (<40% long) → bullish (potential squeeze)
    
    Also fetches top trader sentiment for confirmation.
    
    Endpoints:
    - GET https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=4h&limit=10
    - GET https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=4h&limit=10
    """
    findings = []
    result = {
        "source": "binance_long_short_ratio",
        "timestamp": datetime.utcnow().isoformat(),
        "global_ratio": None,
        "top_trader_ratio": None,
        "findings": findings,
    }

    # Global ratio
    try:
        resp = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "4h", "limit": 10},
            timeout=10,
        )
        resp.raise_for_status()
        ls_data = resp.json()
        if ls_data:
            current = ls_data[0]
            long_pct = float(current.get("longAccount", 0.5))
            short_pct = float(current.get("shortAccount", 0.5))
            ls_ratio = float(current.get("longShortRatio", 1.0))

            result["global_ratio"] = {
                "long_pct": round(long_pct, 4),
                "short_pct": round(short_pct, 4),
                "ratio": round(ls_ratio, 4),
            }

            if long_pct > 0.70:
                findings.append({
                    "signal": "extreme_retail_long",
                    "long_pct": round(long_pct * 100, 1),
                    "implication": f"{long_pct*100:.1f}% of accounts long — extreme retail positioning. Contrarian bearish.",
                    "action": "Reduce long exposure. Market likely to flush retail longs.",
                    "confidence": 0.60,
                })
            elif long_pct < 0.40:
                findings.append({
                    "signal": "extreme_retail_short",
                    "long_pct": round(long_pct * 100, 1),
                    "implication": f"Only {long_pct*100:.1f}% of accounts long — extreme short positioning. Contrarian bullish.",
                    "action": "Favor long entries. Short squeeze conditions building.",
                    "confidence": 0.58,
                })
    except Exception as e:
        logger.error("Global L/S ratio research failed: %s", e)

    # Top trader ratio
    try:
        resp = requests.get(
            "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
            params={"symbol": "BTCUSDT", "period": "4h", "limit": 10},
            timeout=10,
        )
        resp.raise_for_status()
        top_data = resp.json()
        if top_data:
            top = top_data[0]
            top_long = float(top.get("longAccount", 0.5))
            result["top_trader_ratio"] = {
                "long_pct": round(top_long, 4),
                "short_pct": round(float(top.get("shortAccount", 0.5)), 4),
                "ratio": round(float(top.get("longShortRatio", 1.0)), 4),
            }

            # Top traders leaning hard one way is a stronger signal
            if top_long > 0.65:
                findings.append({
                    "signal": "top_traders_long",
                    "long_pct": round(top_long * 100, 1),
                    "implication": f"Top traders {top_long*100:.1f}% long — smart money bullish.",
                    "action": "Confirm long bias. Widen long take-profits.",
                    "confidence": 0.62,
                })
            elif top_long < 0.38:
                findings.append({
                    "signal": "top_traders_short",
                    "long_pct": round(top_long * 100, 1),
                    "implication": f"Top traders {top_long*100:.1f}% long (i.e., heavily short) — smart money bearish.",
                    "action": "Tighten long stops. Favor short setups.",
                    "confidence": 0.62,
                })
    except Exception as e:
        logger.error("Top trader ratio research failed: %s", e)

    return result


# ── Research Module 5: Cross-Market Correlation ──────────────────────────

def research_cross_market(btc_ohlc: list) -> Optional[dict]:
    """Analyze BTC correlation with traditional markets.
    
    When BTC decouples from SPY/QQQ, it often signals regime changes.
    High correlation = risk-on/risk-off trading
    Low correlation = crypto-specific narrative driving price
    """
    if not btc_ohlc or len(btc_ohlc) < 20:
        return None

    # We only have BTC data, so analyze internal correlation metrics
    closes = [c["close"] for c in btc_ohlc]
    volumes = [c["volume"] for c in btc_ohlc]

    # Price-volume divergence
    # Rising price + falling volume = weak rally
    # Falling price + rising volume = capitulation (potential bottom)
    recent_5 = closes[-5:]
    recent_5_vol = volumes[-5:]
    prev_5 = closes[-10:-5]
    prev_5_vol = volumes[-10:-5]

    price_change = (sum(recent_5) / 5 - sum(prev_5) / 5) / (sum(prev_5) / 5) * 100
    vol_change = (sum(recent_5_vol) / 5 - sum(prev_5_vol) / 5) / (sum(prev_5_vol) / 5) * 100 if sum(prev_5_vol) > 0 else 0

    findings = []

    if price_change > 2 and vol_change < -20:
        findings.append({
            "signal": "weak_rally",
            "price_change": round(price_change, 2),
            "vol_change": round(vol_change, 2),
            "implication": "Rising price on falling volume — weak rally, likely to reverse",
            "action": "Tighten long take-profits, prepare short entries",
            "confidence": 0.55,
        })
    elif price_change < -2 and vol_change > 30:
        findings.append({
            "signal": "capitulation",
            "price_change": round(price_change, 2),
            "vol_change": round(vol_change, 2),
            "implication": "Falling price on surging volume — capitulation pattern, potential bottom",
            "action": "Watch for bullish divergence on RSI. Prime long entry zone.",
            "confidence": 0.6,
        })

    return {
        "source": "cross_market",
        "timestamp": datetime.utcnow().isoformat(),
        "price_change_5d": round(price_change, 2),
        "vol_change_5d": round(vol_change, 2),
        "findings": findings,
    }


# ── Main Research Runner ─────────────────────────────────────────────────

def run_full_research(ohlc: list = None) -> dict:
    """Run all research modules and compile a report."""
    logger.info("Running full research sweep...")

    modules = {
        "fng_patterns": research_fng_patterns(),
        "volatility": research_volatility_transitions(ohlc) if ohlc else None,
        "funding_rates": research_funding_rates() if getattr(config, 'ENABLE_BINANCE_RESEARCH', True) else None,
        "open_interest": research_open_interest() if getattr(config, 'ENABLE_BINANCE_RESEARCH', True) else None,
        "long_short_ratio": research_long_short_ratio() if getattr(config, 'ENABLE_BINANCE_RESEARCH', True) else None,
        "cross_market": research_cross_market(ohlc) if ohlc else None,
    }

    # Collect all findings
    all_findings = []
    for name, result in modules.items():
        if result and result.get("findings"):
            for f in result["findings"]:
                f["module"] = name
                all_findings.append(f)
                _save_finding({
                    "timestamp": datetime.utcnow().isoformat(),
                    "module": name,
                    **f,
                })

    # Sort by confidence
    all_findings.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "modules_run": len([m for m in modules.values() if m is not None]),
        "total_findings": len(all_findings),
        "top_findings": all_findings[:5],
        "details": {k: v for k, v in modules.items() if v is not None},
    }

    # Save report
    os.makedirs(RESEARCH_DIR, exist_ok=True)
    report_path = os.path.join(RESEARCH_DIR, f"research_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Research complete: %d modules, %d findings", report["modules_run"], report["total_findings"])
    for f in all_findings:
        logger.info("  [%.0f%%] %s: %s", f.get("confidence", 0) * 100, f.get("signal", "?"), f.get("implication", ""))

    return report


def format_research_report(report: dict) -> str:
    """Format research as readable text."""
    lines = [
        f"🔬 CryptoBot Research Report",
        f"Time: {report['timestamp'][:19]}Z",
        f"Modules: {report['modules_run']} | Findings: {report['total_findings']}",
        "",
    ]

    if not report["top_findings"]:
        lines.append("No actionable findings this cycle.")
    else:
        lines.append("Top Findings:")
        for i, f in enumerate(report["top_findings"], 1):
            conf = f.get("confidence", 0) * 100
            lines.append(f"  {i}. [{conf:.0f}%] {f.get('signal', '?')}")
            lines.append(f"     → {f.get('implication', '')}")
            lines.append(f"     Action: {f.get('action', '')}")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from utils.kraken_client import KrakenClient
    kraken = KrakenClient()
    ohlc = kraken.get_ohlc(interval=60, count=100)
    report = run_full_research(ohlc)
    print(format_research_report(report))
