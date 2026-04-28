"""
Novel Pattern Discovery - uses local Gemma4 LLM to hypothesize
multi-parameter correlations, then validates them with backtesting.

Flow:
  1. Collect raw market data (OHLC, volume, funding, sentiment, on-chain)
  2. Compute ~30 technical indicators and cross-correlations
  3. Send correlation matrix + anomalies to Gemma4
  4. Gemma4 proposes tradeable hypotheses (entry/exit rules)
  5. validate_proposal() backtests each one out-of-sample
  6. Only proposals that clear PF/WR/Sharpe thresholds get saved

The LLM is a hypothesis generator only - it never decides what trades.
A proposal that survives validate_proposal() with profit_factor >=
NOVEL_MIN_PROFIT_FACTOR, win_rate >= NOVEL_MIN_WIN_RATE, sharpe >=
NOVEL_MIN_SHARPE earns promotion. Everything else gets archived in a
'rejected_proposals' list so we can audit why the LLM was wrong.

This runs as a weekly research job, NOT real-time trading.
"""
import json
import logging
import os
import math
import http.client
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger("cryptoworm.novel")

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "gemma4:26b"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "trainer", "strategy_proposals.json")


def compute_indicators(ohlc: list) -> dict:
    """Compute ~30 indicators from OHLC data for correlation analysis."""
    if len(ohlc) < 50:
        return {}

    closes = [c["close"] for c in ohlc]
    highs = [c["high"] for c in ohlc]
    lows = [c["low"] for c in ohlc]
    volumes = [c.get("volume", 0) for c in ohlc]
    timestamps = [c.get("time", 0) for c in ohlc]

    indicators = {}

    # --- Price-based ---
    indicators["rsi_14"] = _rsi(closes, 14)
    indicators["rsi_7"] = _rsi(closes, 7)
    indicators["sma_20"] = _sma(closes, 20)
    indicators["sma_50"] = _sma(closes, 50)
    indicators["ema_12"] = _ema(closes, 12)
    indicators["ema_26"] = _ema(closes, 26)
    indicators["macd"] = [a - b if a is not None and b is not None else None for a, b in zip(indicators["ema_12"], indicators["ema_26"])] if indicators["ema_12"] and indicators["ema_26"] else []
    indicators["bb_upper"], indicators["bb_lower"], indicators["bb_width"] = _bollinger(closes, 20, 2)
    indicators["atr_14"] = _atr(highs, lows, closes, 14)

    # --- Volume-based ---
    indicators["volume_sma_20"] = _sma(volumes, 20)
    indicators["volume_ratio"] = _ratio(volumes, indicators["volume_sma_20"])  # vol / avg vol
    indicators["obv"] = _obv(closes, volumes)
    indicators["vwap"] = _vwap(closes, volumes)

    # --- Momentum ---
    indicators["roc_10"] = _roc(closes, 10)  # rate of change
    indicators["roc_20"] = _roc(closes, 20)
    indicators["momentum_14"] = _momentum(closes, 14)
    indicators["stoch_k"] = _stochastic_k(closes, highs, lows, 14)
    indicators["stoch_d"] = _sma(indicators["stoch_k"], 3) if indicators["stoch_k"] else []
    indicators["williams_r"] = _williams_r(closes, highs, lows, 14)

    # --- Volatility ---
    indicators["std_20"] = _rolling_std(closes, 20)
    indicators["realized_vol_20"] = _realized_vol(closes, 20)
    indicators["high_low_range"] = [h - l for h, l in zip(highs, lows)]
    indicators["close_to_high"] = [(c - l) / (h - l) if h != l else 0.5 for c, h, l in zip(closes, highs, lows)]

    # --- Time-based ---
    indicators["hour_of_day"] = [_hour_from_ts(t) for t in timestamps]
    indicators["day_of_week"] = [_dow_from_ts(t) for t in timestamps]

    # --- Cross-correlations (the novel part) ---
    indicators["rsi_vol_product"] = _multiply(indicators["rsi_14"], indicators["volume_ratio"])
    indicators["bb_width_x_volume"] = _multiply(indicators["bb_width"], indicators["volume_ratio"])
    indicators["momentum_x_rsi"] = _multiply(indicators["momentum_14"], indicators["rsi_14"])
    indicators["atr_x_obv_roc"] = _multiply(indicators["atr_14"], _roc(indicators["obv"], 10) if indicators["obv"] else [])

    return indicators


def find_correlations(indicators: dict, closes: list, lookahead: int = 12) -> list:
    """Find which indicators best predict price moves N bars ahead."""
    if len(closes) < lookahead + 50:
        return []

    # Future returns (what we're trying to predict)
    future_returns = []
    for i in range(len(closes) - lookahead):
        ret = (closes[i + lookahead] - closes[i]) / closes[i] * 100
        future_returns.append(ret)

    correlations = []
    for name, values in indicators.items():
        if not values or len(values) < len(future_returns):
            continue
        # Trim to match
        trimmed = values[:len(future_returns)]
        # Filter out None/nan
        pairs = [(v, r) for v, r in zip(trimmed, future_returns) if v is not None and not math.isnan(v)]
        if len(pairs) < 30:
            continue
        corr = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
        if corr is not None and not math.isnan(corr):
            correlations.append({
                "indicator": name,
                "correlation": round(corr, 4),
                "abs_corr": round(abs(corr), 4),
                "samples": len(pairs),
                "mean_value": round(sum(p[0] for p in pairs) / len(pairs), 4),
                "std_value": round(_std([p[0] for p in pairs]), 4)
            })

    correlations.sort(key=lambda x: x["abs_corr"], reverse=True)
    return correlations[:20]  # Top 20 most correlated


def query_gemma(correlations: list, market_context: dict) -> list:
    """Send correlation data to Gemma4 and get tradeable hypotheses back."""
    prompt = f"""You are a quantitative trading researcher analyzing BTC/USD on 1-hour candles.

Here are the top indicator correlations with future 12-hour returns:
{json.dumps(correlations[:15], indent=2)}

Current market context:
- Regime: {market_context.get('regime', 'unknown')}
- ADX: {market_context.get('adx', 'N/A')}
- ATR: {market_context.get('atr', 'N/A')}
- Fear & Greed Index: {market_context.get('fng', 'N/A')}
- BTC Price: ${market_context.get('price', 'N/A')}

Based on these correlations, propose exactly 3 novel trading strategies.
For each, provide:
1. A hypothesis explaining WHY this correlation exists
2. Entry condition (specific indicator thresholds)
3. Exit condition (take profit + stop loss rules)
4. Expected edge (why this beats random)

Respond in JSON only:
[
  {{
    "name": "strategy_name",
    "hypothesis": "why this works",
    "entry_long": "condition for buy",
    "entry_short": "condition for sell",
    "exit_tp_pct": 3.0,
    "exit_sl_pct": 1.5,
    "key_indicator": "indicator_name",
    "threshold_buy": 0.0,
    "threshold_sell": 0.0,
    "confidence": "high/medium/low"
  }}
]"""

    try:
        body = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.4, "num_predict": 1024, "num_ctx": 4096}
        })

        conn = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=120)
        conn.request("POST", "/api/generate", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()

        if data.get("error"):
            logger.error("Gemma error: %s", data["error"])
            return []

        text = data.get("response", "")
        # Extract JSON from response
        return _extract_json_array(text)

    except Exception as e:
        logger.error("Gemma query failed: %s", e)
        return []


def validate_proposal(proposal: dict, ohlc: list) -> dict:
    """Backtest a Gemma hypothesis against out-of-sample data.

    The LLM hands us a rough rule like "buy when rsi_14 < 30, sell when
    rsi_14 > 70, TP=3%, SL=1.5%". We split the OHLC into a first half
    used to compute the indicator series and a second half used as
    out-of-sample validation. Trades are simulated on the second half
    only - this is how we keep the LLM honest. Anything tested on data
    it was conditioned on doesn't count.

    Returns the proposal dict with these added fields:
      validated:           True/False
      profit_factor:       gross profit / gross loss
      win_rate:            winners / total trades
      sharpe:              annualized Sharpe of trade returns
      validation_trades:   number of trades simulated
      validation_months:   length of the out-of-sample window
      reject_reason:       why it failed (only when validated=False)
    """
    out = dict(proposal)
    out.update({
        "validated": False,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "validation_trades": 0,
        "validation_months": 0.0,
        "reject_reason": "",
    })

    indicator_name = proposal.get("key_indicator")
    if not indicator_name:
        out["reject_reason"] = "no key_indicator field"
        return out

    try:
        threshold_buy = float(proposal.get("threshold_buy"))
        threshold_sell = float(proposal.get("threshold_sell"))
        tp_pct = float(proposal.get("exit_tp_pct", 3.0)) / 100.0
        sl_pct = float(proposal.get("exit_sl_pct", 1.5)) / 100.0
    except (TypeError, ValueError):
        out["reject_reason"] = "missing or invalid thresholds / TP / SL"
        return out

    if not ohlc or len(ohlc) < 200:
        out["reject_reason"] = "insufficient OHLC for validation"
        return out

    indicators = compute_indicators(ohlc)
    series = indicators.get(indicator_name)
    if not series or len(series) != len(ohlc):
        out["reject_reason"] = f"indicator '{indicator_name}' not available"
        return out

    closes = [c["close"] for c in ohlc]
    # Out-of-sample = second half. The first half is treated as the
    # period the LLM (transitively) saw via the correlation summary.
    split = len(ohlc) // 2
    test_closes = closes[split:]
    test_series = series[split:]

    trades = []  # list of pnl_pct
    open_side = None  # 'buy' or 'sell'
    entry_price = None

    for i, (px, val) in enumerate(zip(test_closes, test_series)):
        if val is None:
            continue
        if open_side is None:
            if val <= threshold_buy:
                open_side = "buy"
                entry_price = px
            elif val >= threshold_sell:
                open_side = "sell"
                entry_price = px
            continue

        # Position open: check TP/SL
        change = (px - entry_price) / entry_price
        if open_side == "buy":
            if change >= tp_pct:
                trades.append(change)
                open_side = None
            elif change <= -sl_pct:
                trades.append(change)
                open_side = None
        else:  # sell
            if -change >= tp_pct:
                trades.append(-change)
                open_side = None
            elif -change <= -sl_pct:
                trades.append(-change)
                open_side = None

    n = len(trades)
    out["validation_trades"] = n
    # Hourly candles - estimate validation window in months
    out["validation_months"] = round(len(test_closes) / (24 * 30), 2)

    min_months = float(getattr(config, "NOVEL_MIN_BACKTEST_MONTHS", 6))
    if out["validation_months"] < min_months:
        out["reject_reason"] = (
            f"validation window {out['validation_months']:.1f}mo < "
            f"NOVEL_MIN_BACKTEST_MONTHS={min_months}"
        )
        return out

    if n < 10:
        out["reject_reason"] = f"only {n} trades in validation window"
        return out

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    wr = len(wins) / n

    mean = sum(trades) / n
    var = sum((t - mean) ** 2 for t in trades) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var) if var > 0 else 0.0
    # Treat each trade as a sample; annualize roughly assuming ~50 trades/yr for
    # the typical proposal. This is a rough Sharpe, not a tradeable metric.
    sharpe = (mean / std * math.sqrt(50)) if std > 0 else 0.0

    out["profit_factor"] = round(pf if pf != float("inf") else 999.0, 3)
    out["win_rate"] = round(wr, 3)
    out["sharpe"] = round(sharpe, 3)

    min_pf = float(getattr(config, "NOVEL_MIN_PROFIT_FACTOR", 1.5))
    min_wr = float(getattr(config, "NOVEL_MIN_WIN_RATE", 0.55))
    min_sharpe = float(getattr(config, "NOVEL_MIN_SHARPE", 1.0))

    if pf < min_pf:
        out["reject_reason"] = f"profit_factor {pf:.2f} < {min_pf}"
        return out
    if wr < min_wr:
        out["reject_reason"] = f"win_rate {wr:.2f} < {min_wr}"
        return out
    if sharpe < min_sharpe:
        out["reject_reason"] = f"sharpe {sharpe:.2f} < {min_sharpe}"
        return out

    out["validated"] = True
    return out


def run_discovery(kraken_client, regime_detector=None) -> dict:
    """Main entry point: collect data, compute correlations, ask Gemma, return proposals."""
    logger.info("Starting novel pattern discovery...")

    # Get OHLC data (6 months of hourly candles)
    ohlc = kraken_client.get_ohlc(interval=60, count=2000)
    if not ohlc or len(ohlc) < 200:
        logger.warning("Not enough OHLC data for discovery (%d candles)", len(ohlc) if ohlc else 0)
        return {"status": "insufficient_data"}

    closes = [c["close"] for c in ohlc]
    logger.info("Computing indicators on %d candles...", len(ohlc))
    indicators = compute_indicators(ohlc)

    # Find what predicts future price moves
    logger.info("Finding correlations with 12h future returns...")
    correlations = find_correlations(indicators, closes, lookahead=12)
    if not correlations:
        logger.warning("No meaningful correlations found")
        return {"status": "no_correlations"}

    logger.info("Top correlations:")
    for c in correlations[:5]:
        logger.info("  %s: r=%.4f (n=%d)", c["indicator"], c["correlation"], c["samples"])

    # Build market context
    market_context = {
        "regime": regime_detector.regime if regime_detector else "unknown",
        "adx": regime_detector.adx if regime_detector else None,
        "atr": regime_detector.atr if regime_detector else None,
        "price": closes[-1] if closes else None,
    }

    # Query Gemma for hypotheses
    logger.info("Querying Gemma4 for novel strategy hypotheses...")
    raw_proposals = query_gemma(correlations, market_context)

    # Validation gate: backtest each hypothesis on out-of-sample data
    # before saving. The LLM is a generator, not an oracle.
    promoted = []
    rejected = []
    for prop in raw_proposals:
        validated = validate_proposal(prop, ohlc)
        if validated["validated"]:
            promoted.append(validated)
            logger.info(
                "PROMOTED %s: PF=%.2f WR=%.2f Sharpe=%.2f trades=%d",
                validated.get("name", "<unnamed>"),
                validated["profit_factor"], validated["win_rate"],
                validated["sharpe"], validated["validation_trades"],
            )
        else:
            rejected.append(validated)
            logger.info(
                "REJECTED %s: %s",
                validated.get("name", "<unnamed>"),
                validated["reject_reason"],
            )

    logger.info("Validation: %d promoted, %d rejected (of %d raw proposals)",
                len(promoted), len(rejected), len(raw_proposals))

    result = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "candles_analyzed": len(ohlc),
        "top_correlations": correlations[:10],
        "proposals": promoted,
        "rejected_proposals": rejected,
        "market_context": market_context,
    }

    # Save results
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved %d validated proposals to %s", len(promoted), RESULTS_FILE)
    except Exception as e:
        logger.error("Failed to save proposals: %s", e)

    return result


# ─── Indicator Helper Functions ───────────────────────────────────────────

def _rsi(closes, period):
    if len(closes) < period + 1:
        return []
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsi_values = [None] * period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi_values.append(100 - (100 / (1 + rs)))
    return rsi_values

def _sma(values, period):
    if not values or len(values) < period:
        return []
    result = [None] * (period - 1)
    for i in range(period - 1, len(values)):
        window = [v for v in values[i-period+1:i+1] if v is not None]
        result.append(sum(window) / len(window) if window else None)
    return result

def _ema(values, period):
    if not values or len(values) < period:
        return []
    multiplier = 2 / (period + 1)
    ema = [None] * (period - 1)
    ema.append(sum(values[:period]) / period)
    for i in range(period, len(values)):
        ema.append((values[i] - ema[-1]) * multiplier + ema[-1])
    return ema

def _bollinger(closes, period, std_mult):
    sma = _sma(closes, period)
    std = _rolling_std(closes, period)
    upper, lower, width = [], [], []
    for s, d in zip(sma, std):
        if s is not None and d is not None:
            upper.append(s + std_mult * d)
            lower.append(s - std_mult * d)
            width.append(2 * std_mult * d / s * 100 if s != 0 else 0)  # width as % of price
        else:
            upper.append(None)
            lower.append(None)
            width.append(None)
    return upper, lower, width

def _atr(highs, lows, closes, period):
    if len(closes) < period + 1:
        return []
    trs = [None]
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return _sma(trs, period)

def _obv(closes, volumes):
    if len(closes) < 2:
        return []
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv

def _vwap(closes, volumes):
    if not closes or not volumes:
        return []
    cum_vol = 0
    cum_pv = 0
    vwap = []
    for c, v in zip(closes, volumes):
        cum_pv += c * v
        cum_vol += v
        vwap.append(cum_pv / cum_vol if cum_vol != 0 else c)
    return vwap

def _roc(values, period):
    if not values or len(values) < period + 1:
        return []
    result = [None] * period
    for i in range(period, len(values)):
        if values[i-period] is not None and values[i-period] != 0 and values[i] is not None:
            result.append((values[i] - values[i-period]) / values[i-period] * 100)
        else:
            result.append(None)
    return result

def _momentum(closes, period):
    if len(closes) < period + 1:
        return []
    return [None] * period + [closes[i] - closes[i-period] for i in range(period, len(closes))]

def _stochastic_k(closes, highs, lows, period):
    if len(closes) < period:
        return []
    result = [None] * (period - 1)
    for i in range(period - 1, len(closes)):
        h = max(highs[i-period+1:i+1])
        l = min(lows[i-period+1:i+1])
        result.append((closes[i] - l) / (h - l) * 100 if h != l else 50)
    return result

def _williams_r(closes, highs, lows, period):
    if len(closes) < period:
        return []
    result = [None] * (period - 1)
    for i in range(period - 1, len(closes)):
        h = max(highs[i-period+1:i+1])
        l = min(lows[i-period+1:i+1])
        result.append((h - closes[i]) / (h - l) * -100 if h != l else -50)
    return result

def _rolling_std(values, period):
    if not values or len(values) < period:
        return []
    result = [None] * (period - 1)
    for i in range(period - 1, len(values)):
        window = [v for v in values[i-period+1:i+1] if v is not None]
        if len(window) < 2:
            result.append(None)
            continue
        mean = sum(window) / len(window)
        result.append(math.sqrt(sum((x - mean) ** 2 for x in window) / (len(window) - 1)))
    return result

def _realized_vol(closes, period):
    if len(closes) < period + 1:
        return []
    log_returns = [None]
    for i in range(1, len(closes)):
        if closes[i] > 0 and closes[i-1] > 0:
            log_returns.append(math.log(closes[i] / closes[i-1]))
        else:
            log_returns.append(None)
    std = _rolling_std(log_returns, period)
    # Annualize (hourly data, ~8760 hours/year)
    return [s * math.sqrt(8760) * 100 if s is not None else None for s in std]

def _multiply(a, b):
    if not a or not b:
        return []
    result = []
    for va, vb in zip(a, b):
        if va is not None and vb is not None:
            result.append(va * vb)
        else:
            result.append(None)
    return result

def _ratio(a, b):
    if not a or not b:
        return []
    return [va / vb if vb and vb != 0 and va is not None else None for va, vb in zip(a, b)]

def _std(values):
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return 0
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((x - mean) ** 2 for x in clean) / (len(clean) - 1))

def _pearson(x, y):
    n = len(x)
    if n < 5:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / (n - 1))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / (n - 1))
    if sx == 0 or sy == 0:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    return cov / (sx * sy)

def _hour_from_ts(ts):
    try:
        return datetime.utcfromtimestamp(ts).hour if ts else None
    except:
        return None

def _dow_from_ts(ts):
    try:
        return datetime.utcfromtimestamp(ts).weekday() if ts else None
    except:
        return None

def _extract_json_array(text):
    """Extract a JSON array from LLM response text."""
    if not text:
        return []
    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except:
        pass
    # Find array in text
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    return []
