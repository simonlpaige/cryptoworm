"""
CryptoBot Configuration
Paper trading bot — NO real orders ever placed.
"""
import os

# ── Kraken API (used ONLY for price data, never for orders) ──────────────
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_PRIVATE_KEY = os.getenv("KRAKEN_PRIVATE_KEY")

if not KRAKEN_API_KEY:
    raise RuntimeError("Missing required environment variable: KRAKEN_API_KEY")
if not KRAKEN_PRIVATE_KEY:
    raise RuntimeError("Missing required environment variable: KRAKEN_PRIVATE_KEY")

# ── Paper-Trading Account ────────────────────────────────────────────────
INITIAL_BALANCE = 500.0          # USD virtual balance
PAIR = "XXBTZUSD"                # Kraken pair name for BTC/USD
PAIR_DISPLAY = "BTC/USD"

# ── Portfolio Structure (80/20 HODL/Grid) ────────────────────────────────
HODL_ALLOCATION_PCT = 80.0       # 80% ($400) buy-and-hold BTC — the real moneymaker
GRID_POOL_PCT = 20.0             # 20% ($100) active grid trading — extract value from chop
# Backtest showed HODL returned 8,022% over 9 years. Grid's edge is defensive
# (saves money in crashes) but can't compete with HODL in bull runs.
# This split captures both: growth from HODL, income from Grid.

# ── Risk Rules ──────────────────────────────────────────────────────────
MAX_RISK_PER_TRADE_PCT = 20.0    # 20% of GRID POOL per trade ($20 on $100 pool)
MAX_CONCURRENT_POSITIONS = 3     # 3 concurrent positions max
STOP_LOSS_PCT = 2.0              # default stop-loss %
DAILY_MAX_LOSS_PCT = 10.0        # $10 on $100 grid pool
WEEKLY_MAX_LOSS_PCT = 15.0       # $15 on $100 grid pool
DRAWDOWN_PAUSE_PCT = 20.0        # 20% drawdown pause — backtested winner (turned -$114 into +$15)

# ── Grid Bot (Strategy 1) ───────────────────────────────────────────────
GRID_ALLOCATION_PCT = 85.0       # 85% of GRID POOL (not total balance)
GRID_RANGE_PCT = 15.0            # ±15% — wide range won the backtest
GRID_LEVELS = 15                 # 15 grid lines — wide config was best
GRID_RESERVE_PCT = 15.0          # keep 15% grid reserve
MAX_GRID_POSITIONS = 3           # 3 grid positions

# ── Sentiment Swing (Strategy 2) ────────────────────────────────────────
SENTIMENT_API_URL = "https://api.alternative.me/fng/"
FEAR_THRESHOLD = 20              # extreme fear → buy signal (tightened from 25)
GREED_THRESHOLD = 80             # extreme greed → sell signal (tightened from 75)
SWING_TAKE_PROFIT_PCT = 3.0     # 3% target (reduced from 5% - take profits faster)
SWING_STOP_LOSS_PCT = 1.5       # 1.5% stop (tightened from 2%)

# ── Political / Macro Signals (Strategy 6) ──────────────────────────────
POLITICAL_SIGNAL_THRESHOLD = 50   # composite score ±50 triggers trade
POLITICAL_DECAY_HOURS = 4.0       # political signals lose relevance after 4h
POLITICAL_ENABLE_TRUMP = True     # toggle Trump/Truth Social signals
POLITICAL_ENABLE_CONGRESS = True  # toggle congressional trading signals
POLITICAL_ENABLE_FED = True       # toggle Federal Reserve/macro signals
POLITICAL_ENABLE_SEC = True       # toggle SEC EDGAR institutional signals

# ── Strategy Enable/Disable Toggles ─────────────────────────────────────
# Dead-weight strategies (1 trade each in 6-month backtest) — disabled by default
ENABLE_EMA_MACD = False           # EMA/MACD: 1 trade in 6mo backtest, wastes compute
ENABLE_BOLLINGER = False          # Bollinger: 1 trade in 6mo backtest, wastes compute
ENABLE_TARIFF_WHIPLASH = False    # Tariff whiplash: 1 trade in 6mo, needs real tariff events
ENABLE_CONGRESS_FRONTRUN = True   # Congressional front-running: 50% WR, 2.42 PF — keep active
ENABLE_RSI_DIVERGENCE = False     # RSI divergence: disabled - 1 trade in 6mo, fights Grid (opens opposite positions)

# ── ML Signal Generator ─────────────────────────────────────────────────
ENABLE_ML_SIGNAL = True           # Enable XGBoost ML signal filter
ML_RETRAIN_INTERVAL_TICKS = 288   # Retrain every 288 ticks (24h at 5min ticks)
ML_MIN_HISTORY = 100              # Min candles before first prediction
ML_CONFIDENCE_THRESHOLD = 0.6     # Signal threshold (>0.6 = BUY/SELL, else HOLD)

# ── Funding Rate Monitor ───────────────────────────────────────────────
ENABLE_FUNDING_MONITOR = True     # Enable funding rate tracking (informational)

# ── Volatility-Scaled Grid ─────────────────────────────────────────────
GRID_VOL_SCALING = True           # Scale grid spacing with realized volatility

# ── Realistic Backtesting Costs ─────────────────────────────────────────
REALISTIC_SLIPPAGE_PCT = 0.05     # 0.05% slippage per trade
REALISTIC_FEE_PCT = 0.075         # 0.075% fee one-way (0.15% round-trip)

# ── Research — Binance Futures (451 from US IP, disable to stop log spam) ──
ENABLE_BINANCE_RESEARCH = False   # US IP gets 451; re-enable with VPN/proxy

# ── Scheduling ───────────────────────────────────────────────────────────
CHECK_INTERVAL_SECONDS = 300     # 5 minutes

# ── Paths ────────────────────────────────────────────────────────────────
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_PATH = os.path.join(BOT_DIR, "TRADE_LOG.md")
STATE_FILE = os.path.join(BOT_DIR, "bot_state.json")
LOG_FILE = os.path.join(BOT_DIR, "bot.log")
