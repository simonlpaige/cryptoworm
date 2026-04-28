"""
XGBoost ML Signal Generator
============================
Uses XGBoost trained on accumulated OHLC + feature data with walk-forward
validation. Predicts probability of price going up/down in next N candles.

Outputs: BUY (>0.6 prob up), SELL (>0.6 prob down), HOLD (else).
Retrains every 24h (288 ticks at 5min). Minimum 100 candles before first prediction.
"""
import json
import logging
import math
import os
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

logger = logging.getLogger("cryptobot.ml_signal")

# Lazy-load xgboost to avoid import-time crash if not installed
_xgb = None


def _get_xgb():
    global _xgb
    if _xgb is None:
        import xgboost as xgb
        _xgb = xgb
    return _xgb


# Model save path
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainer")
_MODEL_PATH = os.path.join(_MODEL_DIR, "ml_model.json")
_IMPORTANCES_PATH = os.path.join(_MODEL_DIR, "ml_feature_importances.json")


class MLSignalGenerator:
    """XGBoost-based signal generator for paper trading."""

    def __init__(self, retrain_interval: int = 288, min_history: int = 100,
                 confidence_threshold: float = 0.6):
        self._retrain_interval = retrain_interval
        self._min_history = min_history
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._feature_names: List[str] = []
        self._tick_count = 0
        self._last_train_tick = 0
        self._ohlc_history: List[Dict[str, float]] = []
        self._feature_history: List[Dict[str, Optional[float]]] = []
        self._last_signal = "HOLD"
        self._last_confidence = 0.5

        # Try to load saved model
        self._load_model()

    def add_candle(self, candle: Dict[str, float]):
        """Accumulate OHLC candle data for training."""
        self._ohlc_history.append(candle)

    def add_features(self, features: Dict[str, Optional[float]]):
        """Accumulate computed features aligned with candles."""
        self._feature_history.append(features)

    def update(self, ohlc_data: List[Dict[str, float]],
               features: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """Run one tick of the ML signal generator.

        Args:
            ohlc_data: Full OHLC history available (from Kraken).
            features: Current feature dict from compute_features().

        Returns:
            Dict with signal, confidence, and metadata.
        """
        self._tick_count += 1

        # Accumulate latest candle and features
        if ohlc_data:
            # Only add the latest candle if it's new
            latest = ohlc_data[-1]
            if not self._ohlc_history or self._ohlc_history[-1] != latest:
                self._ohlc_history.append(latest)
                self._feature_history.append(features)

        result = {
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": "insufficient data",
            "tick": self._tick_count,
        }

        # Check if we need to retrain
        ticks_since_train = self._tick_count - self._last_train_tick
        if ticks_since_train >= self._retrain_interval and len(self._ohlc_history) >= self._min_history:
            self._train()

        # Make prediction if model exists and we have features
        if self._model is not None and features:
            signal, confidence = self._predict(features)
            result["signal"] = signal
            result["confidence"] = confidence
            result["reason"] = f"XGBoost prediction (p={confidence:.3f})"
            self._last_signal = signal
            self._last_confidence = confidence
        elif len(self._ohlc_history) < self._min_history:
            result["reason"] = f"accumulating history ({len(self._ohlc_history)}/{self._min_history})"

        return result

    def _build_training_data(self) -> Tuple[Optional[Any], Optional[Any], List[str]]:
        """Build feature matrix X and label vector y from accumulated data.

        Label: 1 if price goes up >0% in next 4 candles, else 0.
        Uses temporal ordering (no shuffle).
        """
        if len(self._ohlc_history) < self._min_history + 4:
            return None, None, []

        # Align features with labels
        # For each candle i, label = 1 if close[i+4] > close[i]
        lookforward = 4
        n = len(self._feature_history) - lookforward
        if n < 50:
            return None, None, []

        # Get feature names from the first non-empty feature dict
        feature_names = []
        for f in self._feature_history:
            if f:
                feature_names = sorted(f.keys())
                break
        if not feature_names:
            return None, None, []

        X_rows = []
        y_labels = []
        for i in range(n):
            if i >= len(self._feature_history) or not self._feature_history[i]:
                continue
            feat = self._feature_history[i]
            row = [feat.get(name) for name in feature_names]
            # Convert None to NaN for XGBoost
            row = [float('nan') if v is None else float(v) for v in row]

            # Label: next 4 candles return > 0
            if i + lookforward < len(self._ohlc_history):
                future_close = self._ohlc_history[i + lookforward]["close"]
                current_close = self._ohlc_history[i]["close"]
                if current_close > 0:
                    label = 1 if future_close > current_close else 0
                else:
                    continue
            else:
                continue

            X_rows.append(row)
            y_labels.append(label)

        if len(X_rows) < 50:
            return None, None, []

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_labels, dtype=np.int32)
        return X, y, feature_names

    def _train(self):
        """Train XGBoost model using walk-forward validation."""
        xgb = _get_xgb()
        X, y, feature_names = self._build_training_data()
        if X is None or y is None:
            logger.info("ML train skipped: insufficient training data")
            return

        n = len(y)
        # 70/15/15 temporal split
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        if train_end < 30 or val_end - train_end < 5:
            logger.info("ML train skipped: splits too small (n=%d)", n)
            return

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info("ML training: %d train, %d val, %d test samples, %d features",
                     len(y_train), len(y_val), len(y_test), X.shape[1])

        try:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
                tree_method="hist",
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Evaluate on test set
            if len(y_test) > 0:
                test_preds = model.predict(X_test)
                test_acc = sum(1 for p, a in zip(test_preds, y_test) if p == a) / len(y_test)
                logger.info("ML test accuracy: %.1f%% (%d samples)", test_acc * 100, len(y_test))

            self._model = model
            self._feature_names = feature_names
            self._last_train_tick = self._tick_count

            # Save model and log feature importances
            self._save_model()
            self._log_feature_importances()

            logger.info("ML model retrained successfully at tick %d", self._tick_count)

        except Exception as e:
            logger.error("ML training failed: %s", e)

    def _predict(self, features: Dict[str, Optional[float]]) -> Tuple[str, float]:
        """Predict signal from current features.

        Returns (signal, confidence) where signal is BUY/SELL/HOLD.
        """
        if self._model is None or not self._feature_names:
            return "HOLD", 0.5

        try:
            row = [features.get(name) for name in self._feature_names]
            row = [float('nan') if v is None else float(v) for v in row]
            X = np.array([row], dtype=np.float32)

            # Get probability of class 1 (price goes up)
            proba = self._model.predict_proba(X)[0]
            prob_up = proba[1] if len(proba) > 1 else proba[0]
            prob_down = 1 - prob_up

            if prob_up >= self._confidence_threshold:
                return "BUY", prob_up
            elif prob_down >= self._confidence_threshold:
                return "SELL", prob_down
            else:
                return "HOLD", max(prob_up, prob_down)

        except Exception as e:
            logger.warning("ML prediction failed: %s", e)
            return "HOLD", 0.5

    def _save_model(self):
        """Save model to JSON file."""
        if self._model is None:
            return
        try:
            self._model.save_model(_MODEL_PATH)
            # Also save feature names
            meta = {
                "feature_names": self._feature_names,
                "tick_count": self._tick_count,
                "saved_at": time.time(),
                "history_size": len(self._ohlc_history),
            }
            meta_path = _MODEL_PATH.replace(".json", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            logger.info("ML model saved to %s", _MODEL_PATH)
        except Exception as e:
            logger.warning("Failed to save ML model: %s", e)

    def _load_model(self):
        """Load model from JSON file if available."""
        if not os.path.exists(_MODEL_PATH):
            return
        try:
            xgb = _get_xgb()
            model = xgb.XGBClassifier()
            model.load_model(_MODEL_PATH)
            self._model = model

            # Load feature names
            meta_path = _MODEL_PATH.replace(".json", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self._feature_names = meta.get("feature_names", [])

            logger.info("ML model loaded from %s (%d features)",
                        _MODEL_PATH, len(self._feature_names))
        except Exception as e:
            logger.warning("Failed to load ML model: %s", e)
            self._model = None

    def _log_feature_importances(self):
        """Log and save feature importances after training."""
        if self._model is None or not self._feature_names:
            return
        try:
            importances = self._model.feature_importances_
            feat_imp = sorted(
                zip(self._feature_names, importances),
                key=lambda x: x[1], reverse=True,
            )
            logger.info("ML Feature Importances (top 10):")
            for name, imp in feat_imp[:10]:
                logger.info("  %s: %.4f", name, imp)

            # Save to file
            imp_dict = {name: float(imp) for name, imp in feat_imp}
            with open(_IMPORTANCES_PATH, "w") as f:
                json.dump(imp_dict, f, indent=2)
        except Exception as e:
            logger.warning("Failed to log feature importances: %s", e)

    @property
    def signal(self) -> str:
        return self._last_signal

    @property
    def confidence(self) -> float:
        return self._last_confidence

    @property
    def has_model(self) -> bool:
        return self._model is not None
