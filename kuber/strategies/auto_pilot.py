"""Autonomous ensemble strategy that adapts to market regime and live performance."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import inspect

import numpy as np
import pandas as pd

from kuber.strategies.base import BaseStrategy, Signal, SignalType
from kuber.strategies.combined import TrendFollowingCombined, MeanReversionCombined, AdaptiveStrategy
from kuber.strategies.drl_ppo import PPOStrategy, LSTMStrategy


@dataclass
class StrategyStats:
    score: float = 0.0
    trades: int = 0
    wins: int = 0
    recent_pnls: deque = field(default_factory=lambda: deque(maxlen=50))
    cooldown: int = 0
    last_pnl: float = 0.0


class AutoPilotStrategy(BaseStrategy):
    """Meta strategy that routes capital to the best recent performers automatically."""

    def __init__(
        self,
        strategies: Optional[List[BaseStrategy]] = None,
        evaluation_bars: int = 12,
        decay: float = 0.85,
        activation_threshold: float = 0.15,
        cooldown_bars: int = 15,
        loss_cooldown_threshold: float = 0.008,
    ) -> None:
        super().__init__(name="AutoPilot")

        if strategies is None:
            strategies = [
                TrendFollowingCombined(),
                MeanReversionCombined(),
                AdaptiveStrategy(),
                PPOStrategy(lookback=20, confidence_threshold=0.5, train_on_init=False),
                LSTMStrategy(lookback=10, threshold=0.30),
            ]

        self.strategies = strategies
        self.evaluation_bars = evaluation_bars
        self.decay = decay
        self.activation_threshold = activation_threshold
        self.cooldown_bars = cooldown_bars
        self.loss_cooldown_threshold = loss_cooldown_threshold
        self.min_weight = 0.05

        self.performance_tracker: Dict[str, Dict[str, StrategyStats]] = {}
        self.pending_positions: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.strategy_styles = self._infer_styles()
        self._supports_position_cache: Dict[str, bool] = {}

    def _infer_styles(self) -> Dict[str, str]:
        styles = {}
        for strat in self.strategies:
            if "Trend" in strat.name:
                styles[strat.name] = "trend"
            elif "Reversion" in strat.name:
                styles[strat.name] = "mean"
            elif isinstance(strat, PPOStrategy) or isinstance(strat, LSTMStrategy):
                styles[strat.name] = "ml"
            elif isinstance(strat, AdaptiveStrategy):
                styles[strat.name] = "adaptive"
            else:
                styles[strat.name] = "general"
        return styles

    def _get_stats(self, symbol: str, strategy_name: str) -> StrategyStats:
        symbol_stats = self.performance_tracker.setdefault(symbol, {})
        return symbol_stats.setdefault(strategy_name, StrategyStats())

    def _tick_cooldowns(self, symbol: str) -> None:
        for stats in self.performance_tracker.get(symbol, {}).values():
            if stats.cooldown > 0:
                stats.cooldown -= 1

    def _kelly_fraction(self, stats: StrategyStats) -> float:
        if not stats.recent_pnls:
            return 0.0
        pnl_array = np.array(stats.recent_pnls, dtype=np.float32)
        mean = pnl_array.mean()
        variance = pnl_array.var() + 1e-6
        kelly = float(np.clip(mean / variance, -0.25, 0.5))
        return kelly

    def _strategy_available(self, symbol: str, strategy_name: str) -> bool:
        stats = self._get_stats(symbol, strategy_name)
        return stats.cooldown <= 0

    def _detect_regime(self, data: pd.DataFrame) -> Dict[str, float]:
        df = data.tail(200).copy()
        if len(df) < 50:
            return {"adx": 0, "vol": 0, "bias": 0}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum.reduce([tr1, tr2, tr3])
        atr_series = pd.Series(tr).rolling(window=14).mean()
        atr = atr_series.iloc[-1]
        if np.isnan(atr):
            atr = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else 1.0

        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_series = pd.Series(plus_dm).rolling(window=14).sum()
        minus_series = pd.Series(minus_dm).rolling(window=14).sum()
        plus_di = 100 * plus_series.iloc[-1] / (atr * 14 + 1e-8)
        minus_di = 100 * minus_series.iloc[-1] / (atr * 14 + 1e-8)
        if np.isnan(plus_di):
            plus_di = 0
        if np.isnan(minus_di):
            minus_di = 0

        adx = abs(plus_di - minus_di) / (abs(plus_di) + abs(minus_di) + 1e-8) * 100
        vol = pd.Series(close).pct_change().std() * np.sqrt(252)
        bias = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0

        return {"adx": float(adx), "vol": float(vol), "bias": float(bias)}

    def _update_pending_positions(self, symbol: str, data: pd.DataFrame) -> None:
        pending = self.pending_positions.get(symbol)
        if not pending:
            return

        resolved = []
        for strat_name, trade in pending.items():
            entry_bar = int(trade["entry_bar"])
            if len(data) - 1 < entry_bar + self.evaluation_bars:
                continue
            exit_idx = entry_bar + self.evaluation_bars
            if exit_idx >= len(data):
                continue
            exit_price = data['close'].iloc[exit_idx]
            entry_price = trade["entry_price"]
            if entry_price <= 0:
                resolved.append(strat_name)
                continue
            pnl = trade["direction"] * ((exit_price - entry_price) / entry_price)
            self._record_performance(symbol, strat_name, pnl)
            resolved.append(strat_name)

        for strat_name in resolved:
            pending.pop(strat_name, None)

    def _record_performance(self, symbol: str, strategy_name: str, pnl: float) -> None:
        stats = self._get_stats(symbol, strategy_name)
        stats.score = self.decay * stats.score + (1 - self.decay) * pnl
        stats.trades += 1
        if pnl > 0:
            stats.wins += 1
        stats.last_pnl = pnl
        stats.recent_pnls.append(pnl)
        if pnl < -self.loss_cooldown_threshold:
            stats.cooldown = self.cooldown_bars

    def _compute_weight(self, symbol: str, strategy_name: str, regime: Dict[str, float]) -> float:
        stats = self._get_stats(symbol, strategy_name)
        if stats.cooldown > 0:
            return 0.0

        base = stats.score + 0.1
        win_rate = stats.wins / stats.trades if stats.trades else 0.5
        base += 0.15 * (win_rate - 0.5)
        base += self._kelly_fraction(stats)
        base = max(self.min_weight, base)

        style = self.strategy_styles.get(strategy_name, "general")
        adx = regime.get("adx", 0)
        vol = regime.get("vol", 0)
        bias = regime.get("bias", 0)

        if style == "trend":
            style_mult = 1.35 if adx > 25 or abs(bias) > 0.025 else 0.8
        elif style == "mean":
            style_mult = 1.25 if adx < 18 and vol < 0.4 else 0.75
        elif style == "adaptive":
            style_mult = 1.15 if 0.2 < vol < 0.6 else 0.9
        elif style == "ml":
            style_mult = 1.2 if vol > 0.45 else 0.9
        else:
            style_mult = 1.05 if vol < 0.4 else 1.15

        if stats.recent_pnls and np.mean(np.array(stats.recent_pnls)) < -0.002:
            style_mult *= 0.7

        return max(0.0, base * style_mult)

    def _queue_signal(self, symbol: str, strategy_name: str, direction: int, current_price: float, current_bar: int) -> None:
        if direction == 0:
            return
        pending = self.pending_positions.setdefault(symbol, {})
        # Only track the most recent signal per strategy
        pending[strategy_name] = {
            "direction": direction,
            "entry_price": current_price,
            "entry_bar": current_bar,
        }

    def _strategy_supports_position(self, strategy: BaseStrategy) -> bool:
        name = strategy.name
        if name in self._supports_position_cache:
            return self._supports_position_cache[name]
        sig = inspect.signature(strategy.generate_signal)
        supports = len(sig.parameters) >= 5  # self + symbol + data + current_price + position
        self._supports_position_cache[name] = supports
        return supports

    def _invoke_strategy(self, strategy: BaseStrategy, symbol: str, data: pd.DataFrame, current_price: float, position: Optional[object]) -> Optional[Signal]:
        if self._strategy_supports_position(strategy):
            return strategy.generate_signal(symbol, data, current_price, position)
        return strategy.generate_signal(symbol, data, current_price)

    def generate_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_price: float,
        position: Optional[object] = None,
    ) -> Optional[Signal]:
        if not self.validate_data(data, min_rows=80):
            return None

        self._update_pending_positions(symbol, data)
        self._tick_cooldowns(symbol)
        regime = self._detect_regime(data)

        weighted_sum = 0.0
        total_weight = 0.0
        contributing = []
        suppressed: List[Dict[str, str]] = []

        for strategy in self.strategies:
            if not self._strategy_available(symbol, strategy.name):
                suppressed.append({
                    "strategy": strategy.name,
                    "reason": "cooldown",
                })
                continue
            signal = self._invoke_strategy(strategy, symbol, data, current_price, position)
            if not signal:
                continue
            direction = 0
            if signal.signal_type in (SignalType.BUY, SignalType.STRONG_BUY):
                direction = 1
            elif signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
                direction = -1
            else:
                direction = 0

            if direction == 0:
                continue

            weight = self._compute_weight(symbol, strategy.name, regime)
            weighted_sum += weight * direction * signal.strength
            total_weight += weight
            contributing.append({
                "strategy": strategy.name,
                "direction": direction,
                "strength": signal.strength,
                "weight": weight,
                "cooldown": self._get_stats(symbol, strategy.name).cooldown,
            })

            self._queue_signal(symbol, strategy.name, direction, current_price, len(data) - 1)

        if total_weight == 0:
            return None

        consensus = weighted_sum / total_weight
        if abs(consensus) < self.activation_threshold:
            return None

        signal_type = SignalType.BUY if consensus > 0 else SignalType.SELL
        if abs(consensus) > 0.5:
            signal_type = SignalType.STRONG_BUY if consensus > 0 else SignalType.STRONG_SELL

        metadata = {
            "regime": regime,
            "contributors": contributing,
            "consensus": consensus,
            "suppressed": suppressed,
        }

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            price=current_price,
            strength=min(1.0, abs(consensus)),
            metadata=metadata,
        )
