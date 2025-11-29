"""
Combined Strategy Module

Provides a framework for combining multiple strategies with
weighted voting and ensemble methods.
"""

from typing import Optional, Any, List, Dict
import pandas as pd
import numpy as np

from kuber.strategies.base import BaseStrategy, Signal, SignalType
from kuber.strategies.moving_average import MACrossoverStrategy
from kuber.strategies.momentum import RSIStrategy, MACDStrategy
from kuber.strategies.volatility import BollingerBandsStrategy


class CombinedStrategy(BaseStrategy):
    """
    Combined Strategy that aggregates signals from multiple sub-strategies.
    
    Uses weighted voting to generate consensus signals.
    """
    
    def __init__(self, strategies: Optional[List[BaseStrategy]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 min_agreement: float = 0.5,
                 **kwargs):
        """
        Initialize Combined strategy.
        
        Args:
            strategies: List of sub-strategies
            weights: Weight for each strategy (by name)
            min_agreement: Minimum weighted agreement for signal
        """
        super().__init__(name="Combined", **kwargs)
        
        # Default strategies if none provided
        if strategies is None:
            strategies = [
                MACrossoverStrategy(fast_period=10, slow_period=30),
                RSIStrategy(period=14),
                MACDStrategy(),
                BollingerBandsStrategy()
            ]
            
        self.strategies = strategies
        self.min_agreement = min_agreement
        
        # Set up weights
        if weights is None:
            # Equal weights by default
            self.weights = {s.name: 1.0 / len(strategies) for s in strategies}
        else:
            self.weights = weights
            
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate consensus signal from sub-strategies."""
        if not self.validate_data(data):
            return None
            
        buy_score = 0.0
        sell_score = 0.0
        contributing_strategies = []
        all_metadata = {}
        
        for strategy in self.strategies:
            signal = strategy.generate_signal(symbol, data, current_price, position)
            weight = self.weights.get(strategy.name, 1.0 / len(self.strategies))
            
            if signal:
                if signal.signal_type in (SignalType.BUY, SignalType.STRONG_BUY):
                    score = signal.strength * weight
                    if signal.signal_type == SignalType.STRONG_BUY:
                        score *= 1.5
                    buy_score += score
                    contributing_strategies.append(f"{strategy.name}:BUY({signal.strength:.2f})")
                    
                elif signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
                    score = signal.strength * weight
                    if signal.signal_type == SignalType.STRONG_SELL:
                        score *= 1.5
                    sell_score += score
                    contributing_strategies.append(f"{strategy.name}:SELL({signal.strength:.2f})")
                    
                all_metadata[strategy.name] = signal.metadata
                
        # Determine consensus
        net_score = buy_score - sell_score
        
        if buy_score >= self.min_agreement and buy_score > sell_score:
            signal_type = SignalType.STRONG_BUY if buy_score > 0.7 else SignalType.BUY
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=min(1.0, buy_score),
                price=current_price,
                reason=f"Consensus: {' | '.join(contributing_strategies)}",
                metadata={
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "net_score": net_score,
                    "strategies": all_metadata
                }
            )
            
        elif sell_score >= self.min_agreement and sell_score > buy_score:
            signal_type = SignalType.STRONG_SELL if sell_score > 0.7 else SignalType.SELL
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=min(1.0, sell_score),
                price=current_price,
                reason=f"Consensus: {' | '.join(contributing_strategies)}",
                metadata={
                    "buy_score": buy_score,
                    "sell_score": sell_score,
                    "net_score": net_score,
                    "strategies": all_metadata
                }
            )
            
        return None
        
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0) -> None:
        """Add a strategy to the ensemble."""
        self.strategies.append(strategy)
        self.weights[strategy.name] = weight
        
        # Re-normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
    def remove_strategy(self, name: str) -> None:
        """Remove a strategy from the ensemble."""
        self.strategies = [s for s in self.strategies if s.name != name]
        if name in self.weights:
            del self.weights[name]
            
        # Re-normalize weights
        if self.weights:
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}


class TrendFollowingCombined(CombinedStrategy):
    """
    Combined strategy optimized for trend following.
    
    Uses MA crossovers, MACD, and momentum indicators.
    """
    
    def __init__(self, **kwargs):
        from kuber.strategies.momentum import MomentumStrategy
        
        strategies = [
            MACrossoverStrategy(fast_period=20, slow_period=50),
            MACDStrategy(),
            MomentumStrategy(period=14)
        ]
        
        weights = {
            "MACrossover": 0.4,
            "MACD": 0.35,
            "Momentum": 0.25
        }
        
        super().__init__(
            strategies=strategies,
            weights=weights,
            min_agreement=0.5,
            **kwargs
        )
        self.name = "TrendFollowing"


class MeanReversionCombined(CombinedStrategy):
    """
    Combined strategy optimized for mean reversion.
    
    Uses RSI, Bollinger Bands, and statistical indicators.
    """
    
    def __init__(self, **kwargs):
        from kuber.strategies.mean_reversion import MeanReversionStrategy
        
        strategies = [
            RSIStrategy(period=14, oversold=30, overbought=70),
            BollingerBandsStrategy(period=20, std_dev=2.0),
            MeanReversionStrategy(lookback=20, z_score_threshold=2.0)
        ]
        
        weights = {
            "RSI": 0.3,
            "BollingerBands": 0.35,
            "MeanReversion": 0.35
        }
        
        super().__init__(
            strategies=strategies,
            weights=weights,
            min_agreement=0.6,  # Higher threshold for mean reversion
            **kwargs
        )
        self.name = "MeanReversionCombined"


class AdaptiveStrategy(BaseStrategy):
    """
    Adaptive Strategy that switches between trend and mean reversion
    based on market regime detection.
    """
    
    def __init__(self, trend_threshold: float = 0.02, **kwargs):
        """
        Initialize Adaptive strategy.
        
        Args:
            trend_threshold: ADX or trend strength threshold
        """
        super().__init__(name="Adaptive", trend_threshold=trend_threshold, **kwargs)
        self.trend_threshold = trend_threshold
        self.trend_strategy = TrendFollowingCombined()
        self.reversion_strategy = MeanReversionCombined()
        
    def detect_regime(self, data: pd.DataFrame) -> str:
        """
        Detect market regime (trending vs. ranging).
        
        Returns:
            'trend' or 'range'
        """
        df = data.copy()
        
        # Calculate ADX for trend strength
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=14).sum() / (atr * 14)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14).sum() / (atr * 14)
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(window=14).mean()
        
        if len(adx) > 0 and not pd.isna(adx.iloc[-1]):
            if adx.iloc[-1] > 25:  # ADX > 25 indicates trend
                return "trend"
                
        # Alternative: check price range
        price_range = (df["close"].max() - df["close"].min()) / df["close"].mean()
        if price_range > self.trend_threshold:
            return "trend"
            
        return "range"
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on detected regime."""
        if not self.validate_data(data, min_rows=50):
            return None
            
        regime = self.detect_regime(data)
        
        if regime == "trend":
            signal = self.trend_strategy.generate_signal(symbol, data, current_price, position)
        else:
            signal = self.reversion_strategy.generate_signal(symbol, data, current_price, position)
            
        if signal:
            signal.metadata["regime"] = regime
            signal.metadata["strategy_used"] = self.trend_strategy.name if regime == "trend" else self.reversion_strategy.name
            
        return signal
