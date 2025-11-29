"""
Mean Reversion Trading Strategy

Identifies when price deviates significantly from its mean and
trades the expected reversion.
"""

from typing import Optional, Any
import pandas as pd
import numpy as np

from kuber.strategies.base import BaseStrategy, Signal, SignalType


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.
    
    Assumes prices tend to revert to their mean over time.
    Buys when price is significantly below mean, sells when above.
    """
    
    def __init__(self, lookback: int = 20, z_score_threshold: float = 2.0,
                 exit_z_score: float = 0.5, **kwargs):
        """
        Initialize Mean Reversion strategy.
        
        Args:
            lookback: Period for calculating mean and std
            z_score_threshold: Z-score threshold for entry (e.g., 2.0 = 2 std devs)
            exit_z_score: Z-score threshold for exit
        """
        super().__init__(
            name="MeanReversion",
            lookback=lookback,
            z_score_threshold=z_score_threshold,
            exit_z_score=exit_z_score,
            **kwargs
        )
        self.lookback = lookback
        self.z_score_threshold = z_score_threshold
        self.exit_z_score = exit_z_score
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        df = data.copy()
        
        # Rolling mean and standard deviation
        df["rolling_mean"] = df["close"].rolling(window=self.lookback).mean()
        df["rolling_std"] = df["close"].rolling(window=self.lookback).std()
        
        # Z-score
        df["z_score"] = (df["close"] - df["rolling_mean"]) / df["rolling_std"]
        
        # Distance from mean as percentage
        df["mean_distance_pct"] = ((df["close"] - df["rolling_mean"]) / 
                                   df["rolling_mean"] * 100)
        
        # Trend filter (optional)
        df["trend_sma"] = df["close"].rolling(window=self.lookback * 2).mean()
        df["above_trend"] = df["close"] > df["trend_sma"]
        
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on mean reversion."""
        if not self.validate_data(data, min_rows=self.lookback * 2 + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty:
            return None
            
        z_score = df["z_score"].iloc[-1]
        prev_z = df["z_score"].iloc[-2]
        mean = df["rolling_mean"].iloc[-1]
        distance_pct = df["mean_distance_pct"].iloc[-1]
        
        if pd.isna(z_score) or pd.isna(prev_z):
            return None
            
        # Entry signals
        if z_score <= -self.z_score_threshold:
            # Price significantly below mean - potential buy
            # Check if Z-score is turning (reversal confirmation)
            if z_score > prev_z:
                strength = min(1.0, abs(z_score) / (self.z_score_threshold * 1.5))
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"Mean reversion buy: Z-score {z_score:.2f}, {distance_pct:.1f}% below mean",
                    metadata={
                        "z_score": z_score,
                        "rolling_mean": mean,
                        "distance_pct": distance_pct,
                        "target_price": mean  # Mean as initial target
                    }
                )
                self._record_signal(signal)
                return signal
                
        elif z_score >= self.z_score_threshold:
            # Price significantly above mean - potential sell
            if z_score < prev_z:
                strength = min(1.0, abs(z_score) / (self.z_score_threshold * 1.5))
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"Mean reversion sell: Z-score {z_score:.2f}, {distance_pct:.1f}% above mean",
                    metadata={
                        "z_score": z_score,
                        "rolling_mean": mean,
                        "distance_pct": distance_pct,
                        "target_price": mean
                    }
                )
                self._record_signal(signal)
                return signal
                
        # Exit signals (for existing positions)
        if position:
            # If we have a long position and price has reverted to mean
            if hasattr(position, 'quantity') and position.quantity > 0:
                if prev_z < 0 and z_score >= self.exit_z_score:
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=0.7,
                        price=current_price,
                        reason=f"Mean reversion target reached (Z-score: {z_score:.2f})",
                        metadata={"z_score": z_score, "exit_type": "target"}
                    )
                    self._record_signal(signal)
                    return signal
                    
        return None


class PairsTrading(BaseStrategy):
    """
    Pairs Trading Strategy.
    
    Trades the spread between two correlated assets,
    buying the underperformer and selling the outperformer.
    
    Note: This is a simplified single-asset implementation.
    For true pairs trading, you'd need to track two assets.
    """
    
    def __init__(self, lookback: int = 60, z_threshold: float = 2.0, **kwargs):
        """
        Initialize Pairs Trading strategy.
        
        Args:
            lookback: Period for spread calculation
            z_threshold: Z-score threshold for entry
        """
        super().__init__(
            name="PairsTrading",
            lookback=lookback,
            z_threshold=z_threshold,
            **kwargs
        )
        self.lookback = lookback
        self.z_threshold = z_threshold
        # Ratio/spread history would be stored here
        self.spread_history = []
        
    def calculate_spread(self, price_a: pd.Series, price_b: pd.Series) -> pd.Series:
        """
        Calculate the spread between two price series.
        
        Uses a simple ratio approach.
        """
        return np.log(price_a / price_b)
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """
        Generate signal based on pairs spread.
        
        Note: This simplified version uses price relative to its own history
        as a proxy. For real pairs trading, integrate with another asset.
        """
        if not self.validate_data(data, min_rows=self.lookback + 10):
            return None
            
        # Use log returns for mean reversion
        df = data.copy()
        df["log_price"] = np.log(df["close"])
        df["spread_mean"] = df["log_price"].rolling(window=self.lookback).mean()
        df["spread_std"] = df["log_price"].rolling(window=self.lookback).std()
        df["z_spread"] = (df["log_price"] - df["spread_mean"]) / df["spread_std"]
        
        if df.empty or pd.isna(df["z_spread"].iloc[-1]):
            return None
            
        z_spread = df["z_spread"].iloc[-1]
        prev_z = df["z_spread"].iloc[-2]
        
        # Mean reversion signals
        if z_spread <= -self.z_threshold and z_spread > prev_z:
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=min(1.0, abs(z_spread) / self.z_threshold / 1.5),
                price=current_price,
                reason=f"Spread mean reversion: Z-score {z_spread:.2f}",
                metadata={"z_spread": z_spread}
            )
            self._record_signal(signal)
            return signal
            
        elif z_spread >= self.z_threshold and z_spread < prev_z:
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=min(1.0, abs(z_spread) / self.z_threshold / 1.5),
                price=current_price,
                reason=f"Spread mean reversion: Z-score {z_spread:.2f}",
                metadata={"z_spread": z_spread}
            )
            self._record_signal(signal)
            return signal
            
        return None


class StatisticalArbitrage(MeanReversionStrategy):
    """
    Statistical Arbitrage Strategy.
    
    Enhanced mean reversion with additional statistical filters.
    """
    
    def __init__(self, lookback: int = 20, z_score_threshold: float = 2.0,
                 half_life_max: int = 10, **kwargs):
        """
        Initialize Statistical Arbitrage strategy.
        
        Args:
            lookback: Period for mean calculation
            z_score_threshold: Entry threshold
            half_life_max: Maximum half-life for mean reversion (filters out trending assets)
        """
        super().__init__(
            lookback=lookback,
            z_score_threshold=z_score_threshold,
            **kwargs
        )
        self.name = "StatArb"
        self.half_life_max = half_life_max
        
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using OLS.
        
        Lower half-life indicates faster mean reversion.
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align the series
        spread_lag = spread_lag.iloc[1:]
        spread_diff = spread_diff.iloc[1:]
        
        if len(spread_lag) < 10:
            return float('inf')
            
        # Simple OLS regression
        X = spread_lag.values
        y = spread_diff.values
        
        # Calculate beta
        beta = np.cov(X, y)[0, 1] / np.var(X) if np.var(X) > 0 else 0
        
        if beta >= 0:
            return float('inf')  # Not mean reverting
            
        half_life = -np.log(2) / beta
        return half_life
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal with half-life filter."""
        if not self.validate_data(data, min_rows=self.lookback * 2 + 10):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty:
            return None
            
        # Calculate half-life to confirm mean reversion
        half_life = self.calculate_half_life(df["close"] - df["rolling_mean"])
        
        # Filter out assets that don't mean revert fast enough
        if half_life > self.half_life_max or half_life < 0:
            return None
            
        # Get base signal
        signal = super().generate_signal(symbol, data, current_price, position)
        
        if signal:
            # Adjust strength based on half-life
            half_life_factor = 1 - (half_life / self.half_life_max)
            signal.strength *= half_life_factor
            signal.metadata["half_life"] = half_life
            
        return signal
