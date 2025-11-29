"""
Momentum-Based Trading Strategies

Strategies based on momentum indicators like RSI, MACD, and Stochastic.
"""

from typing import Optional, Any
import pandas as pd
import numpy as np

from kuber.strategies.base import BaseStrategy, Signal, SignalType


class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) Strategy.
    
    Generates buy signals when RSI is oversold and sell signals when overbought.
    """
    
    def __init__(self, period: int = 14, oversold: float = 30.0,
                 overbought: float = 70.0, **kwargs):
        """
        Initialize RSI strategy.
        
        Args:
            period: RSI lookback period
            oversold: Oversold threshold (buy zone)
            overbought: Overbought threshold (sell zone)
        """
        super().__init__(
            name="RSI",
            period=period,
            oversold=oversold,
            overbought=overbought,
            **kwargs
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI from price series."""
        delta = prices.diff()
        
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        avg_gain = gains.rolling(window=self.period, min_periods=1).mean()
        avg_loss = losses.rolling(window=self.period, min_periods=1).mean()
        
        # Use EMA for smoother RSI
        avg_gain = gains.ewm(alpha=1/self.period, min_periods=self.period).mean()
        avg_loss = losses.ewm(alpha=1/self.period, min_periods=self.period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator."""
        df = data.copy()
        df["rsi"] = self.calculate_rsi(df["close"])
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on RSI levels."""
        if not self.validate_data(data, min_rows=self.period + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty or pd.isna(df["rsi"].iloc[-1]):
            return None
            
        rsi = df["rsi"].iloc[-1]
        prev_rsi = df["rsi"].iloc[-2]
        
        # Calculate signal strength based on extremity
        if rsi <= self.oversold:
            # Oversold - potential buy
            strength = (self.oversold - rsi) / self.oversold
            strength = min(1.0, strength * 2)
            
            # Check if RSI is turning up from oversold
            if rsi > prev_rsi:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY if strength < 0.8 else SignalType.STRONG_BUY,
                    strength=strength,
                    price=current_price,
                    reason=f"RSI oversold at {rsi:.1f}, turning up",
                    metadata={"rsi": rsi, "prev_rsi": prev_rsi}
                )
                self._record_signal(signal)
                return signal
                
        elif rsi >= self.overbought:
            # Overbought - potential sell
            strength = (rsi - self.overbought) / (100 - self.overbought)
            strength = min(1.0, strength * 2)
            
            # Check if RSI is turning down from overbought
            if rsi < prev_rsi:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL if strength < 0.8 else SignalType.STRONG_SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"RSI overbought at {rsi:.1f}, turning down",
                    metadata={"rsi": rsi, "prev_rsi": prev_rsi}
                )
                self._record_signal(signal)
                return signal
                
        return None


class MACDStrategy(BaseStrategy):
    """
    Moving Average Convergence Divergence (MACD) Strategy.
    
    Generates signals based on MACD line crossing the signal line
    and histogram changes.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, **kwargs):
        """
        Initialize MACD strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        super().__init__(
            name="MACD",
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            **kwargs
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators."""
        df = data.copy()
        
        # Calculate EMAs
        fast_ema = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=self.slow_period, adjust=False).mean()
        
        # MACD line
        df["macd"] = fast_ema - slow_ema
        
        # Signal line
        df["macd_signal"] = df["macd"].ewm(span=self.signal_period, adjust=False).mean()
        
        # Histogram
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Crossover detection
        df["macd_crossover"] = np.sign(df["macd"] - df["macd_signal"]).diff()
        
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on MACD crossover."""
        if not self.validate_data(data, min_rows=self.slow_period + self.signal_period + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty or pd.isna(df["macd_crossover"].iloc[-1]):
            return None
            
        crossover = df["macd_crossover"].iloc[-1]
        macd = df["macd"].iloc[-1]
        signal_line = df["macd_signal"].iloc[-1]
        histogram = df["macd_hist"].iloc[-1]
        
        # Calculate strength based on histogram magnitude
        hist_magnitude = abs(histogram)
        avg_hist = df["macd_hist"].abs().rolling(window=20).mean().iloc[-1]
        strength = min(1.0, hist_magnitude / (avg_hist * 2)) if avg_hist > 0 else 0.5
        
        if crossover > 0:  # Bullish crossover
            # Check if crossover happened below zero (stronger signal)
            signal_type = SignalType.STRONG_BUY if macd < 0 else SignalType.BUY
            
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                price=current_price,
                reason=f"MACD bullish crossover (MACD: {macd:.4f}, Signal: {signal_line:.4f})",
                metadata={
                    "macd": macd,
                    "signal": signal_line,
                    "histogram": histogram
                }
            )
            self._record_signal(signal)
            return signal
            
        elif crossover < 0:  # Bearish crossover
            # Check if crossover happened above zero (stronger signal)
            signal_type = SignalType.STRONG_SELL if macd > 0 else SignalType.SELL
            
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                price=current_price,
                reason=f"MACD bearish crossover (MACD: {macd:.4f}, Signal: {signal_line:.4f})",
                metadata={
                    "macd": macd,
                    "signal": signal_line,
                    "histogram": histogram
                }
            )
            self._record_signal(signal)
            return signal
            
        return None


class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator Strategy.
    
    Uses %K and %D lines to identify overbought/oversold conditions.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3,
                 oversold: float = 20.0, overbought: float = 80.0, **kwargs):
        """
        Initialize Stochastic strategy.
        
        Args:
            k_period: %K lookback period
            d_period: %D smoothing period
            oversold: Oversold threshold
            overbought: Overbought threshold
        """
        super().__init__(
            name="Stochastic",
            k_period=k_period,
            d_period=d_period,
            oversold=oversold,
            overbought=overbought,
            **kwargs
        )
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic oscillator."""
        df = data.copy()
        
        # Calculate %K
        low_min = df["low"].rolling(window=self.k_period).min()
        high_max = df["high"].rolling(window=self.k_period).max()
        
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        
        # Calculate %D (signal line)
        df["stoch_d"] = df["stoch_k"].rolling(window=self.d_period).mean()
        
        # Crossover
        df["stoch_crossover"] = np.sign(df["stoch_k"] - df["stoch_d"]).diff()
        
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on Stochastic crossover in oversold/overbought zones."""
        if not self.validate_data(data, min_rows=self.k_period + self.d_period + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty:
            return None
            
        stoch_k = df["stoch_k"].iloc[-1]
        stoch_d = df["stoch_d"].iloc[-1]
        crossover = df["stoch_crossover"].iloc[-1]
        
        if pd.isna(stoch_k) or pd.isna(stoch_d):
            return None
            
        # Buy signal: %K crosses above %D in oversold zone
        if crossover > 0 and stoch_d < self.oversold:
            strength = (self.oversold - stoch_d) / self.oversold
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=min(1.0, strength),
                price=current_price,
                reason=f"Stochastic bullish crossover in oversold zone (K: {stoch_k:.1f}, D: {stoch_d:.1f})",
                metadata={"stoch_k": stoch_k, "stoch_d": stoch_d}
            )
            self._record_signal(signal)
            return signal
            
        # Sell signal: %K crosses below %D in overbought zone
        elif crossover < 0 and stoch_d > self.overbought:
            strength = (stoch_d - self.overbought) / (100 - self.overbought)
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=min(1.0, strength),
                price=current_price,
                reason=f"Stochastic bearish crossover in overbought zone (K: {stoch_k:.1f}, D: {stoch_d:.1f})",
                metadata={"stoch_k": stoch_k, "stoch_d": stoch_d}
            )
            self._record_signal(signal)
            return signal
            
        return None


class MomentumStrategy(BaseStrategy):
    """
    Simple Momentum Strategy.
    
    Measures rate of change in price to identify momentum shifts.
    """
    
    def __init__(self, period: int = 10, threshold: float = 5.0, **kwargs):
        """
        Initialize Momentum strategy.
        
        Args:
            period: Lookback period for momentum calculation
            threshold: Percentage threshold for signal generation
        """
        super().__init__(
            name="Momentum",
            period=period,
            threshold=threshold,
            **kwargs
        )
        self.period = period
        self.threshold = threshold
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicator."""
        df = data.copy()
        
        # Rate of Change (ROC)
        df["roc"] = ((df["close"] - df["close"].shift(self.period)) / 
                     df["close"].shift(self.period) * 100)
        
        # Momentum slope
        df["roc_slope"] = df["roc"].diff()
        
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on momentum."""
        if not self.validate_data(data, min_rows=self.period + 10):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty or pd.isna(df["roc"].iloc[-1]):
            return None
            
        roc = df["roc"].iloc[-1]
        prev_roc = df["roc"].iloc[-2]
        roc_slope = df["roc_slope"].iloc[-1]
        
        # Strong positive momentum with upward slope
        if roc > self.threshold and roc_slope > 0 and prev_roc < self.threshold:
            strength = min(1.0, roc / (self.threshold * 3))
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                price=current_price,
                reason=f"Strong positive momentum: {roc:.1f}% over {self.period} periods",
                metadata={"roc": roc, "roc_slope": roc_slope}
            )
            self._record_signal(signal)
            return signal
            
        # Strong negative momentum with downward slope
        elif roc < -self.threshold and roc_slope < 0 and prev_roc > -self.threshold:
            strength = min(1.0, abs(roc) / (self.threshold * 3))
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                price=current_price,
                reason=f"Strong negative momentum: {roc:.1f}% over {self.period} periods",
                metadata={"roc": roc, "roc_slope": roc_slope}
            )
            self._record_signal(signal)
            return signal
            
        return None
