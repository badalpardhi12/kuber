"""
Moving Average Strategies

Various strategies based on moving average crossovers and trends.
"""

from typing import Optional, Any
import pandas as pd
import numpy as np

from kuber.strategies.base import BaseStrategy, Signal, SignalType


class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average Strategy.
    
    Generates buy signals when price crosses above SMA,
    and sell signals when price crosses below.
    """
    
    def __init__(self, period: int = 20, **kwargs):
        """
        Initialize SMA strategy.
        
        Args:
            period: SMA lookback period
        """
        super().__init__(name="SMA", period=period, **kwargs)
        self.period = period
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA indicator."""
        df = data.copy()
        df["sma"] = df["close"].rolling(window=self.period).mean()
        df["above_sma"] = df["close"] > df["sma"]
        df["crossover"] = df["above_sma"].astype(int).diff()
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on SMA crossover."""
        if not self.validate_data(data, min_rows=self.period + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty or pd.isna(df["crossover"].iloc[-1]):
            return None
            
        crossover = df["crossover"].iloc[-1]
        sma_value = df["sma"].iloc[-1]
        price_distance = (current_price - sma_value) / sma_value * 100
        
        if crossover > 0:  # Price crossed above SMA
            strength = min(1.0, abs(price_distance) / 5)
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                price=current_price,
                reason=f"Price crossed above SMA({self.period})",
                metadata={"sma": sma_value, "distance_pct": price_distance}
            )
            self._record_signal(signal)
            return signal
            
        elif crossover < 0:  # Price crossed below SMA
            strength = min(1.0, abs(price_distance) / 5)
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                price=current_price,
                reason=f"Price crossed below SMA({self.period})",
                metadata={"sma": sma_value, "distance_pct": price_distance}
            )
            self._record_signal(signal)
            return signal
            
        return None


class EMAStrategy(BaseStrategy):
    """
    Exponential Moving Average Strategy.
    
    Similar to SMA but uses EMA for faster response to price changes.
    """
    
    def __init__(self, period: int = 20, **kwargs):
        """
        Initialize EMA strategy.
        
        Args:
            period: EMA lookback period
        """
        super().__init__(name="EMA", period=period, **kwargs)
        self.period = period
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA indicator."""
        df = data.copy()
        df["ema"] = df["close"].ewm(span=self.period, adjust=False).mean()
        df["above_ema"] = df["close"] > df["ema"]
        df["crossover"] = df["above_ema"].astype(int).diff()
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on EMA crossover."""
        if not self.validate_data(data, min_rows=self.period + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty or pd.isna(df["crossover"].iloc[-1]):
            return None
            
        crossover = df["crossover"].iloc[-1]
        ema_value = df["ema"].iloc[-1]
        
        if crossover > 0:
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.6,
                price=current_price,
                reason=f"Price crossed above EMA({self.period})",
                metadata={"ema": ema_value}
            )
            self._record_signal(signal)
            return signal
            
        elif crossover < 0:
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=0.6,
                price=current_price,
                reason=f"Price crossed below EMA({self.period})",
                metadata={"ema": ema_value}
            )
            self._record_signal(signal)
            return signal
            
        return None


class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    Uses two moving averages - generates buy signal when fast MA
    crosses above slow MA, and sell when fast crosses below slow.
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 use_ema: bool = True, **kwargs):
        """
        Initialize MA Crossover strategy.
        
        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
            use_ema: Use EMA instead of SMA
        """
        super().__init__(
            name="MACrossover",
            fast_period=fast_period,
            slow_period=slow_period,
            use_ema=use_ema,
            **kwargs
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate fast and slow moving averages."""
        df = data.copy()
        
        if self.use_ema:
            df["fast_ma"] = df["close"].ewm(span=self.fast_period, adjust=False).mean()
            df["slow_ma"] = df["close"].ewm(span=self.slow_period, adjust=False).mean()
        else:
            df["fast_ma"] = df["close"].rolling(window=self.fast_period).mean()
            df["slow_ma"] = df["close"].rolling(window=self.slow_period).mean()
            
        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]
        df["ma_crossover"] = np.sign(df["ma_diff"]).diff()
        
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on MA crossover."""
        if not self.validate_data(data, min_rows=self.slow_period + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty or pd.isna(df["ma_crossover"].iloc[-1]):
            return None
            
        crossover = df["ma_crossover"].iloc[-1]
        ma_diff = df["ma_diff"].iloc[-1]
        fast_ma = df["fast_ma"].iloc[-1]
        slow_ma = df["slow_ma"].iloc[-1]
        
        # Calculate strength based on MA separation
        ma_spread = abs(ma_diff) / slow_ma * 100
        strength = min(1.0, ma_spread / 3)
        
        if crossover > 0:  # Bullish crossover
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                price=current_price,
                reason=f"Fast MA({self.fast_period}) crossed above Slow MA({self.slow_period})",
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "spread_pct": ma_spread
                }
            )
            self._record_signal(signal)
            return signal
            
        elif crossover < 0:  # Bearish crossover
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                price=current_price,
                reason=f"Fast MA({self.fast_period}) crossed below Slow MA({self.slow_period})",
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "spread_pct": ma_spread
                }
            )
            self._record_signal(signal)
            return signal
            
        return None


class GoldenCrossStrategy(MACrossoverStrategy):
    """
    Golden Cross / Death Cross Strategy.
    
    Classic strategy using 50-day and 200-day moving averages.
    Golden Cross (bullish): 50 MA crosses above 200 MA
    Death Cross (bearish): 50 MA crosses below 200 MA
    """
    
    def __init__(self, use_ema: bool = False, **kwargs):
        """
        Initialize Golden Cross strategy.
        
        Args:
            use_ema: Use EMA instead of SMA (classic uses SMA)
        """
        super().__init__(
            fast_period=50,
            slow_period=200,
            use_ema=use_ema,
            **kwargs
        )
        self.name = "GoldenCross"
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate Golden Cross / Death Cross signal."""
        signal = super().generate_signal(symbol, data, current_price, position)
        
        if signal:
            if signal.signal_type == SignalType.BUY:
                signal.reason = "Golden Cross: 50 MA crossed above 200 MA"
                signal.signal_type = SignalType.STRONG_BUY
                signal.strength = min(1.0, signal.strength * 1.5)
            elif signal.signal_type == SignalType.SELL:
                signal.reason = "Death Cross: 50 MA crossed below 200 MA"
                signal.signal_type = SignalType.STRONG_SELL
                signal.strength = min(1.0, signal.strength * 1.5)
                
        return signal


class TripleMAStrategy(BaseStrategy):
    """
    Triple Moving Average Strategy.
    
    Uses three moving averages to confirm trend direction and
    filter out false signals.
    """
    
    def __init__(self, fast_period: int = 5, medium_period: int = 20,
                 slow_period: int = 50, **kwargs):
        """
        Initialize Triple MA strategy.
        
        Args:
            fast_period: Fast MA period
            medium_period: Medium MA period  
            slow_period: Slow MA period
        """
        super().__init__(
            name="TripleMA",
            fast_period=fast_period,
            medium_period=medium_period,
            slow_period=slow_period,
            **kwargs
        )
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate three moving averages."""
        df = data.copy()
        df["fast_ma"] = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        df["medium_ma"] = df["close"].ewm(span=self.medium_period, adjust=False).mean()
        df["slow_ma"] = df["close"].ewm(span=self.slow_period, adjust=False).mean()
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal when all MAs align."""
        if not self.validate_data(data, min_rows=self.slow_period + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        fast_ma = df["fast_ma"].iloc[-1]
        medium_ma = df["medium_ma"].iloc[-1]
        slow_ma = df["slow_ma"].iloc[-1]
        
        prev_fast = df["fast_ma"].iloc[-2]
        prev_medium = df["medium_ma"].iloc[-2]
        
        # Check if MAs are properly aligned
        bullish_alignment = fast_ma > medium_ma > slow_ma
        bearish_alignment = fast_ma < medium_ma < slow_ma
        
        # Check for transition
        was_bullish = prev_fast > prev_medium
        is_bullish = fast_ma > medium_ma
        
        if bullish_alignment and not was_bullish and is_bullish:
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=0.8,
                price=current_price,
                reason="Triple MA bullish alignment",
                metadata={
                    "fast_ma": fast_ma,
                    "medium_ma": medium_ma,
                    "slow_ma": slow_ma
                }
            )
            self._record_signal(signal)
            return signal
            
        elif bearish_alignment and was_bullish and not is_bullish:
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=0.8,
                price=current_price,
                reason="Triple MA bearish alignment",
                metadata={
                    "fast_ma": fast_ma,
                    "medium_ma": medium_ma,
                    "slow_ma": slow_ma
                }
            )
            self._record_signal(signal)
            return signal
            
        return None
