"""
Volatility-Based Trading Strategies

Strategies based on volatility indicators like Bollinger Bands and ATR.
"""

from typing import Optional, Any
import pandas as pd
import numpy as np

from kuber.strategies.base import BaseStrategy, Signal, SignalType


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.
    
    Uses price position relative to Bollinger Bands to identify
    overbought/oversold conditions and potential reversals.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, **kwargs):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            period: Moving average period
            std_dev: Number of standard deviations for bands
        """
        super().__init__(
            name="BollingerBands",
            period=period,
            std_dev=std_dev,
            **kwargs
        )
        self.period = period
        self.std_dev = std_dev
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df = data.copy()
        
        # Middle band (SMA)
        df["bb_middle"] = df["close"].rolling(window=self.period).mean()
        
        # Standard deviation
        df["bb_std"] = df["close"].rolling(window=self.period).std()
        
        # Upper and lower bands
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * self.std_dev)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * self.std_dev)
        
        # Band width (volatility indicator)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
        
        # %B - where price is relative to bands
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on Bollinger Bands."""
        if not self.validate_data(data, min_rows=self.period + 5):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty:
            return None
            
        upper = df["bb_upper"].iloc[-1]
        lower = df["bb_lower"].iloc[-1]
        middle = df["bb_middle"].iloc[-1]
        bb_pct = df["bb_pct"].iloc[-1]
        prev_bb_pct = df["bb_pct"].iloc[-2]
        
        if pd.isna(bb_pct) or pd.isna(prev_bb_pct):
            return None
            
        # Price bouncing off lower band - potential buy
        if bb_pct < 0.1 and prev_bb_pct < bb_pct:
            strength = min(1.0, (0.2 - bb_pct) / 0.2) if bb_pct < 0.2 else 0.5
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=strength,
                price=current_price,
                reason=f"Price at lower Bollinger Band, bouncing up (%B: {bb_pct:.2f})",
                metadata={
                    "bb_upper": upper,
                    "bb_middle": middle,
                    "bb_lower": lower,
                    "bb_pct": bb_pct
                }
            )
            self._record_signal(signal)
            return signal
            
        # Price touching upper band and turning down - potential sell
        elif bb_pct > 0.9 and prev_bb_pct > bb_pct:
            strength = min(1.0, (bb_pct - 0.8) / 0.2) if bb_pct > 0.8 else 0.5
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=strength,
                price=current_price,
                reason=f"Price at upper Bollinger Band, turning down (%B: {bb_pct:.2f})",
                metadata={
                    "bb_upper": upper,
                    "bb_middle": middle,
                    "bb_lower": lower,
                    "bb_pct": bb_pct
                }
            )
            self._record_signal(signal)
            return signal
            
        # Band squeeze breakout (optional: high volatility expansion)
        bb_width = df["bb_width"].iloc[-1]
        prev_width = df["bb_width"].iloc[-2]
        avg_width = df["bb_width"].rolling(window=20).mean().iloc[-1]
        
        # Breakout from squeeze
        if prev_width < avg_width * 0.7 and bb_width > avg_width:
            if current_price > upper:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.7,
                    price=current_price,
                    reason="Bollinger Band squeeze breakout (bullish)",
                    metadata={"bb_width": bb_width, "avg_width": avg_width}
                )
                self._record_signal(signal)
                return signal
            elif current_price < lower:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=0.7,
                    price=current_price,
                    reason="Bollinger Band squeeze breakout (bearish)",
                    metadata={"bb_width": bb_width, "avg_width": avg_width}
                )
                self._record_signal(signal)
                return signal
                
        return None


class ATRStrategy(BaseStrategy):
    """
    Average True Range (ATR) Based Strategy.
    
    Uses ATR for volatility-based position sizing and stop-loss levels.
    Also identifies volatility expansion/contraction for trading signals.
    """
    
    def __init__(self, period: int = 14, atr_multiplier: float = 2.0, **kwargs):
        """
        Initialize ATR strategy.
        
        Args:
            period: ATR lookback period
            atr_multiplier: Multiplier for ATR-based stops
        """
        super().__init__(
            name="ATR",
            period=period,
            atr_multiplier=atr_multiplier,
            **kwargs
        )
        self.period = period
        self.atr_multiplier = atr_multiplier
        
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        
        return atr
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR and related indicators."""
        df = data.copy()
        
        df["atr"] = self.calculate_atr(df)
        df["atr_pct"] = df["atr"] / df["close"] * 100  # ATR as percentage of price
        
        # Volatility expansion/contraction
        df["atr_sma"] = df["atr"].rolling(window=20).mean()
        df["volatility_ratio"] = df["atr"] / df["atr_sma"]
        
        # Keltner Channel (ATR-based bands)
        ema = df["close"].ewm(span=20, adjust=False).mean()
        df["kc_upper"] = ema + (df["atr"] * self.atr_multiplier)
        df["kc_lower"] = ema - (df["atr"] * self.atr_multiplier)
        df["kc_middle"] = ema
        
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on ATR analysis."""
        if not self.validate_data(data, min_rows=self.period + 20):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty:
            return None
            
        atr = df["atr"].iloc[-1]
        volatility_ratio = df["volatility_ratio"].iloc[-1]
        kc_upper = df["kc_upper"].iloc[-1]
        kc_lower = df["kc_lower"].iloc[-1]
        
        if pd.isna(atr) or pd.isna(volatility_ratio):
            return None
            
        # Volatility expansion with breakout
        if volatility_ratio > 1.5:  # High volatility
            if current_price > kc_upper:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.6,
                    price=current_price,
                    reason=f"Volatility expansion with bullish breakout (ATR ratio: {volatility_ratio:.2f})",
                    metadata={
                        "atr": atr,
                        "volatility_ratio": volatility_ratio,
                        "suggested_stop": current_price - (atr * self.atr_multiplier)
                    }
                )
                self._record_signal(signal)
                return signal
                
            elif current_price < kc_lower:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=0.6,
                    price=current_price,
                    reason=f"Volatility expansion with bearish breakout (ATR ratio: {volatility_ratio:.2f})",
                    metadata={
                        "atr": atr,
                        "volatility_ratio": volatility_ratio,
                        "suggested_stop": current_price + (atr * self.atr_multiplier)
                    }
                )
                self._record_signal(signal)
                return signal
                
        return None
        
    def get_stop_loss_price(self, entry_price: float, atr: float, 
                           is_long: bool = True) -> float:
        """
        Calculate ATR-based stop loss price.
        
        Args:
            entry_price: Trade entry price
            atr: Current ATR value
            is_long: True for long position, False for short
            
        Returns:
            Stop loss price
        """
        stop_distance = atr * self.atr_multiplier
        
        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
            
    def get_position_size(self, account_value: float, risk_pct: float,
                         entry_price: float, stop_price: float) -> float:
        """
        Calculate position size based on risk.
        
        Args:
            account_value: Total account value
            risk_pct: Percentage of account to risk (e.g., 1.0 for 1%)
            entry_price: Planned entry price
            stop_price: Stop loss price
            
        Returns:
            Number of shares to buy
        """
        risk_amount = account_value * (risk_pct / 100)
        risk_per_share = abs(entry_price - stop_price)
        
        if risk_per_share <= 0:
            return 0
            
        shares = risk_amount / risk_per_share
        return shares


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy.
    
    Identifies low volatility periods (consolidation) and trades
    the subsequent breakout.
    """
    
    def __init__(self, lookback: int = 20, breakout_threshold: float = 1.5, **kwargs):
        """
        Initialize Volatility Breakout strategy.
        
        Args:
            lookback: Lookback period for volatility calculation
            breakout_threshold: Multiplier for breakout identification
        """
        super().__init__(
            name="VolatilityBreakout",
            lookback=lookback,
            breakout_threshold=breakout_threshold,
            **kwargs
        )
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        df = data.copy()
        
        # Historical volatility
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=self.lookback).std() * np.sqrt(252) * 100
        
        # Average volatility
        df["avg_volatility"] = df["volatility"].rolling(window=self.lookback * 2).mean()
        
        # Volatility ratio
        df["vol_ratio"] = df["volatility"] / df["avg_volatility"]
        
        # Recent range
        df["range_high"] = df["high"].rolling(window=self.lookback).max()
        df["range_low"] = df["low"].rolling(window=self.lookback).min()
        
        return df
        
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                       current_price: float,
                       position: Optional[Any] = None) -> Optional[Signal]:
        """Generate signal based on volatility breakout."""
        if not self.validate_data(data, min_rows=self.lookback * 3):
            return None
            
        df = self.calculate_indicators(data)
        
        if df.empty:
            return None
            
        vol_ratio = df["vol_ratio"].iloc[-1]
        prev_vol_ratio = df["vol_ratio"].iloc[-2]
        range_high = df["range_high"].iloc[-2]  # Previous period's range
        range_low = df["range_low"].iloc[-2]
        
        if pd.isna(vol_ratio) or pd.isna(prev_vol_ratio):
            return None
            
        # Low volatility followed by expansion (potential breakout)
        if prev_vol_ratio < 0.7 and vol_ratio > 1.0:
            # Bullish breakout
            if current_price > range_high:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=min(1.0, vol_ratio / self.breakout_threshold),
                    price=current_price,
                    reason=f"Volatility breakout above range high ${range_high:.2f}",
                    metadata={
                        "vol_ratio": vol_ratio,
                        "range_high": range_high,
                        "range_low": range_low
                    }
                )
                self._record_signal(signal)
                return signal
                
            # Bearish breakout
            elif current_price < range_low:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=min(1.0, vol_ratio / self.breakout_threshold),
                    price=current_price,
                    reason=f"Volatility breakdown below range low ${range_low:.2f}",
                    metadata={
                        "vol_ratio": vol_ratio,
                        "range_high": range_high,
                        "range_low": range_low
                    }
                )
                self._record_signal(signal)
                return signal
                
        return None
