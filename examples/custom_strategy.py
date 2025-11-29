#!/usr/bin/env python3
"""
Custom Strategy Example for Kuber Trading Platform

This example demonstrates how to:
1. Create a custom trading strategy
2. Implement signal generation logic
3. Combine technical indicators
4. Backtest the custom strategy
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from kuber.core.broker import RobinhoodBroker
from kuber.strategies.base import BaseStrategy, Signal, SignalType
from kuber.backtest import Backtester, BacktestConfig

# Load environment variables
load_dotenv()


class DualMomentumStrategy(BaseStrategy):
    """
    Dual Momentum Strategy
    
    Combines absolute momentum (time-series) with relative momentum
    (cross-sectional) to generate more robust signals.
    
    Entry conditions:
    - Stock is above its 200-day SMA (absolute momentum)
    - Stock's 12-month return is positive
    - Stock is in top 30% of universe by momentum (relative)
    
    Exit conditions:
    - Stock falls below 200-day SMA
    - Stock's momentum turns negative
    """
    
    def __init__(
        self,
        lookback_period: int = 252,  # 1 year of trading days
        sma_period: int = 200,
        top_pct: float = 0.3,
    ):
        super().__init__(name="Dual Momentum")
        self.lookback_period = lookback_period
        self.sma_period = sma_period
        self.top_pct = top_pct
        
        self.description = (
            f"Dual Momentum: {lookback_period}-day lookback, "
            f"{sma_period}-day SMA filter"
        )
    
    def calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum as the percentage return over lookback period."""
        if len(data) < self.lookback_period:
            return 0.0
        
        current_price = data["close"].iloc[-1]
        past_price = data["close"].iloc[-self.lookback_period]
        
        return (current_price - past_price) / past_price
    
    def calculate_sma(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return data["close"].mean()
        return data["close"].tail(period).mean()
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on dual momentum."""
        if data is None or len(data) < self.sma_period:
            return Signal(SignalType.HOLD, 0.0, {})
        
        # Calculate indicators
        current_price = data["close"].iloc[-1]
        sma_200 = self.calculate_sma(data, self.sma_period)
        momentum = self.calculate_momentum(data)
        
        # Calculate trend strength
        price_vs_sma = (current_price - sma_200) / sma_200
        
        # Generate signal
        metadata = {
            "current_price": current_price,
            "sma_200": sma_200,
            "price_vs_sma_pct": price_vs_sma * 100,
            "momentum_12m": momentum * 100,
        }
        
        # Strong Buy: Above SMA with positive momentum
        if current_price > sma_200 and momentum > 0.10:  # >10% momentum
            strength = min(1.0, (momentum + price_vs_sma) / 2)
            return Signal(SignalType.BUY, strength, metadata)
        
        # Moderate Buy: Above SMA with moderate momentum
        elif current_price > sma_200 and momentum > 0:
            strength = min(0.7, momentum + 0.3)
            return Signal(SignalType.BUY, strength, metadata)
        
        # Strong Sell: Below SMA with negative momentum
        elif current_price < sma_200 and momentum < -0.10:
            strength = min(1.0, abs(momentum) + abs(price_vs_sma) / 2)
            return Signal(SignalType.SELL, strength, metadata)
        
        # Moderate Sell: Below SMA
        elif current_price < sma_200 * 0.98:  # 2% below SMA
            strength = min(0.7, abs(price_vs_sma))
            return Signal(SignalType.SELL, strength, metadata)
        
        # Hold: Mixed signals
        return Signal(SignalType.HOLD, 0.0, metadata)


class MeanReversionRSIStrategy(BaseStrategy):
    """
    Mean Reversion with RSI Confirmation
    
    Looks for price deviations from the mean, confirmed by
    RSI extremes, to identify reversion opportunities.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        std_threshold: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        super().__init__(name="Mean Reversion RSI")
        self.lookback = lookback
        self.std_threshold = std_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def calculate_rsi(self, data: pd.DataFrame) -> float:
        """Calculate RSI indicator."""
        if len(data) < self.rsi_period + 1:
            return 50.0
        
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_z_score(self, data: pd.DataFrame) -> float:
        """Calculate Z-score of current price vs rolling mean."""
        if len(data) < self.lookback:
            return 0.0
        
        rolling_mean = data["close"].tail(self.lookback).mean()
        rolling_std = data["close"].tail(self.lookback).std()
        current_price = data["close"].iloc[-1]
        
        if rolling_std == 0:
            return 0.0
        
        return (current_price - rolling_mean) / rolling_std
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate mean reversion signal with RSI confirmation."""
        if data is None or len(data) < max(self.lookback, self.rsi_period + 1):
            return Signal(SignalType.HOLD, 0.0, {})
        
        z_score = self.calculate_z_score(data)
        rsi = self.calculate_rsi(data)
        
        metadata = {
            "z_score": z_score,
            "rsi": rsi,
            "std_threshold": self.std_threshold,
        }
        
        # Oversold: Price significantly below mean + RSI confirms
        if z_score < -self.std_threshold and rsi < self.rsi_oversold:
            strength = min(1.0, abs(z_score) / 3 + (self.rsi_oversold - rsi) / 100)
            return Signal(SignalType.BUY, strength, metadata)
        
        # Overbought: Price significantly above mean + RSI confirms
        if z_score > self.std_threshold and rsi > self.rsi_overbought:
            strength = min(1.0, z_score / 3 + (rsi - self.rsi_overbought) / 100)
            return Signal(SignalType.SELL, strength, metadata)
        
        return Signal(SignalType.HOLD, 0.0, metadata)


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy
    
    Identifies breakouts based on:
    - Price breaking above/below Bollinger Bands
    - Volume confirmation
    - ATR-based volatility expansion
    """
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        volume_multiplier: float = 1.5,
    ):
        super().__init__(name="Volatility Breakout")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.volume_multiplier = volume_multiplier
    
    def calculate_bollinger_bands(self, data: pd.DataFrame):
        """Calculate Bollinger Bands."""
        if len(data) < self.bb_period:
            return None, None, None
        
        sma = data["close"].rolling(window=self.bb_period).mean()
        std = data["close"].rolling(window=self.bb_period).std()
        
        upper = sma + (std * self.bb_std)
        lower = sma - (std * self.bb_std)
        
        return sma.iloc[-1], upper.iloc[-1], lower.iloc[-1]
    
    def calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        if len(data) < self.atr_period + 1:
            return 0.0
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().iloc[-1]
        
        return atr
    
    def is_volume_confirmed(self, data: pd.DataFrame) -> bool:
        """Check if current volume exceeds average."""
        if "volume" not in data.columns or len(data) < 20:
            return True  # Assume confirmed if no volume data
        
        avg_volume = data["volume"].tail(20).mean()
        current_volume = data["volume"].iloc[-1]
        
        return current_volume > avg_volume * self.volume_multiplier
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """Generate breakout signal."""
        if data is None or len(data) < max(self.bb_period, self.atr_period + 1):
            return Signal(SignalType.HOLD, 0.0, {})
        
        sma, upper_band, lower_band = self.calculate_bollinger_bands(data)
        if sma is None:
            return Signal(SignalType.HOLD, 0.0, {})
        
        current_price = data["close"].iloc[-1]
        atr = self.calculate_atr(data)
        volume_confirmed = self.is_volume_confirmed(data)
        
        metadata = {
            "current_price": current_price,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "sma": sma,
            "atr": atr,
            "volume_confirmed": volume_confirmed,
        }
        
        # Bullish breakout
        if current_price > upper_band:
            strength = 0.8 if volume_confirmed else 0.5
            # Increase strength based on how far above band
            breakout_pct = (current_price - upper_band) / upper_band
            strength = min(1.0, strength + breakout_pct)
            return Signal(SignalType.BUY, strength, metadata)
        
        # Bearish breakdown
        if current_price < lower_band:
            strength = 0.8 if volume_confirmed else 0.5
            breakdown_pct = (lower_band - current_price) / lower_band
            strength = min(1.0, strength + breakdown_pct)
            return Signal(SignalType.SELL, strength, metadata)
        
        return Signal(SignalType.HOLD, 0.0, metadata)


def demo_custom_strategy(broker: RobinhoodBroker):
    """Demonstrate custom strategy usage."""
    print("\n" + "=" * 60)
    print("📊 Testing Custom Strategies")
    print("=" * 60)
    
    # Create custom strategies
    strategies = [
        DualMomentumStrategy(),
        MeanReversionRSIStrategy(),
        VolatilityBreakoutStrategy(),
    ]
    
    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
    
    for strategy in strategies:
        print(f"\n🔍 {strategy.name}")
        print("-" * 40)
        
        for symbol in symbols:
            # Get historical data
            data = broker.get_historicals(
                symbol,
                interval="day",
                span="year",
            )
            
            if data is None or data.empty:
                continue
            
            # Generate signal
            signal = strategy.generate_signal(data)
            
            # Format signal output
            signal_emoji = {
                SignalType.BUY: "🟢",
                SignalType.SELL: "🔴",
                SignalType.HOLD: "🟡",
            }
            
            emoji = signal_emoji.get(signal.signal_type, "⚪")
            print(f"  {emoji} {symbol}: {signal.signal_type.value} "
                  f"(strength: {signal.strength:.2f})")


def backtest_custom_strategy(broker: RobinhoodBroker):
    """Backtest a custom strategy."""
    print("\n" + "=" * 60)
    print("📈 Backtesting Custom Strategy")
    print("=" * 60)
    
    # Configure backtest
    config = BacktestConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        initial_capital=100000.0,
        commission=0.0,
        slippage=0.001,
        position_size=0.1,
    )
    
    # Create and run backtest
    strategy = DualMomentumStrategy()
    backtester = Backtester(broker, config)
    result = backtester.run(strategy)
    
    # Print results
    print(f"\n📊 {strategy.name} Backtest Results:")
    print("-" * 40)
    print(f"  Total Return:  {result.total_return * 100:>8.2f}%")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:>8.2f}")
    print(f"  Max Drawdown:  {result.max_drawdown * 100:>8.2f}%")
    print(f"  Total Trades:  {result.total_trades:>8}")
    print(f"  Win Rate:      {result.win_rate * 100:>8.2f}%")
    
    return result


def main():
    """Main function demonstrating custom strategies."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           KUBER CUSTOM STRATEGY EXAMPLES                     ║
║                                                              ║
║  Learn how to create and test your own trading strategies   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Connect to broker
    print("🔐 Connecting to Robinhood...")
    
    broker = RobinhoodBroker(
        username=os.getenv("ROBINHOOD_USERNAME"),
        password=os.getenv("ROBINHOOD_PASSWORD"),
        totp_secret=os.getenv("ROBINHOOD_TOTP_SECRET"),
    )
    
    if not broker.login():
        print("❌ Failed to connect to Robinhood")
        return
    
    print("✅ Connected!\n")
    
    try:
        # Demo custom strategies
        demo_custom_strategy(broker)
        
        # Backtest custom strategy
        backtest_custom_strategy(broker)
        
        print("\n" + "=" * 60)
        print("✅ Custom strategy examples completed!")
        print("=" * 60)
        
    finally:
        broker.logout()
        print("\n👋 Logged out")


if __name__ == "__main__":
    main()
