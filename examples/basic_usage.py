#!/usr/bin/env python3
"""
Basic Usage Example for Kuber Trading Platform

This example demonstrates how to:
1. Connect to Robinhood
2. Set up a simple trading strategy
3. Run the trading engine
"""

import os
from dotenv import load_dotenv

from kuber.core.broker import RobinhoodBroker
from kuber.core.engine import TradingEngine
from kuber.core.portfolio import Portfolio
from kuber.strategies.momentum import RSIStrategy
from kuber.strategies.moving_average import GoldenCrossStrategy
from kuber.risk import RiskManager, RiskParameters

# Load environment variables
load_dotenv()


def main():
    """Main function demonstrating basic Kuber usage."""
    
    # =========================================================================
    # Step 1: Initialize the broker connection
    # =========================================================================
    print("🔐 Connecting to Robinhood...")
    
    broker = RobinhoodBroker(
        username=os.getenv("ROBINHOOD_USERNAME"),
        password=os.getenv("ROBINHOOD_PASSWORD"),
        totp_secret=os.getenv("ROBINHOOD_TOTP_SECRET"),  # Optional for 2FA
    )
    
    if not broker.login():
        print("❌ Failed to connect to Robinhood")
        return
    
    print("✅ Successfully connected to Robinhood!")
    
    # =========================================================================
    # Step 2: Display account information
    # =========================================================================
    print("\n📊 Account Information:")
    print("-" * 40)
    
    profile = broker.get_account_profile()
    print(f"  Account Type: {profile.get('account_type', 'N/A')}")
    
    buying_power = broker.get_buying_power()
    print(f"  Buying Power: ${buying_power:,.2f}")
    
    portfolio_value = broker.get_portfolio_value()
    print(f"  Portfolio Value: ${portfolio_value:,.2f}")
    
    # =========================================================================
    # Step 3: Display current holdings
    # =========================================================================
    print("\n📈 Current Holdings:")
    print("-" * 40)
    
    holdings = broker.get_holdings()
    if holdings:
        for symbol, data in holdings.items():
            quantity = float(data.get("quantity", 0))
            avg_cost = float(data.get("average_buy_price", 0))
            equity = float(data.get("equity", 0))
            print(f"  {symbol}: {quantity:.2f} shares @ ${avg_cost:.2f} = ${equity:.2f}")
    else:
        print("  No holdings found")
    
    # =========================================================================
    # Step 4: Configure risk management
    # =========================================================================
    print("\n⚙️ Configuring Risk Management...")
    
    risk_params = RiskParameters(
        max_position_pct=0.10,       # Max 10% in single position
        max_portfolio_risk=0.20,     # Max 20% portfolio at risk
        stop_loss_pct=0.05,          # 5% stop loss
        take_profit_pct=0.15,        # 15% take profit
        max_daily_loss_pct=0.03,     # Stop if 3% daily loss
        risk_per_trade=0.01,         # Risk 1% per trade
    )
    
    risk_manager = RiskManager(risk_params)
    print("✅ Risk management configured")
    
    # =========================================================================
    # Step 5: Create and configure trading strategies
    # =========================================================================
    print("\n📊 Setting up Trading Strategies...")
    
    # RSI Strategy - good for overbought/oversold detection
    rsi_strategy = RSIStrategy(
        period=14,
        oversold=30,
        overbought=70,
    )
    
    # Golden Cross Strategy - identifies trend reversals
    golden_cross = GoldenCrossStrategy()
    
    print(f"  Strategy 1: {rsi_strategy.name}")
    print(f"  Strategy 2: {golden_cross.name}")
    
    # =========================================================================
    # Step 6: Initialize the trading engine
    # =========================================================================
    print("\n🚀 Initializing Trading Engine...")
    
    # Define watchlist (stocks to monitor)
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    
    engine = TradingEngine(
        broker=broker,
        risk_manager=risk_manager,
        strategies=[rsi_strategy, golden_cross],
        watchlist=watchlist,
        min_signals=1,  # Execute if at least 1 strategy agrees
    )
    
    print("✅ Trading engine initialized")
    print(f"  Watchlist: {', '.join(watchlist)}")
    
    # =========================================================================
    # Step 7: Analyze current market conditions
    # =========================================================================
    print("\n🔍 Analyzing Market Conditions...")
    print("-" * 40)
    
    for symbol in watchlist[:3]:  # Analyze first 3 for demo
        # Get historical data
        data = broker.get_historicals(symbol, interval="day", span="3month")
        
        if data is None or data.empty:
            print(f"  {symbol}: No data available")
            continue
        
        # Get current price
        current_price = broker.get_quote(symbol)
        if current_price is None:
            continue
        
        # Generate signals from each strategy
        rsi_signal = rsi_strategy.generate_signal(data)
        gc_signal = golden_cross.generate_signal(data)
        
        print(f"\n  {symbol} @ ${current_price:.2f}")
        print(f"    RSI Signal: {rsi_signal.signal_type.value} "
              f"(strength: {rsi_signal.strength:.2f})")
        print(f"    Golden Cross Signal: {gc_signal.signal_type.value} "
              f"(strength: {gc_signal.strength:.2f})")
    
    # =========================================================================
    # Step 8: Paper trading simulation (no real trades)
    # =========================================================================
    print("\n" + "=" * 50)
    print("📝 PAPER TRADING MODE")
    print("=" * 50)
    print("""
This example runs in paper trading mode by default.
No real trades will be executed.

To run the trading engine in live mode:
    engine.start(interval_minutes=15)

To run a single analysis cycle:
    engine.run_cycle()

Always start with paper trading to test your strategies!
    """)
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    broker.logout()
    print("✅ Logged out successfully")


if __name__ == "__main__":
    main()
