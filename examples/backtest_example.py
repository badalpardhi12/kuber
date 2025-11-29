#!/usr/bin/env python3
"""
Backtesting Example for Kuber Trading Platform

This example demonstrates how to:
1. Configure a backtest
2. Run historical strategy testing
3. Analyze backtest results
4. Run walk-forward analysis
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from kuber.core.broker import RobinhoodBroker
from kuber.backtest import Backtester, BacktestConfig
from kuber.strategies.momentum import RSIStrategy, MACDStrategy
from kuber.strategies.moving_average import MACrossoverStrategy, GoldenCrossStrategy
from kuber.strategies.volatility import BollingerBandsStrategy
from kuber.strategies.combined import TrendFollowingStrategy

# Load environment variables
load_dotenv()


def print_backtest_results(result):
    """Pretty print backtest results."""
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n📈 Performance Metrics:")
    print("-" * 40)
    print(f"  Total Return:     {result.total_return * 100:>10.2f}%")
    print(f"  Annual Return:    {result.annualized_return * 100:>10.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown * 100:>10.2f}%")
    
    print(f"\n📊 Trade Statistics:")
    print("-" * 40)
    print(f"  Total Trades:     {result.total_trades:>10}")
    print(f"  Winning Trades:   {result.winning_trades:>10}")
    print(f"  Losing Trades:    {result.losing_trades:>10}")
    print(f"  Win Rate:         {result.win_rate * 100:>10.2f}%")
    print(f"  Profit Factor:    {result.profit_factor:>10.2f}")
    print(f"  Avg Win:          {result.avg_win * 100:>10.2f}%")
    print(f"  Avg Loss:         {result.avg_loss * 100:>10.2f}%")
    
    print(f"\n💰 Portfolio Summary:")
    print("-" * 40)
    print(f"  Starting Capital: ${result.starting_capital:>12,.2f}")
    print(f"  Final Value:      ${result.final_value:>12,.2f}")
    print(f"  Total Profit:     ${result.final_value - result.starting_capital:>12,.2f}")


def example_single_strategy_backtest(broker: RobinhoodBroker):
    """Run backtest with a single strategy."""
    print("\n" + "=" * 60)
    print("🧪 Example 1: Single Strategy Backtest (RSI)")
    print("=" * 60)
    
    # Configure the backtest
    config = BacktestConfig(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date=datetime.now() - timedelta(days=365),  # 1 year ago
        end_date=datetime.now(),
        initial_capital=100000.0,
        commission=0.0,  # Robinhood has no commission
        slippage=0.001,  # 0.1% slippage assumption
        position_size=0.1,  # 10% of portfolio per position
    )
    
    # Create strategy
    strategy = RSIStrategy(period=14, oversold=30, overbought=70)
    
    # Run backtest
    backtester = Backtester(broker, config)
    result = backtester.run(strategy)
    
    # Print results
    print_backtest_results(result)
    
    return result


def example_strategy_comparison(broker: RobinhoodBroker):
    """Compare multiple strategies."""
    print("\n" + "=" * 60)
    print("🧪 Example 2: Strategy Comparison")
    print("=" * 60)
    
    # Configure the backtest
    config = BacktestConfig(
        symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        initial_capital=100000.0,
        commission=0.0,
        slippage=0.001,
        position_size=0.1,
    )
    
    # Define strategies to compare
    strategies = {
        "RSI": RSIStrategy(period=14, oversold=30, overbought=70),
        "MACD": MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
        "MA Crossover": MACrossoverStrategy(short_period=20, long_period=50),
        "Golden Cross": GoldenCrossStrategy(),
        "Bollinger Bands": BollingerBandsStrategy(period=20, std_dev=2.0),
    }
    
    # Run backtests and collect results
    results = {}
    backtester = Backtester(broker, config)
    
    for name, strategy in strategies.items():
        print(f"\n  Testing {name}...", end=" ")
        result = backtester.run(strategy)
        results[name] = result
        print(f"Return: {result.total_return * 100:.2f}%")
    
    # Summary comparison
    print("\n" + "-" * 60)
    print("📊 STRATEGY COMPARISON SUMMARY")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>10} {'Win Rate':>10} {'Max DD':>10}")
    print("-" * 60)
    
    # Sort by total return
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1].total_return, 
                           reverse=True)
    
    for name, result in sorted_results:
        print(f"{name:<20} "
              f"{result.total_return * 100:>9.2f}% "
              f"{result.sharpe_ratio:>10.2f} "
              f"{result.win_rate * 100:>9.2f}% "
              f"{result.max_drawdown * 100:>9.2f}%")
    
    # Find best strategy
    best_strategy = sorted_results[0][0]
    print(f"\n🏆 Best performing strategy: {best_strategy}")
    
    return results


def example_walk_forward_analysis(broker: RobinhoodBroker):
    """Run walk-forward analysis for more realistic results."""
    print("\n" + "=" * 60)
    print("🧪 Example 3: Walk-Forward Analysis")
    print("=" * 60)
    
    print("""
Walk-forward analysis divides historical data into multiple
training and testing periods to provide more realistic
performance estimates and detect overfitting.
    """)
    
    # Configure the backtest
    config = BacktestConfig(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date=datetime.now() - timedelta(days=730),  # 2 years
        end_date=datetime.now(),
        initial_capital=100000.0,
        commission=0.0,
        slippage=0.001,
        position_size=0.1,
    )
    
    # Create strategy
    strategy = TrendFollowingStrategy()
    
    # Run walk-forward analysis
    backtester = Backtester(broker, config)
    wf_results = backtester.walk_forward_analysis(
        strategy=strategy,
        n_splits=5,  # 5 train/test periods
        train_pct=0.7,  # 70% training, 30% testing
    )
    
    print("\n📊 Walk-Forward Results by Period:")
    print("-" * 60)
    print(f"{'Period':<10} {'Train Return':>15} {'Test Return':>15} {'Overfit?':>10}")
    print("-" * 60)
    
    for i, result in enumerate(wf_results, 1):
        train_ret = result.get("train_return", 0) * 100
        test_ret = result.get("test_return", 0) * 100
        overfit = "Yes" if train_ret > test_ret * 2 else "No"
        print(f"Period {i:<3} {train_ret:>14.2f}% {test_ret:>14.2f}% {overfit:>10}")
    
    # Calculate average out-of-sample performance
    avg_test_return = sum(r.get("test_return", 0) for r in wf_results) / len(wf_results)
    print("-" * 60)
    print(f"Average out-of-sample return: {avg_test_return * 100:.2f}%")
    
    return wf_results


def example_parameter_optimization(broker: RobinhoodBroker):
    """Demonstrate strategy parameter optimization."""
    print("\n" + "=" * 60)
    print("🧪 Example 4: Parameter Optimization")
    print("=" * 60)
    
    print("""
Testing different RSI parameter combinations to find
the optimal settings for the given stock universe.
    """)
    
    # Configure the backtest
    config = BacktestConfig(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        initial_capital=100000.0,
        commission=0.0,
        slippage=0.001,
        position_size=0.1,
    )
    
    # Parameter grid for RSI
    param_grid = {
        "period": [7, 14, 21],
        "oversold": [20, 25, 30],
        "overbought": [70, 75, 80],
    }
    
    # Run optimization
    backtester = Backtester(broker, config)
    
    # Use RSIStrategy as base
    strategy = RSIStrategy()
    
    best_result = backtester.run_optimization(
        strategy=strategy,
        param_grid=param_grid,
        metric="sharpe",  # Optimize for Sharpe ratio
    )
    
    print("\n🏆 Optimal Parameters Found:")
    print("-" * 40)
    print(f"  Period: {best_result.get('best_params', {}).get('period', 14)}")
    print(f"  Oversold: {best_result.get('best_params', {}).get('oversold', 30)}")
    print(f"  Overbought: {best_result.get('best_params', {}).get('overbought', 70)}")
    print(f"\n  Best Sharpe Ratio: {best_result.get('best_score', 0):.2f}")
    
    return best_result


def main():
    """Main function running all backtest examples."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║            KUBER BACKTESTING EXAMPLES                        ║
║                                                              ║
║  This script demonstrates various backtesting capabilities   ║
║  of the Kuber trading platform.                             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize broker
    print("🔐 Connecting to Robinhood for historical data...")
    
    broker = RobinhoodBroker(
        username=os.getenv("ROBINHOOD_USERNAME"),
        password=os.getenv("ROBINHOOD_PASSWORD"),
        totp_secret=os.getenv("ROBINHOOD_TOTP_SECRET"),
    )
    
    if not broker.login():
        print("❌ Failed to connect. Using mock data for demonstration...")
        # In a real scenario, you might use mock data here
        return
    
    print("✅ Connected successfully!\n")
    
    try:
        # Run examples
        example_single_strategy_backtest(broker)
        example_strategy_comparison(broker)
        example_walk_forward_analysis(broker)
        example_parameter_optimization(broker)
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    finally:
        broker.logout()
        print("\n👋 Logged out from Robinhood")


if __name__ == "__main__":
    main()
