"""
Backtesting Engine Module

Provides historical data backtesting for trading strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
import pandas as pd
import numpy as np
from loguru import logger

from kuber.strategies.base import BaseStrategy, Signal, SignalType
from kuber.core.portfolio import Portfolio, Position, Trade
from kuber.risk import RiskManager, RiskParameters, PortfolioRiskMetrics


@dataclass
class BacktestConfig:
    """Configuration for backtest."""
    initial_capital: float = 10000.0
    commission_per_trade: float = 0.0     # Robinhood is commission-free
    slippage_pct: float = 0.05            # 0.05% slippage
    max_position_pct: float = 10.0
    stop_loss_pct: float = 5.0
    take_profit_pct: float = 15.0
    enable_shorting: bool = False
    enable_margin: bool = False
    data_start: Optional[str] = None
    data_end: Optional[str] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: float = 0.0
    
    # Data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Dict] = field(default_factory=list)
    signals: List[Dict] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate summary string."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                      BACKTEST RESULTS                             ║
╠══════════════════════════════════════════════════════════════════╣
║ PERFORMANCE METRICS                                               ║
║   Total Return:        {self.total_return:>10.2f}%                          ║
║   Annual Return:       {self.annual_return:>10.2f}%                          ║
║   Sharpe Ratio:        {self.sharpe_ratio:>10.2f}                            ║
║   Sortino Ratio:       {self.sortino_ratio:>10.2f}                            ║
║   Max Drawdown:        {self.max_drawdown:>10.2f}%                          ║
║   Profit Factor:       {self.profit_factor:>10.2f}                            ║
╠══════════════════════════════════════════════════════════════════╣
║ TRADE STATISTICS                                                  ║
║   Total Trades:        {self.total_trades:>10d}                              ║
║   Win Rate:            {self.win_rate:>10.2f}%                          ║
║   Winning Trades:      {self.winning_trades:>10d}                              ║
║   Losing Trades:       {self.losing_trades:>10d}                              ║
║   Avg Win:             ${self.avg_win:>10.2f}                          ║
║   Avg Loss:            ${self.avg_loss:>10.2f}                          ║
║   Largest Win:         ${self.largest_win:>10.2f}                          ║
║   Largest Loss:        ${self.largest_loss:>10.2f}                          ║
║   Avg Holding Period:  {self.avg_holding_period:>10.1f} days                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    Simulates historical trading based on strategy signals
    and calculates performance metrics.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.portfolio = Portfolio(initial_cash=self.config.initial_capital)
        self.risk_manager = RiskManager()
        
        # State tracking
        self._equity_history: List[Dict] = []
        self._trades: List[Dict] = []
        self._signals: List[Dict] = []
        self._open_positions: Dict[str, Dict] = {}
        
    def run(self, strategy: BaseStrategy, 
            data: Dict[str, pd.DataFrame],
            benchmark: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Dict mapping symbol to OHLCV DataFrame
            benchmark: Optional benchmark data (e.g., SPY) for comparison
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Reset state
        self._reset()
        
        # Get aligned dates across all symbols
        all_dates = self._get_aligned_dates(data)
        
        if len(all_dates) < 20:
            logger.error("Insufficient data for backtest")
            return BacktestResult()
            
        logger.info(f"Backtesting from {all_dates[0]} to {all_dates[-1]}")
        logger.info(f"Trading {len(data)} symbols over {len(all_dates)} days")
        
        # Run simulation
        for i, date in enumerate(all_dates):
            self._process_day(date, i, strategy, data)
            
        # Calculate final results
        result = self._calculate_results()
        
        logger.success(f"Backtest complete: {result.total_return:.2f}% return, "
                      f"{result.total_trades} trades")
        
        return result
        
    def _reset(self) -> None:
        """Reset backtester state."""
        self.portfolio = Portfolio(initial_cash=self.config.initial_capital)
        self._equity_history = []
        self._trades = []
        self._signals = []
        self._open_positions = {}
        
    def _get_aligned_dates(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Get dates that exist in all data series."""
        all_dates = None
        
        for symbol, df in data.items():
            if all_dates is None:
                all_dates = df.index
            else:
                all_dates = all_dates.intersection(df.index)
                
        return all_dates.sort_values()
        
    def _process_day(self, date: datetime, day_index: int,
                     strategy: BaseStrategy, data: Dict[str, pd.DataFrame]) -> None:
        """Process a single trading day."""
        
        # Update prices for existing positions
        current_prices = {}
        for symbol, df in data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, "close"]
                
        self.portfolio.update_prices(current_prices)
        
        # Check stop-losses and take-profits for open positions
        self._check_exits(date, current_prices)
        
        # Generate signals for each symbol
        for symbol, df in data.items():
            if date not in df.index:
                continue
                
            # Get data up to current date for strategy
            historical_data = df.loc[:date].copy()
            
            if len(historical_data) < 30:  # Need minimum history
                continue
                
            current_price = current_prices.get(symbol, 0)
            position = self.portfolio.get_position(symbol)
            
            # Get signal from strategy
            signal = strategy.generate_signal(
                symbol=symbol,
                data=historical_data,
                current_price=current_price,
                position=position
            )
            
            if signal and signal.is_actionable():
                self._process_signal(date, signal, current_price)
                
        # Record equity
        self._equity_history.append({
            "date": date,
            "equity": self.portfolio.total_equity,
            "cash": self.portfolio.cash,
            "positions_value": self.portfolio.positions_value
        })
        
    def _check_exits(self, date: datetime, prices: Dict[str, float]) -> None:
        """Check stop-losses and take-profits."""
        for symbol, pos_info in list(self._open_positions.items()):
            if symbol not in prices:
                continue
                
            current_price = prices[symbol]
            entry_price = pos_info["entry_price"]
            stop_loss = pos_info.get("stop_loss", 0)
            take_profit = pos_info.get("take_profit", float("inf"))
            
            # Check stop loss
            if current_price <= stop_loss:
                self._execute_exit(date, symbol, current_price, "stop_loss")
                
            # Check take profit
            elif current_price >= take_profit:
                self._execute_exit(date, symbol, current_price, "take_profit")
                
            # Update trailing stop
            elif "trailing_stop" in pos_info:
                high_price = pos_info.get("highest_price", entry_price)
                if current_price > high_price:
                    pos_info["highest_price"] = current_price
                    # Update trailing stop
                    new_trailing = current_price * (1 - self.config.stop_loss_pct / 100)
                    pos_info["trailing_stop"] = max(pos_info["trailing_stop"], new_trailing)
                    
                if current_price <= pos_info["trailing_stop"]:
                    self._execute_exit(date, symbol, current_price, "trailing_stop")
                    
    def _process_signal(self, date: datetime, signal: Signal, price: float) -> None:
        """Process a trading signal."""
        symbol = signal.symbol
        
        # Record signal
        self._signals.append({
            "date": date,
            "symbol": symbol,
            "type": signal.signal_type.value,
            "strength": signal.strength,
            "price": price,
            "reason": signal.reason
        })
        
        # Execute based on signal type
        if signal.signal_type in (SignalType.BUY, SignalType.STRONG_BUY):
            if symbol not in self._open_positions:
                self._execute_entry(date, signal, price)
                
        elif signal.signal_type in (SignalType.SELL, SignalType.STRONG_SELL):
            if symbol in self._open_positions:
                self._execute_exit(date, symbol, price, "signal")
                
    def _execute_entry(self, date: datetime, signal: Signal, price: float) -> None:
        """Execute a buy entry."""
        symbol = signal.symbol
        
        # Calculate position size
        max_position_value = self.portfolio.cash * (self.config.max_position_pct / 100)
        position_value = max_position_value * signal.strength
        
        # Apply slippage
        entry_price = price * (1 + self.config.slippage_pct / 100)
        
        # Calculate shares
        shares = int(position_value / entry_price)
        
        if shares <= 0 or shares * entry_price > self.portfolio.cash:
            return
            
        # Calculate stop loss and take profit
        stop_loss = entry_price * (1 - self.config.stop_loss_pct / 100)
        take_profit = entry_price * (1 + self.config.take_profit_pct / 100)
        
        # Execute trade
        trade = Trade(
            symbol=symbol,
            side="buy",
            quantity=shares,
            price=entry_price,
            timestamp=date,
            fees=self.config.commission_per_trade
        )
        
        self.portfolio.record_trade(trade)
        
        # Track position
        self._open_positions[symbol] = {
            "entry_date": date,
            "entry_price": entry_price,
            "quantity": shares,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "highest_price": entry_price,
            "trailing_stop": stop_loss
        }
        
        self._trades.append({
            "symbol": symbol,
            "entry_date": date,
            "entry_price": entry_price,
            "quantity": shares,
            "side": "buy"
        })
        
    def _execute_exit(self, date: datetime, symbol: str, 
                      price: float, reason: str) -> None:
        """Execute a sell exit."""
        if symbol not in self._open_positions:
            return
            
        pos_info = self._open_positions[symbol]
        quantity = pos_info["quantity"]
        entry_price = pos_info["entry_price"]
        
        # Apply slippage
        exit_price = price * (1 - self.config.slippage_pct / 100)
        
        # Execute trade
        trade = Trade(
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=exit_price,
            timestamp=date,
            fees=self.config.commission_per_trade
        )
        
        self.portfolio.record_trade(trade)
        
        # Calculate P&L
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = ((exit_price / entry_price) - 1) * 100
        holding_days = (date - pos_info["entry_date"]).days
        
        # Update trade record
        for t in self._trades:
            if t["symbol"] == symbol and "exit_date" not in t:
                t["exit_date"] = date
                t["exit_price"] = exit_price
                t["exit_reason"] = reason
                t["pnl"] = pnl
                t["pnl_pct"] = pnl_pct
                t["holding_days"] = holding_days
                break
                
        # Remove from open positions
        del self._open_positions[symbol]
        
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results and metrics."""
        result = BacktestResult()
        
        # Build equity curve
        if self._equity_history:
            result.equity_curve = pd.DataFrame(self._equity_history)
            result.equity_curve.set_index("date", inplace=True)
            
        result.trades = self._trades
        result.signals = self._signals
        
        # Calculate returns
        if len(result.equity_curve) > 1:
            equity = result.equity_curve["equity"]
            returns = equity.pct_change().dropna()
            
            # Performance metrics
            result.total_return = ((equity.iloc[-1] / equity.iloc[0]) - 1) * 100
            
            # Annualize return
            days = (equity.index[-1] - equity.index[0]).days
            if days > 0:
                result.annual_return = ((1 + result.total_return / 100) ** (365 / days) - 1) * 100
                
            # Risk metrics
            metrics = PortfolioRiskMetrics.calculate_portfolio_metrics(returns, equity)
            result.sharpe_ratio = metrics.get("sharpe_ratio", 0)
            result.sortino_ratio = metrics.get("sortino_ratio", 0)
            result.max_drawdown = metrics.get("max_drawdown", 0)
            result.profit_factor = metrics.get("profit_factor", 0)
            
        # Trade statistics
        completed_trades = [t for t in self._trades if "pnl" in t]
        result.total_trades = len(completed_trades)
        
        if completed_trades:
            pnls = [t["pnl"] for t in completed_trades]
            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p < 0]
            
            result.winning_trades = len(winning)
            result.losing_trades = len(losing)
            result.win_rate = (result.winning_trades / result.total_trades) * 100
            
            result.avg_win = np.mean(winning) if winning else 0
            result.avg_loss = np.mean(losing) if losing else 0
            result.largest_win = max(pnls) if pnls else 0
            result.largest_loss = min(pnls) if pnls else 0
            
            holding_days = [t.get("holding_days", 0) for t in completed_trades]
            result.avg_holding_period = np.mean(holding_days) if holding_days else 0
            
        return result
        
    def run_optimization(self, strategy_class: Type[BaseStrategy],
                        param_grid: Dict[str, List[Any]],
                        data: Dict[str, pd.DataFrame],
                        metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Run parameter optimization for a strategy.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary of parameter names to lists of values
            data: Market data
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Best parameters and results
        """
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_result = None
        best_params = None
        best_metric_value = float("-inf")
        all_results = []
        
        total_combinations = np.prod([len(v) for v in param_values])
        logger.info(f"Running optimization with {total_combinations} parameter combinations")
        
        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))
            
            # Create strategy with parameters
            strategy = strategy_class(**params)
            
            # Run backtest
            result = self.run(strategy, data)
            
            # Get metric value
            metric_value = getattr(result, metric, 0)
            
            all_results.append({
                "params": params,
                "metric": metric_value,
                "result": result
            })
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params
                best_result = result
                
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{total_combinations} combinations")
                
        logger.success(f"Optimization complete. Best {metric}: {best_metric_value:.4f}")
        
        return {
            "best_params": best_params,
            "best_result": best_result,
            "best_metric_value": best_metric_value,
            "all_results": all_results
        }
        
    def walk_forward_analysis(self, strategy: BaseStrategy,
                             data: Dict[str, pd.DataFrame],
                             train_periods: int = 252,
                             test_periods: int = 63) -> List[BacktestResult]:
        """
        Run walk-forward analysis.
        
        Args:
            strategy: Strategy to test
            data: Market data
            train_periods: Training period length (days)
            test_periods: Testing period length (days)
            
        Returns:
            List of test period results
        """
        results = []
        
        # Get aligned dates
        all_dates = self._get_aligned_dates(data)
        
        total_periods = len(all_dates)
        period_start = 0
        
        while period_start + train_periods + test_periods <= total_periods:
            train_end = period_start + train_periods
            test_end = train_end + test_periods
            
            train_dates = all_dates[period_start:train_end]
            test_dates = all_dates[train_end:test_end]
            
            # Create train and test data
            train_data = {
                symbol: df.loc[train_dates[0]:train_dates[-1]]
                for symbol, df in data.items()
            }
            test_data = {
                symbol: df.loc[test_dates[0]:test_dates[-1]]
                for symbol, df in data.items()
            }
            
            # Run backtest on test period
            result = self.run(strategy, test_data)
            result.equity_curve["period_start"] = train_dates[-1]
            results.append(result)
            
            # Move to next period
            period_start += test_periods
            
        logger.info(f"Walk-forward analysis complete: {len(results)} periods")
        
        return results
