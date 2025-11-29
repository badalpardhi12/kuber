"""
Trading Engine Module

The central orchestrator for algorithmic trading operations.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
from enum import Enum
import schedule
from loguru import logger

from kuber.core.broker import RobinhoodBroker
from kuber.core.portfolio import Portfolio, Trade
from kuber.strategies.base import BaseStrategy, Signal, SignalType


class EngineState(Enum):
    """Trading engine states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class TradingMode(Enum):
    """Trading modes."""
    PAPER = "paper"  # Simulated trading
    LIVE = "live"    # Real money trading


class TradingEngine:
    """
    Central trading engine that orchestrates all trading operations.
    
    Manages broker connections, strategies, portfolio, and order execution.
    """
    
    def __init__(self, broker: Optional[RobinhoodBroker] = None,
                 mode: TradingMode = TradingMode.PAPER):
        """
        Initialize the trading engine.
        
        Args:
            broker: RobinhoodBroker instance for live trading
            mode: Trading mode (PAPER or LIVE)
        """
        self.broker = broker
        self.mode = mode
        self.portfolio = Portfolio()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.state = EngineState.STOPPED
        
        # Trading universe
        self.symbols: List[str] = []
        
        # Configuration
        self.config = {
            "max_position_size_pct": 10.0,  # Max 10% in single position
            "max_total_positions": 20,
            "min_trade_interval_seconds": 60,
            "stop_loss_pct": 5.0,  # Default stop loss
            "take_profit_pct": 15.0,  # Default take profit
            "enable_stop_loss": True,
            "enable_take_profit": False,
        }
        
        # State tracking
        self._last_trade_time: Dict[str, datetime] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.signals_generated: List[Dict] = []
        self.orders_executed: List[Dict] = []
        
    def add_strategy(self, name: str, strategy: BaseStrategy) -> None:
        """
        Add a trading strategy.
        
        Args:
            name: Strategy identifier
            strategy: Strategy instance
        """
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")
        
    def remove_strategy(self, name: str) -> None:
        """Remove a trading strategy."""
        if name in self.strategies:
            del self.strategies[name]
            logger.info(f"Removed strategy: {name}")
            
    def set_symbols(self, symbols: List[str]) -> None:
        """
        Set the trading universe.
        
        Args:
            symbols: List of stock ticker symbols to trade
        """
        self.symbols = [s.upper() for s in symbols]
        logger.info(f"Trading universe set: {len(self.symbols)} symbols")
        
    def set_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        self.config.update(kwargs)
        
    def initialize(self) -> bool:
        """
        Initialize the engine and connect to broker.
        
        Returns:
            True if initialization successful
        """
        logger.info("Initializing trading engine...")
        
        if self.mode == TradingMode.LIVE:
            if not self.broker:
                logger.error("Broker required for live trading")
                return False
                
            if not self.broker.is_logged_in:
                if not self.broker.login():
                    logger.error("Failed to connect to broker")
                    return False
                    
            # Sync portfolio with broker
            self.portfolio.update_from_broker(self.broker)
            self.portfolio.initial_cash = self.portfolio.total_equity
            
        logger.success("Trading engine initialized")
        return True
        
    def start(self, interval_seconds: int = 60) -> None:
        """
        Start the trading engine.
        
        Args:
            interval_seconds: How often to run strategy checks
        """
        if self.state == EngineState.RUNNING:
            logger.warning("Engine already running")
            return
            
        if not self.strategies:
            logger.error("No strategies configured")
            return
            
        if not self.symbols:
            logger.error("No symbols configured")
            return
            
        self._running = True
        self.state = EngineState.RUNNING
        logger.success("Trading engine started")
        
        # Schedule regular strategy execution
        schedule.every(interval_seconds).seconds.do(self._run_cycle)
        
        # Run in background thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
    def stop(self) -> None:
        """Stop the trading engine."""
        self._running = False
        self.state = EngineState.STOPPED
        schedule.clear()
        logger.info("Trading engine stopped")
        
    def pause(self) -> None:
        """Pause trading (stops new trades but keeps monitoring)."""
        self.state = EngineState.PAUSED
        logger.info("Trading engine paused")
        
    def resume(self) -> None:
        """Resume trading from paused state."""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING
            logger.info("Trading engine resumed")
            
    def _run_loop(self) -> None:
        """Main event loop for the engine."""
        while self._running:
            schedule.run_pending()
            time.sleep(1)
            
    def _run_cycle(self) -> None:
        """Run one cycle of strategy evaluation and trade execution."""
        if self.state != EngineState.RUNNING:
            return
            
        try:
            # Check if market is open (for live trading)
            if self.mode == TradingMode.LIVE and self.broker:
                if not self.broker.is_market_open():
                    logger.debug("Market closed, skipping cycle")
                    return
                    
            # Get current prices
            prices = self._get_current_prices()
            if not prices:
                return
                
            # Update portfolio prices
            self.portfolio.update_prices(prices)
            
            # Collect signals from all strategies
            all_signals = []
            for name, strategy in self.strategies.items():
                signals = self._get_strategy_signals(strategy, prices)
                for signal in signals:
                    signal.strategy_name = name
                all_signals.extend(signals)
                
            # Process signals
            for signal in all_signals:
                self._process_signal(signal)
                
            # Record equity snapshot
            self.portfolio.record_equity()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            self.state = EngineState.ERROR
            
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        if self.mode == TradingMode.LIVE and self.broker:
            return self.broker.get_latest_prices(self.symbols)
        else:
            # Paper trading - would need simulated data
            return {}
            
    def _get_strategy_signals(self, strategy: BaseStrategy,
                              prices: Dict[str, float]) -> List[Signal]:
        """
        Get signals from a strategy for all symbols.
        
        Args:
            strategy: Strategy to evaluate
            prices: Current prices
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for symbol in self.symbols:
            # Get historical data
            if self.broker:
                df = self.broker.get_historicals(symbol, interval="day", span="month")
            else:
                continue
                
            if df.empty:
                continue
                
            # Get current position
            position = self.portfolio.get_position(symbol)
            
            # Generate signal
            signal = strategy.generate_signal(
                symbol=symbol,
                data=df,
                current_price=prices.get(symbol, 0),
                position=position
            )
            
            if signal and signal.signal_type != SignalType.HOLD:
                signals.append(signal)
                self.signals_generated.append({
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "signal": signal.signal_type.value,
                    "strength": signal.strength,
                    "strategy": strategy.__class__.__name__
                })
                
        return signals
        
    def _process_signal(self, signal: Signal) -> None:
        """
        Process a trading signal and execute orders if appropriate.
        
        Args:
            signal: Trading signal to process
        """
        symbol = signal.symbol
        
        # Check trade interval
        last_trade = self._last_trade_time.get(symbol)
        if last_trade:
            elapsed = (datetime.now() - last_trade).total_seconds()
            if elapsed < self.config["min_trade_interval_seconds"]:
                return
                
        # Calculate position size
        size = self._calculate_position_size(signal)
        if size <= 0:
            return
            
        # Execute trade
        if signal.signal_type == SignalType.BUY:
            self._execute_buy(signal, size)
        elif signal.signal_type == SignalType.SELL:
            self._execute_sell(signal, size)
            
    def _calculate_position_size(self, signal: Signal) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            
        Returns:
            Number of shares to trade
        """
        max_position_value = self.portfolio.total_equity * (
            self.config["max_position_size_pct"] / 100
        )
        
        # Get current price
        price = signal.price
        if price <= 0:
            return 0
            
        # For sell signals, sell existing position
        if signal.signal_type == SignalType.SELL:
            position = self.portfolio.get_position(signal.symbol)
            return position.quantity if position else 0
            
        # For buy signals
        if signal.signal_type == SignalType.BUY:
            # Check position limits
            if len(self.portfolio.positions) >= self.config["max_total_positions"]:
                if signal.symbol not in self.portfolio.positions:
                    logger.debug(f"Max positions reached, skipping {signal.symbol}")
                    return 0
                    
            # Calculate shares based on signal strength
            position_value = max_position_value * signal.strength
            shares = position_value / price
            
            # Check if we have enough cash
            available_cash = self.portfolio.cash * 0.95  # Keep 5% buffer
            max_shares = available_cash / price
            
            return min(shares, max_shares)
            
        return 0
        
    def _execute_buy(self, signal: Signal, quantity: float) -> None:
        """Execute a buy order."""
        if quantity <= 0:
            return
            
        symbol = signal.symbol
        price = signal.price
        
        logger.info(f"BUY signal: {quantity:.2f} shares of {symbol} @ ${price:.2f}")
        
        if self.mode == TradingMode.LIVE and self.broker:
            order = self.broker.buy_market(symbol, quantity)
            if order:
                self.orders_executed.append({
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "side": "buy",
                    "quantity": quantity,
                    "price": price,
                    "order_id": order.get("id")
                })
        else:
            # Paper trading - record trade
            trade = Trade(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                price=price,
                timestamp=datetime.now()
            )
            self.portfolio.record_trade(trade)
            
        self._last_trade_time[symbol] = datetime.now()
        
    def _execute_sell(self, signal: Signal, quantity: float) -> None:
        """Execute a sell order."""
        if quantity <= 0:
            return
            
        symbol = signal.symbol
        price = signal.price
        
        logger.info(f"SELL signal: {quantity:.2f} shares of {symbol} @ ${price:.2f}")
        
        if self.mode == TradingMode.LIVE and self.broker:
            order = self.broker.sell_market(symbol, quantity)
            if order:
                self.orders_executed.append({
                    "timestamp": datetime.now(),
                    "symbol": symbol,
                    "side": "sell",
                    "quantity": quantity,
                    "price": price,
                    "order_id": order.get("id")
                })
        else:
            # Paper trading
            trade = Trade(
                symbol=symbol,
                side="sell",
                quantity=quantity,
                price=price,
                timestamp=datetime.now()
            )
            self.portfolio.record_trade(trade)
            
        self._last_trade_time[symbol] = datetime.now()
        
    def run_once(self) -> Dict[str, List[Signal]]:
        """
        Run one cycle manually (useful for testing/debugging).
        
        Returns:
            Dictionary of signals by strategy name
        """
        prices = self._get_current_prices()
        self.portfolio.update_prices(prices)
        
        all_signals = {}
        for name, strategy in self.strategies.items():
            signals = self._get_strategy_signals(strategy, prices)
            all_signals[name] = signals
            
        return all_signals
        
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            "state": self.state.value,
            "mode": self.mode.value,
            "strategies": list(self.strategies.keys()),
            "symbols": self.symbols,
            "portfolio_equity": self.portfolio.total_equity,
            "positions": len(self.portfolio.positions),
            "signals_count": len(self.signals_generated),
            "orders_count": len(self.orders_executed)
        }
        
    def __repr__(self) -> str:
        return f"TradingEngine(state={self.state.value}, strategies={len(self.strategies)})"
