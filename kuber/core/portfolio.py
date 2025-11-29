"""
Portfolio Management Module

Handles portfolio tracking, performance metrics, and asset allocation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.current_price
        
    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis."""
        return self.quantity * self.avg_cost
        
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return self.market_value - self.cost_basis
        
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent
        }


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    order_id: Optional[str] = None
    fees: float = 0.0
    
    @property
    def total_value(self) -> float:
        """Calculate total trade value including fees."""
        return self.quantity * self.price + self.fees


class Portfolio:
    """
    Portfolio management class.
    
    Tracks positions, calculates performance metrics, and manages
    portfolio state.
    """
    
    def __init__(self, initial_cash: float = 0.0):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash balance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self._equity_history: List[Dict] = []
        
    def update_from_broker(self, broker: Any) -> None:
        """
        Update portfolio from broker data.
        
        Args:
            broker: RobinhoodBroker instance
        """
        # Update cash
        self.cash = broker.get_buying_power()
        
        # Update positions
        holdings = broker.get_holdings()
        self.positions = {}
        
        for symbol, data in holdings.items():
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=float(data.get("quantity", 0)),
                avg_cost=float(data.get("average_buy_price", 0)),
                current_price=float(data.get("price", 0))
            )
            
        logger.info(f"Portfolio updated: {len(self.positions)} positions, ${self.cash:.2f} cash")
        
    def add_position(self, symbol: str, quantity: float, price: float) -> None:
        """
        Add or update a position.
        
        Args:
            symbol: Stock ticker
            quantity: Number of shares to add
            price: Price per share
        """
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_quantity = pos.quantity + quantity
            total_cost = pos.cost_basis + (quantity * price)
            pos.quantity = total_quantity
            pos.avg_cost = total_cost / total_quantity if total_quantity > 0 else 0
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price
            )
            
    def remove_position(self, symbol: str, quantity: float) -> None:
        """
        Remove shares from a position.
        
        Args:
            symbol: Stock ticker
            quantity: Number of shares to remove
        """
        if symbol not in self.positions:
            logger.warning(f"Position {symbol} not found")
            return
            
        pos = self.positions[symbol]
        pos.quantity -= quantity
        
        if pos.quantity <= 0:
            del self.positions[symbol]
            
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for positions.
        
        Args:
            prices: Dictionary mapping symbol to current price
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
                
    def record_trade(self, trade: Trade) -> None:
        """Record a trade in history."""
        self.trades.append(trade)
        
        if trade.side == "buy":
            self.add_position(trade.symbol, trade.quantity, trade.price)
            self.cash -= trade.total_value
        else:
            self.remove_position(trade.symbol, trade.quantity)
            self.cash += trade.total_value - trade.fees
            
    def record_equity(self) -> None:
        """Record current equity snapshot."""
        self._equity_history.append({
            "timestamp": datetime.now(),
            "equity": self.total_equity,
            "cash": self.cash,
            "positions_value": self.positions_value
        })
        
    @property
    def positions_value(self) -> float:
        """Calculate total value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
        
    @property
    def total_equity(self) -> float:
        """Calculate total portfolio equity."""
        return self.cash + self.positions_value
        
    @property
    def total_cost_basis(self) -> float:
        """Calculate total cost basis of all positions."""
        return sum(pos.cost_basis for pos in self.positions.values())
        
    @property
    def total_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
        
    @property
    def total_return_percent(self) -> float:
        """Calculate total return percentage."""
        if self.initial_cash == 0:
            return 0.0
        return ((self.total_equity - self.initial_cash) / self.initial_cash) * 100
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
        
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has a position in symbol."""
        return symbol in self.positions
        
    def get_position_size(self, symbol: str) -> float:
        """Get quantity held for a symbol."""
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0
        
    def get_allocation(self) -> Dict[str, float]:
        """
        Calculate portfolio allocation percentages.
        
        Returns:
            Dictionary mapping symbol to allocation percentage
        """
        total = self.total_equity
        if total == 0:
            return {}
            
        allocation = {"cash": (self.cash / total) * 100}
        for symbol, pos in self.positions.items():
            allocation[symbol] = (pos.market_value / total) * 100
            
        return allocation
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert positions to DataFrame."""
        if not self.positions:
            return pd.DataFrame()
            
        data = [pos.to_dict() for pos in self.positions.values()]
        df = pd.DataFrame(data)
        df.set_index("symbol", inplace=True)
        return df
        
    def get_equity_history(self) -> pd.DataFrame:
        """Get equity history as DataFrame."""
        if not self._equity_history:
            return pd.DataFrame()
            
        df = pd.DataFrame(self._equity_history)
        df.set_index("timestamp", inplace=True)
        return df
        
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary with various performance metrics
        """
        equity_df = self.get_equity_history()
        
        if equity_df.empty or len(equity_df) < 2:
            return {}
            
        returns = equity_df["equity"].pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) - 1
        avg_daily_return = returns.mean()
        volatility = returns.std()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (avg_daily_return / volatility) * np.sqrt(252) if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = (avg_daily_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            "total_return": total_return * 100,
            "avg_daily_return": avg_daily_return * 100,
            "volatility": volatility * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown * 100,
            "num_trades": len(self.trades),
            "win_rate": self._calculate_win_rate()
        }
        
    def _calculate_win_rate(self) -> float:
        """Calculate percentage of winning trades."""
        if not self.trades:
            return 0.0
            
        # Group trades by symbol to calculate realized P&L
        # This is a simplified calculation
        winning_trades = 0
        total_closed = 0
        
        # Track buy costs and sells
        symbol_buys: Dict[str, List[float]] = {}
        
        for trade in self.trades:
            if trade.side == "buy":
                if trade.symbol not in symbol_buys:
                    symbol_buys[trade.symbol] = []
                symbol_buys[trade.symbol].append(trade.price)
            else:
                if trade.symbol in symbol_buys and symbol_buys[trade.symbol]:
                    avg_buy = np.mean(symbol_buys[trade.symbol])
                    if trade.price > avg_buy:
                        winning_trades += 1
                    total_closed += 1
                    
        return (winning_trades / total_closed * 100) if total_closed > 0 else 0.0
        
    def summary(self) -> str:
        """Get portfolio summary string."""
        lines = [
            "=" * 50,
            "PORTFOLIO SUMMARY",
            "=" * 50,
            f"Total Equity:      ${self.total_equity:,.2f}",
            f"Cash:              ${self.cash:,.2f}",
            f"Positions Value:   ${self.positions_value:,.2f}",
            f"Unrealized P&L:    ${self.total_unrealized_pnl:,.2f}",
            f"Total Return:      {self.total_return_percent:.2f}%",
            f"Number of Positions: {len(self.positions)}",
            "-" * 50,
        ]
        
        if self.positions:
            lines.append("POSITIONS:")
            for symbol, pos in sorted(self.positions.items()):
                pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                lines.append(
                    f"  {symbol:8s} | {pos.quantity:8.2f} shares @ ${pos.current_price:8.2f} | "
                    f"P&L: {pnl_sign}${pos.unrealized_pnl:,.2f} ({pnl_sign}{pos.unrealized_pnl_percent:.1f}%)"
                )
                
        lines.append("=" * 50)
        return "\n".join(lines)
        
    def __repr__(self) -> str:
        return f"Portfolio(equity=${self.total_equity:.2f}, positions={len(self.positions)})"
