"""
Risk Management Module

Provides position sizing, stop-loss management, and risk metrics.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, List
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class RiskLevel(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_position_pct: float = 10.0      # Max % of portfolio in single position
    max_sector_pct: float = 30.0        # Max % of portfolio in single sector
    max_total_positions: int = 20       # Maximum number of positions
    stop_loss_pct: float = 5.0          # Default stop loss percentage
    trailing_stop_pct: float = 3.0      # Trailing stop percentage
    take_profit_pct: float = 15.0       # Default take profit percentage
    max_daily_loss_pct: float = 3.0     # Max daily portfolio loss
    max_drawdown_pct: float = 15.0      # Max drawdown before reducing risk
    risk_per_trade_pct: float = 1.0     # Risk per trade as % of portfolio
    margin_usage_max: float = 0.0       # Max margin usage (0 = no margin)
    
    @classmethod
    def conservative(cls) -> "RiskParameters":
        """Conservative risk parameters."""
        return cls(
            max_position_pct=5.0,
            max_total_positions=30,
            stop_loss_pct=3.0,
            trailing_stop_pct=2.0,
            take_profit_pct=10.0,
            max_daily_loss_pct=2.0,
            max_drawdown_pct=10.0,
            risk_per_trade_pct=0.5
        )
        
    @classmethod
    def moderate(cls) -> "RiskParameters":
        """Moderate risk parameters."""
        return cls(
            max_position_pct=10.0,
            max_total_positions=20,
            stop_loss_pct=5.0,
            trailing_stop_pct=3.0,
            take_profit_pct=15.0,
            max_daily_loss_pct=3.0,
            max_drawdown_pct=15.0,
            risk_per_trade_pct=1.0
        )
        
    @classmethod
    def aggressive(cls) -> "RiskParameters":
        """Aggressive risk parameters."""
        return cls(
            max_position_pct=20.0,
            max_total_positions=10,
            stop_loss_pct=8.0,
            trailing_stop_pct=5.0,
            take_profit_pct=25.0,
            max_daily_loss_pct=5.0,
            max_drawdown_pct=25.0,
            risk_per_trade_pct=2.0
        )


class RiskManager:
    """
    Risk Manager for portfolio and trade-level risk management.
    
    Handles position sizing, stop-loss management, and risk metrics.
    """
    
    def __init__(self, params: Optional[RiskParameters] = None):
        """
        Initialize Risk Manager.
        
        Args:
            params: Risk parameters (defaults to moderate)
        """
        self.params = params or RiskParameters.moderate()
        self._daily_pnl: List[float] = []
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0
        
    def calculate_position_size(self, 
                               account_value: float,
                               current_price: float,
                               stop_loss_price: Optional[float] = None,
                               volatility: Optional[float] = None,
                               signal_strength: float = 1.0) -> int:
        """
        Calculate position size based on risk management rules.
        
        Uses the position sizing formula:
        Position Size = (Account Value * Risk %) / (Entry - Stop Loss)
        
        Args:
            account_value: Total account value
            current_price: Current stock price
            stop_loss_price: Stop loss price (if known)
            volatility: Stock volatility (for ATR-based sizing)
            signal_strength: Signal strength from strategy (0-1)
            
        Returns:
            Number of shares to buy
        """
        if current_price <= 0 or account_value <= 0:
            return 0
            
        # Maximum position value
        max_position_value = account_value * (self.params.max_position_pct / 100)
        
        # Adjust for signal strength
        adjusted_position_value = max_position_value * signal_strength
        
        # If we have stop loss price, use risk-based sizing
        if stop_loss_price and stop_loss_price > 0:
            risk_per_share = abs(current_price - stop_loss_price)
            if risk_per_share > 0:
                risk_amount = account_value * (self.params.risk_per_trade_pct / 100)
                risk_based_shares = risk_amount / risk_per_share
                max_shares = adjusted_position_value / current_price
                return int(min(risk_based_shares, max_shares))
                
        # If we have volatility, adjust position size inversely
        if volatility and volatility > 0:
            # Higher volatility = smaller position
            volatility_factor = min(1.0, 0.02 / volatility)  # Target 2% daily volatility
            adjusted_position_value *= volatility_factor
            
        # Calculate shares
        shares = int(adjusted_position_value / current_price)
        
        return max(0, shares)
        
    def calculate_stop_loss(self, entry_price: float, 
                           volatility: Optional[float] = None,
                           atr: Optional[float] = None,
                           is_long: bool = True) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Trade entry price
            volatility: Historical volatility
            atr: Average True Range
            is_long: True for long position
            
        Returns:
            Stop loss price
        """
        if atr and atr > 0:
            # ATR-based stop loss (2x ATR is common)
            stop_distance = atr * 2
        else:
            # Percentage-based stop loss
            stop_distance = entry_price * (self.params.stop_loss_pct / 100)
            
        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
            
    def calculate_take_profit(self, entry_price: float,
                             stop_loss_price: Optional[float] = None,
                             reward_ratio: float = 2.0,
                             is_long: bool = True) -> float:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Trade entry price
            stop_loss_price: Stop loss price (for R:R calculation)
            reward_ratio: Risk:Reward ratio target
            is_long: True for long position
            
        Returns:
            Take profit price
        """
        if stop_loss_price:
            # Use risk:reward ratio
            risk = abs(entry_price - stop_loss_price)
            reward = risk * reward_ratio
        else:
            # Use percentage
            reward = entry_price * (self.params.take_profit_pct / 100)
            
        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward
            
    def calculate_trailing_stop(self, current_price: float,
                               highest_price: float,
                               atr: Optional[float] = None,
                               is_long: bool = True) -> float:
        """
        Calculate trailing stop price.
        
        Args:
            current_price: Current price
            highest_price: Highest price since entry (for long)
            atr: ATR for volatility-based trailing
            is_long: True for long position
            
        Returns:
            Trailing stop price
        """
        if atr and atr > 0:
            trail_distance = atr * 2
        else:
            trail_distance = highest_price * (self.params.trailing_stop_pct / 100)
            
        if is_long:
            return highest_price - trail_distance
        else:
            return highest_price + trail_distance  # Would use lowest_price for short
            
    def check_position_allowed(self, 
                              portfolio_value: float,
                              current_positions: int,
                              position_value: float,
                              sector: Optional[str] = None,
                              sector_exposure: Optional[float] = None) -> tuple:
        """
        Check if a new position is allowed by risk rules.
        
        Args:
            portfolio_value: Current portfolio value
            current_positions: Number of current positions
            position_value: Value of proposed position
            sector: Sector of the stock
            sector_exposure: Current exposure to this sector
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check position count
        if current_positions >= self.params.max_total_positions:
            return False, f"Maximum positions ({self.params.max_total_positions}) reached"
            
        # Check position size
        position_pct = (position_value / portfolio_value) * 100
        if position_pct > self.params.max_position_pct:
            return False, f"Position size {position_pct:.1f}% exceeds max {self.params.max_position_pct}%"
            
        # Check sector exposure
        if sector and sector_exposure:
            new_sector_pct = sector_exposure + (position_value / portfolio_value) * 100
            if new_sector_pct > self.params.max_sector_pct:
                return False, f"Sector exposure {new_sector_pct:.1f}% exceeds max {self.params.max_sector_pct}%"
                
        # Check drawdown
        if self._current_drawdown >= self.params.max_drawdown_pct:
            return False, f"Max drawdown {self.params.max_drawdown_pct}% reached"
            
        return True, "Position allowed"
        
    def update_equity(self, equity: float) -> None:
        """
        Update equity tracking for drawdown calculation.
        
        Args:
            equity: Current portfolio equity
        """
        if equity > self._peak_equity:
            self._peak_equity = equity
            
        if self._peak_equity > 0:
            self._current_drawdown = ((self._peak_equity - equity) / self._peak_equity) * 100
            
    def record_daily_pnl(self, pnl: float) -> None:
        """Record daily P&L for tracking."""
        self._daily_pnl.append(pnl)
        
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        return self._current_drawdown
        
    def is_in_drawdown(self) -> bool:
        """Check if portfolio is in significant drawdown."""
        return self._current_drawdown >= (self.params.max_drawdown_pct * 0.5)
        
    def get_risk_adjustment_factor(self) -> float:
        """
        Get factor to adjust position sizes based on drawdown.
        
        Reduces size as drawdown increases.
        
        Returns:
            Factor between 0.25 and 1.0
        """
        if self._current_drawdown <= 0:
            return 1.0
            
        # Linear reduction based on drawdown
        factor = 1.0 - (self._current_drawdown / self.params.max_drawdown_pct)
        return max(0.25, min(1.0, factor))


class PortfolioRiskMetrics:
    """Calculate portfolio-level risk metrics."""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95,
                     method: str = "historical") -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Series of returns
            confidence: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
            
        Returns:
            VaR as a positive percentage
        """
        if len(returns) < 30:
            return 0.0
            
        if method == "historical":
            var = np.percentile(returns, (1 - confidence) * 100)
        else:
            # Parametric VaR
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence)
            var = returns.mean() + z_score * returns.std()
            
        return abs(var)
        
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence: Confidence level
            
        Returns:
            CVaR as a positive percentage
        """
        if len(returns) < 30:
            return 0.0
            
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        
        return abs(cvar)
        
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                              risk_free_rate: float = 0.02,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 30 or returns.std() == 0:
            return 0.0
            
        daily_rf = risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
        return sharpe
        
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                               risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (only considers downside volatility).
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 30:
            return 0.0
            
        daily_rf = risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
            
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
        return sortino
        
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            Dict with max_drawdown, start, end, duration
        """
        if len(equity_curve) < 2:
            return {"max_drawdown": 0.0}
            
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        
        max_dd = drawdowns.min()
        max_dd_end = drawdowns.idxmin()
        
        # Find start of max drawdown
        max_dd_start = equity_curve[:max_dd_end].idxmax()
        
        return {
            "max_drawdown": abs(max_dd) * 100,
            "start": max_dd_start,
            "end": max_dd_end,
            "recovery": None  # Would need to find when equity > previous peak
        }
        
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series,
                              equity_curve: pd.Series,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).
        
        Args:
            returns: Series of returns
            equity_curve: Series of portfolio values
            periods_per_year: Trading periods per year
            
        Returns:
            Calmar ratio
        """
        if len(returns) < 30:
            return 0.0
            
        annual_return = returns.mean() * periods_per_year
        max_dd = PortfolioRiskMetrics.calculate_max_drawdown(equity_curve)["max_drawdown"]
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
            
        return (annual_return * 100) / max_dd
        
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, 
                      market_returns: pd.Series) -> float:
        """
        Calculate Beta relative to market.
        
        Args:
            stock_returns: Stock/portfolio returns
            market_returns: Market benchmark returns
            
        Returns:
            Beta coefficient
        """
        if len(stock_returns) != len(market_returns) or len(stock_returns) < 30:
            return 1.0
            
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 1.0
            
        return covariance / market_variance
        
    @staticmethod
    def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio positions.
        
        Args:
            returns_df: DataFrame with returns for each position
            
        Returns:
            Correlation matrix
        """
        return returns_df.corr()
        
    @staticmethod
    def calculate_portfolio_metrics(returns: pd.Series,
                                   equity_curve: pd.Series,
                                   risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            returns: Series of portfolio returns
            equity_curve: Series of portfolio values
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of all risk metrics
        """
        metrics = {}
        
        # Return metrics
        metrics["total_return"] = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100
        metrics["annual_return"] = returns.mean() * 252 * 100
        metrics["volatility"] = returns.std() * np.sqrt(252) * 100
        
        # Risk-adjusted metrics
        metrics["sharpe_ratio"] = PortfolioRiskMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
        metrics["sortino_ratio"] = PortfolioRiskMetrics.calculate_sortino_ratio(returns, risk_free_rate)
        
        # Drawdown metrics
        dd_info = PortfolioRiskMetrics.calculate_max_drawdown(equity_curve)
        metrics["max_drawdown"] = dd_info["max_drawdown"]
        metrics["calmar_ratio"] = PortfolioRiskMetrics.calculate_calmar_ratio(returns, equity_curve)
        
        # VaR metrics
        metrics["var_95"] = PortfolioRiskMetrics.calculate_var(returns, 0.95)
        metrics["cvar_95"] = PortfolioRiskMetrics.calculate_cvar(returns, 0.95)
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        metrics["win_rate"] = (winning_days / total_days) * 100 if total_days > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return metrics
