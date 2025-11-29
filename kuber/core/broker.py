"""
Robinhood Broker Integration

This module handles all communication with the Robinhood API using robin_stocks.
It provides authentication, account management, market data, and order execution.
"""

import os
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import pyotp
from loguru import logger

try:
    import robin_stocks.robinhood as rh
except ImportError:
    raise ImportError("Please install robin_stocks: pip install robin_stocks")


class RobinhoodBroker:
    """
    Robinhood broker integration class.
    
    Handles authentication, account data retrieval, market data,
    and order execution through the Robinhood API.
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None,
                 totp_secret: Optional[str] = None, pickle_name: str = "robinhood.pickle"):
        """
        Initialize the Robinhood broker.
        
        Args:
            username: Robinhood username/email
            password: Robinhood password
            totp_secret: TOTP secret for 2FA (optional)
            pickle_name: Name of pickle file to store session
        """
        self.username = username or os.getenv("ROBINHOOD_USERNAME")
        self.password = password or os.getenv("ROBINHOOD_PASSWORD")
        self.totp_secret = totp_secret or os.getenv("ROBINHOOD_TOTP_SECRET")
        self.pickle_name = pickle_name
        self._logged_in = False
        self._account_profile = None
        
    def login(self, store_session: bool = True) -> bool:
        """
        Login to Robinhood.
        
        Args:
            store_session: Whether to store the session in a pickle file
            
        Returns:
            True if login successful, False otherwise
        """
        if not self.username or not self.password:
            logger.error("Username and password are required")
            return False
            
        try:
            login_kwargs = {
                "username": self.username,
                "password": self.password,
                "store_session": store_session,
                "pickle_name": self.pickle_name
            }
            
            # Add TOTP code if 2FA is enabled
            if self.totp_secret:
                totp = pyotp.TOTP(self.totp_secret)
                login_kwargs["mfa_code"] = totp.now()
                
            result = rh.login(**login_kwargs)
            
            if result:
                self._logged_in = True
                self._account_profile = rh.profiles.load_account_profile()
                logger.success(f"Successfully logged in as {self.username}")
                return True
            else:
                logger.error("Login failed - check credentials")
                return False
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
            
    def logout(self) -> None:
        """Logout from Robinhood."""
        rh.logout()
        self._logged_in = False
        self._account_profile = None
        logger.info("Logged out from Robinhood")
        
    @property
    def is_logged_in(self) -> bool:
        """Check if currently logged in."""
        return self._logged_in
        
    def _require_login(self) -> None:
        """Raise error if not logged in."""
        if not self._logged_in:
            raise RuntimeError("Not logged in. Call login() first.")
            
    # ==================== Account Information ====================
    
    def get_account_profile(self) -> Dict[str, Any]:
        """Get account profile information."""
        self._require_login()
        return rh.profiles.load_account_profile()
        
    def get_portfolio_profile(self) -> Dict[str, Any]:
        """Get portfolio profile with equity and cash info."""
        self._require_login()
        return rh.profiles.load_portfolio_profile()
        
    def get_buying_power(self) -> float:
        """Get available buying power."""
        self._require_login()
        profile = rh.profiles.load_account_profile()
        return float(profile.get("buying_power", 0))
        
    def get_cash_balance(self) -> float:
        """Get cash balance."""
        self._require_login()
        profile = rh.profiles.load_account_profile()
        return float(profile.get("cash", 0))
        
    def get_portfolio_equity(self) -> float:
        """Get total portfolio equity."""
        self._require_login()
        profile = rh.profiles.load_portfolio_profile()
        return float(profile.get("equity", 0))
        
    def get_holdings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all current stock holdings.
        
        Returns:
            Dictionary with symbol as key and holding details as value
        """
        self._require_login()
        return rh.account.build_holdings()
        
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open stock positions."""
        self._require_login()
        return rh.account.get_open_stock_positions()
        
    def get_day_trades_count(self) -> int:
        """Get number of day trades in the last 5 trading days."""
        self._require_login()
        day_trades = rh.account.get_day_trades()
        return len(day_trades) if day_trades else 0
        
    # ==================== Market Data ====================
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Quote data dictionary
        """
        return rh.stocks.get_stock_quote_by_symbol(symbol)
        
    def get_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Get quotes for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            List of quote dictionaries
        """
        return rh.stocks.get_quotes(symbols)
        
    def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price for a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Latest price as float
        """
        prices = rh.stocks.get_latest_price(symbol)
        return float(prices[0]) if prices else 0.0
        
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest prices for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary mapping symbol to price
        """
        prices = rh.stocks.get_latest_price(symbols)
        return {symbol: float(price) if price else 0.0 
                for symbol, price in zip(symbols, prices)}
        
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a stock."""
        data = rh.stocks.get_fundamentals(symbol)
        return data[0] if data else {}
        
    def get_historicals(self, symbol: str, interval: str = "day",
                       span: str = "year", bounds: str = "regular") -> pd.DataFrame:
        """
        Get historical price data for a stock.
        
        Args:
            symbol: Stock ticker symbol
            interval: Candle interval - '5minute', '10minute', 'hour', 'day', 'week'
            span: Time span - 'day', 'week', 'month', '3month', 'year', '5year'
            bounds: Market bounds - 'regular', 'extended', 'trading'
            
        Returns:
            DataFrame with OHLCV data
        """
        historicals = rh.stocks.get_stock_historicals(
            symbol, interval=interval, span=span, bounds=bounds
        )
        
        if not historicals:
            return pd.DataFrame()
            
        df = pd.DataFrame(historicals)
        df["begins_at"] = pd.to_datetime(df["begins_at"])
        df.set_index("begins_at", inplace=True)
        
        # Convert price columns to float
        for col in ["open_price", "close_price", "high_price", "low_price"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
                
        # Rename columns for standard OHLCV format
        df = df.rename(columns={
            "open_price": "open",
            "close_price": "close",
            "high_price": "high",
            "low_price": "low",
            "volume": "volume"
        })
        
        return df[["open", "high", "low", "close", "volume"]]
        
    def get_market_hours(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get market hours for a specific date."""
        return rh.markets.get_market_hours("XNYS", date)
        
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        hours = rh.markets.get_market_today_hours("XNYS")
        if not hours or not hours.get("is_open"):
            return False
        return hours.get("is_open", False)
        
    def get_top_movers(self, direction: str = "up") -> List[Dict[str, Any]]:
        """
        Get top movers in S&P 500.
        
        Args:
            direction: 'up' or 'down'
        """
        return rh.markets.get_top_movers_sp500(direction)
        
    # ==================== Order Execution ====================
    
    def buy_market(self, symbol: str, quantity: float,
                   time_in_force: str = "gtc") -> Dict[str, Any]:
        """
        Place a market buy order.
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares (can be fractional)
            time_in_force: Order duration - 'gfd', 'gtc', 'ioc', 'opg'
            
        Returns:
            Order response dictionary
        """
        self._require_login()
        logger.info(f"Placing market buy order: {quantity} shares of {symbol}")
        
        # Use fractional order if quantity is not whole
        if quantity != int(quantity):
            return rh.orders.order_buy_fractional_by_quantity(
                symbol, quantity, timeInForce=time_in_force
            )
        else:
            return rh.orders.order_buy_market(
                symbol, int(quantity), timeInForce=time_in_force
            )
            
    def sell_market(self, symbol: str, quantity: float,
                    time_in_force: str = "gtc") -> Dict[str, Any]:
        """
        Place a market sell order.
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares (can be fractional)
            time_in_force: Order duration - 'gfd', 'gtc', 'ioc', 'opg'
            
        Returns:
            Order response dictionary
        """
        self._require_login()
        logger.info(f"Placing market sell order: {quantity} shares of {symbol}")
        
        if quantity != int(quantity):
            return rh.orders.order_sell_fractional_by_quantity(
                symbol, quantity, timeInForce=time_in_force
            )
        else:
            return rh.orders.order_sell_market(
                symbol, int(quantity), timeInForce=time_in_force
            )
            
    def buy_limit(self, symbol: str, quantity: int, limit_price: float,
                  time_in_force: str = "gtc") -> Dict[str, Any]:
        """
        Place a limit buy order.
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares
            limit_price: Maximum price to pay
            time_in_force: Order duration
            
        Returns:
            Order response dictionary
        """
        self._require_login()
        logger.info(f"Placing limit buy order: {quantity} shares of {symbol} @ ${limit_price}")
        return rh.orders.order_buy_limit(symbol, quantity, limit_price, 
                                         timeInForce=time_in_force)
        
    def sell_limit(self, symbol: str, quantity: int, limit_price: float,
                   time_in_force: str = "gtc") -> Dict[str, Any]:
        """
        Place a limit sell order.
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares
            limit_price: Minimum price to accept
            time_in_force: Order duration
            
        Returns:
            Order response dictionary
        """
        self._require_login()
        logger.info(f"Placing limit sell order: {quantity} shares of {symbol} @ ${limit_price}")
        return rh.orders.order_sell_limit(symbol, quantity, limit_price,
                                          timeInForce=time_in_force)
        
    def buy_stop_loss(self, symbol: str, quantity: int, stop_price: float,
                      time_in_force: str = "gtc") -> Dict[str, Any]:
        """Place a stop loss buy order."""
        self._require_login()
        return rh.orders.order_buy_stop_loss(symbol, quantity, stop_price,
                                             timeInForce=time_in_force)
        
    def sell_stop_loss(self, symbol: str, quantity: int, stop_price: float,
                       time_in_force: str = "gtc") -> Dict[str, Any]:
        """Place a stop loss sell order."""
        self._require_login()
        return rh.orders.order_sell_stop_loss(symbol, quantity, stop_price,
                                              timeInForce=time_in_force)
        
    def buy_stop_limit(self, symbol: str, quantity: int, limit_price: float,
                       stop_price: float, time_in_force: str = "gtc") -> Dict[str, Any]:
        """Place a stop limit buy order."""
        self._require_login()
        return rh.orders.order_buy_stop_limit(symbol, quantity, limit_price,
                                              stop_price, timeInForce=time_in_force)
        
    def sell_stop_limit(self, symbol: str, quantity: int, limit_price: float,
                        stop_price: float, time_in_force: str = "gtc") -> Dict[str, Any]:
        """Place a stop limit sell order."""
        self._require_login()
        return rh.orders.order_sell_stop_limit(symbol, quantity, limit_price,
                                               stop_price, timeInForce=time_in_force)
        
    def buy_trailing_stop(self, symbol: str, quantity: int, trail_amount: float,
                          trail_type: str = "percentage",
                          time_in_force: str = "gtc") -> Dict[str, Any]:
        """
        Place a trailing stop buy order.
        
        Args:
            symbol: Stock ticker symbol
            quantity: Number of shares
            trail_amount: Trail amount (percentage or dollar amount)
            trail_type: 'percentage' or 'amount'
            time_in_force: Order duration
        """
        self._require_login()
        return rh.orders.order_buy_trailing_stop(symbol, quantity, trail_amount,
                                                 trail_type, timeInForce=time_in_force)
        
    def sell_trailing_stop(self, symbol: str, quantity: int, trail_amount: float,
                           trail_type: str = "percentage",
                           time_in_force: str = "gtc") -> Dict[str, Any]:
        """Place a trailing stop sell order."""
        self._require_login()
        return rh.orders.order_sell_trailing_stop(symbol, quantity, trail_amount,
                                                  trail_type, timeInForce=time_in_force)
        
    # ==================== Order Management ====================
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open stock orders."""
        self._require_login()
        return rh.orders.get_all_open_stock_orders()
        
    def get_all_orders(self) -> List[Dict[str, Any]]:
        """Get all stock orders (open, filled, cancelled)."""
        self._require_login()
        return rh.orders.get_all_stock_orders()
        
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get specific order by ID."""
        self._require_login()
        return rh.orders.get_stock_order_info(order_id)
        
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a specific order."""
        self._require_login()
        logger.info(f"Cancelling order: {order_id}")
        return rh.orders.cancel_stock_order(order_id)
        
    def cancel_all_orders(self) -> List[Dict[str, Any]]:
        """Cancel all open stock orders."""
        self._require_login()
        logger.warning("Cancelling all open orders!")
        return rh.orders.cancel_all_stock_orders()
        
    # ==================== Watchlist ====================
    
    def get_watchlist(self, name: str = "Default") -> List[Dict[str, Any]]:
        """Get watchlist by name."""
        self._require_login()
        return rh.account.get_watchlist_by_name(name)
        
    def add_to_watchlist(self, symbols: List[str], name: str = "Default") -> None:
        """Add symbols to watchlist."""
        self._require_login()
        rh.account.post_symbols_to_watchlist(symbols, name)
        
    def remove_from_watchlist(self, symbols: List[str], name: str = "Default") -> None:
        """Remove symbols from watchlist."""
        self._require_login()
        rh.account.delete_symbols_from_watchlist(symbols, name)
        
    # ==================== Dividends & Earnings ====================
    
    def get_dividends(self) -> List[Dict[str, Any]]:
        """Get dividend history."""
        self._require_login()
        return rh.account.get_dividends()
        
    def get_total_dividends(self) -> float:
        """Get total dividends received."""
        self._require_login()
        return rh.account.get_total_dividends()
        
    def get_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        """Get earnings data for a stock."""
        return rh.stocks.get_earnings(symbol)
        
    def get_ratings(self, symbol: str) -> Dict[str, Any]:
        """Get analyst ratings for a stock."""
        return rh.stocks.get_ratings(symbol)
        
    def get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news for a stock."""
        return rh.stocks.get_news(symbol)
