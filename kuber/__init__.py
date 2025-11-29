"""
Kuber - Algorithmic Trading Platform for Robinhood

A Python-based framework for implementing state-of-the-art trading algorithms
to maximize stock investment returns through the Robinhood platform.

Note: This platform is designed for passive investing strategies. As an H1B holder,
ensure your trading activities remain passive (automated, rules-based) and do not
constitute active income generation.
"""

__version__ = "0.1.0"
__author__ = "Kuber Team"

from kuber.core.engine import TradingEngine
from kuber.core.portfolio import Portfolio
from kuber.strategies.base import BaseStrategy

__all__ = [
    "TradingEngine",
    "Portfolio", 
    "BaseStrategy",
    "__version__",
]
