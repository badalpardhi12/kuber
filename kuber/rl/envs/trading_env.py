import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Dict

class TradingEnv(gym.Env):
    """
    A realistic trading environment for Reinforcement Learning.
    
    Attributes:
        df (pd.DataFrame): The market data (OHLCV + indicators).
        initial_balance (float): Starting capital.
        lookback_window (int): How many past bars the agent sees.
        commission (float): Transaction cost per trade (percentage).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, 
                 lookback_window: int = 30, commission: float = 0.001):
        super(TradingEnv, self).__init__()
        
        # Preserve original timestamps for alignment during evaluation/backtests
        self.timestamps = df.index.to_list()
        self.df = df.reset_index(drop=True)
        self.df.columns = [c.lower() for c in self.df.columns]  # Normalize columns to lowercase
        
        # Ensure we only have numeric data for observations
        # We keep 'close' for price simulation, but drop timestamps/objects
        self.df = self.df.select_dtypes(include=[np.number])
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.commission = commission
        
        # Action Space: Continuous value between -1 (Full Short) and 1 (Full Long)
        # 0 means no position (cash).
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space: 
        # [Open, High, Low, Close, Volume, ...Indicators] * lookback_window
        # We normalize these relative to the last close or similar to make them stationary.
        self.n_features = self.df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, self.n_features), 
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position size (in shares/contracts)
        self.equity = initial_balance
        self.history = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.equity = self.initial_balance
        self.history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Execute Action
        target_position_pct = float(action[0])
        current_price = self.df.iloc[self.current_step]['close']
        
        # Calculate target value
        target_value = self.equity * target_position_pct
        current_value = self.position * current_price
        
        # Calculate trade size
        trade_value = target_value - current_value
        trade_cost = abs(trade_value) * self.commission
        
        # Update portfolio
        # If we buy/sell, we pay commission from balance
        self.balance -= trade_cost
        
        # Update position (simplified: assuming instant execution at Close)
        # In reality, we'd use next Open, but for RL training on daily/minute data, Close is often used as proxy
        # or we shift data. Here we assume we trade at 'current_price'.
        if current_price > 0:
            self.position = target_value / current_price
        
        # 2. Step Time
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # 3. Calculate Reward
        # New equity
        next_price = self.df.iloc[self.current_step]['close']
        new_equity = self.balance + (self.position * next_price)
        
        # Reward = Log Return of Equity
        # SOTA: Use Differential Sharpe Ratio or simply Log Returns for PPO
        portfolio_return = (new_equity - self.equity) / self.equity if self.equity > 0 else 0.0
        reward = portfolio_return * 100  # Scale up for stability
        
        self.equity = new_equity
        
        # 4. Get Next Observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        # Return window of data
        # Normalize? For SOTA, we usually use returns or log-returns in the DF itself.
        # Here we assume DF is already pre-processed with stationary features.
        obs = self.df.iloc[self.current_step - self.lookback_window : self.current_step].values
        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        # Map current step back to original timestamp if available
        timestamp = None
        if self.timestamps:
            idx = max(min(self.current_step - 1, len(self.timestamps) - 1), 0)
            timestamp = self.timestamps[idx]
        
        return {
            "step": self.current_step,
            "equity": self.equity,
            "position": self.position,
            "timestamp": timestamp
        }

    def render(self):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}, Position: {self.position:.4f}")
