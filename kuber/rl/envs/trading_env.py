import os
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

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        lookback_window: int = 30,
        commission: float = 0.001,
        max_leverage: float = 1.0,
        random_reset: bool = False,
        episode_length: Optional[int] = None,
        reward_scaling: float = 100.0,
        position_change_penalty: float = 0.0,
        drawdown_penalty: float = 0.0,
    ):
        super(TradingEnv, self).__init__()
        
        # Preserve original timestamps for alignment during evaluation/backtests
        self.timestamps = df.index.to_list()
        self.df = df.reset_index(drop=True)
        self.df.columns = [c.lower() for c in self.df.columns]  # Normalize columns to lowercase
        
        # Ensure we only have numeric data for observations
        # We keep 'close' for price simulation, but drop timestamps/objects
        self.df = self.df.select_dtypes(include=[np.number])
        
        # SEPARATION OF CONCERNS (SOTA FIX):
        # The Agent should ONLY see stationary features (returns, indicators).
        # The Environment needs raw prices to calculate PnL.
        # We assume 'close' is in the dataframe.
        if 'close' not in self.df.columns:
            raise ValueError("Dataframe must contain 'close' column")
            
        self.prices = self.df['close'].values
        # Drop 'close' from observation features to ensure stationarity
        self.obs_df = self.df.drop(columns=['close'])
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.commission = commission
        self.max_leverage = max(1.0, float(max_leverage))
        self.random_reset = random_reset
        self.episode_length = episode_length
        self.reward_scaling = reward_scaling
        self.position_change_penalty = position_change_penalty
        self.drawdown_penalty = drawdown_penalty
        
        # SOTA: Transaction Costs (Critical for Realistic Trading)
        # Default: 0.1% per trade (10 basis points) - industry standard
        # This includes: commissions, bid-ask spread, and minimal slippage
        self.transaction_cost_pct = float(os.environ.get("KUBER_TRANSACTION_COST", 0.001))
        
        # SOTA: Risk Constraints (Prevent catastrophic losses)
        # Max Drawdown: Terminate if equity drops >10% from peak
        self.max_drawdown = float(os.environ.get("KUBER_MAX_DRAWDOWN", 0.10))
        # Stop Loss: Force close if single trade loses >2%
        self.stop_loss_pct = float(os.environ.get("KUBER_STOP_LOSS", 0.02))
        # Take Profit: Force close if single trade gains >4% (2:1 ratio)
        self.take_profit_pct = float(os.environ.get("KUBER_TAKE_PROFIT", 0.04))
        
        # Action Space: Discrete(3)
        # 0: Short (-1.0)
        # 1: Neutral (0.0)
        # 2: Long (1.0)
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: 
        # [Features] * lookback_window
        self.n_features = self.obs_df.shape[1]
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
        self.episode_steps = 0
        self.start_index = self.lookback_window
        
        # DSR State Variables (Recursive)
        self.returns_sum = 0.0  # A_t
        self.returns_sq_sum = 0.0  # B_t
        self.dsr_decay = 0.99  # Decay factor for online DSR (eta)
        
        self.peak_equity = initial_balance
        self.prev_position = 0.0
        
        # SOTA: Track trade entry for Stop Loss / Take Profit
        self.trade_entry_price = None
        self.trade_entry_equity = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if self.random_reset and len(self.df) > self.lookback_window + 2:
            max_start = len(self.df) - 2
            if self.episode_length is not None:
                max_start = max(self.lookback_window + 1, len(self.df) - self.episode_length - 1)
            self.start_index = int(self.np_random.integers(self.lookback_window, max_start))
            self.current_step = self.start_index
        else:
            self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.history = []
        self.episode_steps = 0
        self.prev_position = 0.0
        
        # Reset trade tracking
        self.trade_entry_price = None
        self.trade_entry_equity = None
        
        # Reset DSR stats
        self.returns_sum = 0.0
        self.returns_sq_sum = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Get Action (Target Position Direction: -1, 0, 1)
        if isinstance(action, np.ndarray):
            action = action.item()
        target_pos_dir = float(action - 1)  # 0->-1, 1->0, 2->1
        
        # 2. Calculate Transaction Cost
        # We rebalance from current position (shares) to target position (shares) at current price.
        prev_equity = max(self.equity, 1e-8)
        current_price = self.prices[self.current_step]
        
        # Calculate target shares
        # We allocate entire equity to the position (leverage=1.0 for now)
        # If Short, we sell equity worth of shares (negative shares).
        # If Neutral, 0 shares.
        target_shares = (prev_equity * target_pos_dir) / current_price
        
        # Calculate cost based on change in shares
        # Cost = Value of shares changed * cost_pct
        shares_change = abs(target_shares - self.position)
        value_changed = shares_change * current_price
        transaction_cost = value_changed * self.transaction_cost_pct
        
        # 3. Update Equity for Cost
        equity_after_cost = prev_equity - transaction_cost
        
        # 4. Advance Time
        self.current_step += 1
        self.episode_steps += 1
        
        # Check termination
        terminated = self.current_step >= len(self.df) - 1
        if self.episode_length is not None:
            terminated = terminated or (self.episode_steps >= self.episode_length)
        truncated = False
        
        if terminated:
            # If terminated, we don't calculate next PnL, just return current state
            # But we need to return a valid observation.
            # We can just return the last observation.
            # And reward for this step is just the cost incurred?
            # Or we can say the episode ends AFTER the price move?
            # Let's assume we hold for one last step if possible, but if we are at end of data, we can't.
            # If current_step >= len(df) - 1, we can't get next_price.
            pass

        # 5. Calculate PnL from Price Move
        if not terminated:
            next_price = self.prices[self.current_step]
            price_return = (next_price - current_price) / current_price
            
            # PnL = Value * Return
            # Value = TargetShares * CurrentPrice
            # PnL = TargetShares * CurrentPrice * PriceReturn
            #     = TargetShares * (NextPrice - CurrentPrice)
            pnl = target_shares * (next_price - current_price)
            
            new_equity = equity_after_cost + pnl
        else:
            new_equity = equity_after_cost
            next_price = current_price
            
        # Handle bankruptcy
        if new_equity <= 0:
            new_equity = 0.0
            terminated = True
            
        # SOTA: Risk Constraints
        # 1. Max Drawdown Kill Switch
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - new_equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                terminated = True
                reward = -100.0  # Large penalty for hitting max drawdown
                
        # 2. Track trade entry for Stop Loss / Take Profit
        # If we just opened a position (from neutral), record entry
        if self.prev_position == 0.0 and target_shares != 0.0:
            self.trade_entry_price = current_price
            self.trade_entry_equity = prev_equity
        
        # 3. Stop Loss / Take Profit
        if self.trade_entry_price is not None and target_shares != 0.0 and not terminated:
            # Calculate trade P&L since entry
            trade_pnl_pct = (new_equity - self.trade_entry_equity) / self.trade_entry_equity
            
            # Stop Loss: If losing >2%, force close
            if trade_pnl_pct < -self.stop_loss_pct:
                # Force close by setting target_shares to 0
                # But we can't change the action retroactively, so we'll terminate
                # and give a penalty
                terminated = True
                reward = -50.0  # Penalty for hitting stop loss
                
            # Take Profit: If gaining >4%, force close (with bonus)
            elif trade_pnl_pct > self.take_profit_pct:
                # Same as above, terminate with bonus
                terminated = True
                reward = 50.0  # Bonus for hitting take profit
        
        # If position closed (returned to neutral), reset tracking
        if target_shares == 0.0:
            self.trade_entry_price = None
            self.trade_entry_equity = None
            
        # 6. Calculate Reward
        # Return = (NewEquity - PrevEquity) / PrevEquity
        # This includes transaction cost and PnL
        pct_return = (new_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        
        # Simple PnL Reward: Return * 1000
        # e.g. 1% return -> 10 reward
        # e.g. -1% return -> -10 reward
        reward = pct_return * 1000.0
        
        # Penalties
        # We already deducted transaction cost from equity, so it's in the reward.
        # We can add extra penalty for drawdown if needed, but let's keep it simple first.
        # Just pure PnL.
        
        # Clip reward to reasonable range [-100, 100] (10% move)
        reward = np.clip(reward, -100.0, 100.0)
        
        # 7. Update State
        self.equity = new_equity
        self.position = target_shares
        self.prev_position = target_shares # Not strictly needed if we use self.position
        self.balance = new_equity - (self.position * next_price) # Cash component (approx)
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # 8. Get Observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        # Return window of data
        # Use obs_df which has NO raw prices, only stationary features
        obs = self.obs_df.iloc[self.current_step - self.lookback_window : self.current_step].values
        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        # Map current step back to original timestamp if available
        idx = max(self.current_step - 1, 0)
        timestamp = None
        if self.timestamps:
            idx = min(idx, len(self.timestamps) - 1)
            timestamp = self.timestamps[idx]

        close_price = None
        if 0 <= idx < len(self.prices):
            close_price = float(self.prices[idx])
            
        exposure_pct = 0.0
        if self.equity > 0 and close_price is not None:
            exposure_pct = (self.position * close_price) / self.equity
        return {
            "step": self.current_step,
            "equity": self.equity,
            "position": self.position,
            "cash": self.balance,
            "price": close_price,
            "exposure_pct": exposure_pct,
            "timestamp": timestamp
        }

    def render(self):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}, Position: {self.position:.4f}")
