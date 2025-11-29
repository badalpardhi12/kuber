"""
Deep Reinforcement Learning Strategy using Proximal Policy Optimization (PPO)
Powered by Stable Baselines 3

This strategy uses a pre-trained PPO agent to generate trading signals.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from stable_baselines3 import PPO

from .base import BaseStrategy, Signal, SignalType

# Model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_trading_agent")

class PPOStrategy(BaseStrategy):
    """
    PPO Strategy that uses a pre-trained Reinforcement Learning agent.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, lookback: int = 30, **kwargs):
        super().__init__(name="PPO_RL", **kwargs)
        self.lookback = lookback
        self.model = None
        
        if os.path.exists(model_path + ".zip"):
            try:
                self.model = PPO.load(model_path)
                print(f"Loaded PPO model from {model_path}")
            except Exception as e:
                print(f"Failed to load PPO model: {e}")
        else:
            print(f"PPO model not found at {model_path}. Please run scripts/train_rl_agent.py first.")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate a trading signal using the PPO agent.
        """
        if self.model is None or len(df) < self.lookback:
            return Signal(self.name, SignalType.HOLD)
            
        # Construct observation
        # Must match the environment's observation space exactly
        # Assuming the env uses raw OHLCV + indicators
        # We need to ensure the columns match what the env expects
        
        # Take the last 'lookback' rows
        obs_df = df.iloc[-self.lookback:]
        
        # Convert to numpy array (ensure type is float32)
        obs = obs_df.values.astype(np.float32)
        
        # Predict action
        # deterministic=True for consistent evaluation
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Action is a continuous value between -1 and 1 (Position Size)
        # We convert this to a discrete signal for the engine
        position_size = action[0]
        
        signal_type = SignalType.HOLD
        strength = abs(position_size)
        
        if position_size > 0.2:
            signal_type = SignalType.BUY
        elif position_size < -0.2:
            signal_type = SignalType.SELL
            
        return Signal(
            symbol=df['symbol'].iloc[-1] if 'symbol' in df.columns else "UNKNOWN",
            signal_type=signal_type,
            strength=float(strength),
            price=df['close'].iloc[-1],
            strategy_name=self.name,
            reason=f"PPO Agent Action: {position_size:.4f}"
        )

# Legacy/Placeholder for LSTM if needed, or remove it.
# I'll keep a dummy class to avoid breaking imports if referenced elsewhere
class LSTMStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(name="LSTM_Legacy", **kwargs)
        
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        return Signal("LSTM", SignalType.HOLD)
