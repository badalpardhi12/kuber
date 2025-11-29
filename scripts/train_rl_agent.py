import os
import sys
import time
import multiprocessing
import shutil
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CallbackList,
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuber.rl.envs import TradingEnv
from kuber.rl.policies import TransformerFeatureExtractor
from kuber.data.providers import download_polygon_intraday_history, has_polygon_api_key
from kuber.data.features import add_technical_indicators

# Configuration
TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "TSLA", "AMD", "MSFT", "AMZN"]
TIMESTEPS = 1_000_000  # Increased for SOTA convergence
LOOKBACK_WINDOW = 30
TEST_WINDOW_DAYS = 60
N_CPU = multiprocessing.cpu_count()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TB_LOG_DIR = os.path.join(BASE_DIR, "tensorboard")
ENABLE_TENSORBOARD = True
if ENABLE_TENSORBOARD:
    try:
        import tensorboard  # type: ignore  # noqa: F401
    except ImportError:
        ENABLE_TENSORBOARD = False

# Allow quick overrides for experimentation
TIMESTEPS = int(os.environ.get("KUBER_PPO_TIMESTEPS", TIMESTEPS))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)


def linear_schedule(initial_value: float, final_value: float):
    """Linear schedule used by PPO for learning rate/clip range."""

    if initial_value == final_value:
        return initial_value

    def schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule


def load_market_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch intraday market data with Polygon preferred."""

    df = pd.DataFrame()

    if has_polygon_api_key():
        datasets = download_polygon_intraday_history([ticker], start, end, interval_minutes=5)
        df = datasets.get(ticker, pd.DataFrame())

    if df.empty:
        df = yf.download(ticker, start=start, end=end, interval="1h", progress=False)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    return df


class NormalizedEvalCallback(EvalCallback):
    """Keeps VecNormalize statistics in sync between train and eval envs."""

    def __init__(self, train_env: VecNormalize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_env = train_env

    def _on_step(self) -> bool:
        if isinstance(self.eval_env, VecNormalize) and isinstance(self.train_env, VecNormalize):
            self.eval_env.obs_rms = self.train_env.obs_rms.copy()
            self.eval_env.ret_rms = self.train_env.ret_rms.copy()
        return super()._on_step()

# Removed local add_technical_indicators definition as it is imported now


def train_ticker_agent(ticker: str):
    """
    Trains a PPO agent for a specific ticker.
    """
    print(f"[{ticker}] 📥 Fetching data...")
    train_start = datetime.now() - timedelta(days=365 * 2)
    train_end = datetime.now() - timedelta(days=TEST_WINDOW_DAYS)

    try:
        df = load_market_data(ticker, train_start, train_end)
        if df.empty:
            print(f"[{ticker}] ❌ No training data found.")
            return

        df = add_technical_indicators(df)
        print(f"[{ticker}] 📊 Training data shape after engineering: {df.shape}")

        eval_df = load_market_data(ticker, datetime.now() - timedelta(days=TEST_WINDOW_DAYS), datetime.now())
        if eval_df.empty:
            print(f"[{ticker}] ⚠️ No evaluation data. Eval callback disabled.")
        else:
            eval_df = add_technical_indicators(eval_df)
            print(f"[{ticker}] 📈 Evaluation window shape: {eval_df.shape}")

        # Create Environment
        env = DummyVecEnv([lambda: TradingEnv(df, lookback_window=LOOKBACK_WINDOW)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        eval_env = None
        if not eval_df.empty:
            eval_env = DummyVecEnv([lambda: TradingEnv(eval_df, lookback_window=LOOKBACK_WINDOW)])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
            eval_env.training = False

        # Define Model
        policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(
                lookback_window=LOOKBACK_WINDOW,
                d_model=128,
                nhead=8,
                num_layers=3,
                features_dim=256
            ),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=nn.SiLU,
            ortho_init=False,
        )
        
        # Determine device
        device = "mps" if torch.backends.mps.is_available() else "auto"
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=0, 
            learning_rate=linear_schedule(3e-4, 5e-5),
            n_steps=4096, 
            batch_size=512,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(0.2, 0.05),
            clip_range_vf=linear_schedule(0.2, 0.05),
            ent_coef=0.005,
            vf_coef=0.7,
            n_epochs=20,
            max_grad_norm=0.5,
            target_kl=0.02,
            tensorboard_log=os.path.join(TB_LOG_DIR, ticker) if ENABLE_TENSORBOARD else None,
            policy_kwargs=policy_kwargs,
            device=device
        )
        
        callbacks: List[EvalCallback] = []
        if eval_env is not None:
            best_model_dir = os.path.join(MODELS_DIR, "best_models", ticker)
            eval_log_dir = os.path.join(MODELS_DIR, "eval_logs", ticker)
            os.makedirs(best_model_dir, exist_ok=True)
            os.makedirs(eval_log_dir, exist_ok=True)

            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5,
                min_evals=2,
                verbose=1
            )

            eval_callback = NormalizedEvalCallback(
                train_env=env,
                eval_env=eval_env,
                best_model_save_path=best_model_dir,
                log_path=eval_log_dir,
                eval_freq=max(10_000, TIMESTEPS // 20),
                deterministic=True,
                render=False,
                callback_on_new_best=stop_callback,
            )

            callbacks.append(eval_callback)

        callback_list = CallbackList(callbacks) if callbacks else None

        print(f"[{ticker}] 🖥️  Model Device: {model.device}")

        print(f"[{ticker}] 🧠 Training PPO Agent ({TIMESTEPS} steps)...")
        start_time = time.time()
        model.learn(total_timesteps=TIMESTEPS, callback=callback_list, progress_bar=True)
        duration = time.time() - start_time
        
        # Save Model
        model_path = os.path.join(MODELS_DIR, f"ppo_agent_{ticker}")
        model.save(model_path)

        best_model_zip = os.path.join(MODELS_DIR, "best_models", ticker, "best_model.zip")
        if os.path.exists(best_model_zip):
            shutil.copy(best_model_zip, f"{model_path}.zip")
        
        # Save normalization stats so we can use them during inference/backtest
        env.save(os.path.join(MODELS_DIR, f"vec_normalize_{ticker}.pkl"))
        
        print(f"[{ticker}] ✅ Saved model to {model_path} (Time: {duration:.1f}s)")
        
    except Exception as e:
        print(f"[{ticker}] ❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"🚀 Starting Parallel RL Training on {len(TICKERS)} tickers using {N_CPU} cores")
    
    if torch.backends.mps.is_available():
        print("🍏 MPS (Metal) is available! Using GPU acceleration.")
    else:
        print("⚠️ MPS not detected. Using CPU.")

    if not ENABLE_TENSORBOARD:
        print("ℹ️ TensorBoard logging disabled. Set KUBER_ENABLE_TENSORBOARD=1 after installing tensorboard to enable.")
        
    print("=" * 60)
    
    # Use multiprocessing to train agents in parallel
    # We limit processes to avoid OOM or rate limits
    max_processes = min(N_CPU, len(TICKERS))
    
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.map(train_ticker_agent, TICKERS)
        
    print("=" * 60)
    print("🎉 All agents trained successfully!")

if __name__ == "__main__":
    # Fix for macOS multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
