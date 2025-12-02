import os
import sys
import time
import multiprocessing
import shutil
from datetime import datetime, timedelta
from functools import partial
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.utils import set_random_seed

# SOTA: RecurrentPPO for LSTM-based temporal modeling
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False
    print("Warning: sb3-contrib not installed. RecurrentPPO unavailable.")

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from kuber.rl.envs import TradingEnv
from kuber.rl.policies import TransformerFeatureExtractor
from kuber.data.providers import download_polygon_intraday_history, has_polygon_api_key
from kuber.data.features import add_technical_indicators

# Configuration
TICKERS = os.environ.get("KUBER_TICKERS", "SPY,QQQ,IWM,NVDA,AAPL,TSLA,AMD,MSFT,AMZN").split(",")
TIMESTEPS = 100_000  # Increased for SOTA convergence
LOOKBACK_WINDOW = 30
TEST_WINDOW_DAYS = 60
N_CPU = multiprocessing.cpu_count()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TB_LOG_DIR = os.path.join(BASE_DIR, "tensorboard")
EPISODE_STEPS = int(os.environ.get("KUBER_PPO_EPISODE_STEPS", 78 * 60))  # ~60 trading days
GLOBAL_SEED = int(os.environ.get("KUBER_PPO_SEED", 42))
# Safe Parallelism: 4 envs per ticker is a good balance for M4 Max
N_ENVS = max(1, int(os.environ.get("KUBER_PPO_ENVS", 4)))
REWARD_SCALING = float(os.environ.get("KUBER_REWARD_SCALING", 100.0))
POSITION_CHANGE_PENALTY = float(os.environ.get("KUBER_POSITION_PENALTY", 0.0005))
DRAWDOWN_PENALTY = float(os.environ.get("KUBER_DRAWDOWN_PENALTY", 0.25))
MAX_LEVERAGE = float(os.environ.get("KUBER_MAX_LEVERAGE", 1.0))

# SOTA: LSTM Architecture Option
USE_LSTM = os.environ.get("KUBER_USE_LSTM", "0") == "1"
if USE_LSTM and not HAS_RECURRENT_PPO:
    print("ERROR: KUBER_USE_LSTM=1 but sb3-contrib not installed!")
    print("Install with: pip install sb3-contrib")
    sys.exit(1)

ENABLE_TENSORBOARD = os.environ.get("KUBER_ENABLE_TENSORBOARD", "1") == "1"
if ENABLE_TENSORBOARD:
    try:
        import tensorboard  # type: ignore  # noqa: F401
    except ImportError:
        ENABLE_TENSORBOARD = False

# Allow quick overrides for experimentation
TIMESTEPS = int(os.environ.get("KUBER_PPO_TIMESTEPS", TIMESTEPS))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)
set_random_seed(GLOBAL_SEED)
REFERENCE_NOW = datetime.utcnow()


class RichReportingCallback(BaseCallback):
    """Callback that sends progress updates to a multiprocessing queue."""
    
    def __init__(self, queue: multiprocessing.Queue, ticker: str, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.queue = queue
        self.ticker = ticker
        self.total_timesteps = total_timesteps
        self.last_update = 0

    def _on_step(self) -> bool:
        # Update every 100 steps to reduce overhead
        if self.num_timesteps - self.last_update >= 100 or self.num_timesteps >= self.total_timesteps:
            try:
                # Send (ticker, current_step, total_steps, info_dict)
                # We can extract some info like mean reward if available, but for now just progress
                self.queue.put(("progress", self.ticker, self.num_timesteps, self.total_timesteps))
                self.last_update = self.num_timesteps
            except Exception:
                pass # Queue might be closed
        return True
        
    def _on_training_end(self) -> None:
        try:
            self.queue.put(("done", self.ticker, self.total_timesteps, self.total_timesteps))
        except Exception:
            pass


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
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif 'date' in df.columns:
        df = df.set_index('date')
    df = df.sort_index()
    return df


def sanitize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure datetime index is clean, timezone-naive, and duplicate-free."""

    if df.empty:
        return df

    clean_df = df.copy()
    clean_df.index = pd.to_datetime(clean_df.index, utc=True)
    clean_df.index = clean_df.index.tz_convert(None)
    clean_df = clean_df.sort_index()
    clean_df = clean_df[~clean_df.index.duplicated(keep="last")]
    return clean_df


def align_n_steps(desired_steps: int, n_envs: int) -> int:
    if desired_steps % n_envs == 0:
        return desired_steps
    return ((desired_steps // n_envs) + 1) * n_envs


TRAIN_N_STEPS = align_n_steps(2048, N_ENVS)


def ticker_seed(ticker: str) -> int:
    base = sum(ord(ch) for ch in ticker.upper())
    return (GLOBAL_SEED + base) % (2**31 - 1)


def _make_env(dataframe: pd.DataFrame, env_seed: int, **env_kwargs):
    def _init():
        env = TradingEnv(dataframe, **env_kwargs)
        env.reset(seed=env_seed)
        return env

    return _init


def build_vec_env(dataframe: pd.DataFrame, n_envs: int, base_seed: int, **env_kwargs):
    env_fns = []
    for idx in range(n_envs):
        env_seed = base_seed + (idx * 7919)
        env_fns.append(_make_env(dataframe, env_seed=env_seed, **env_kwargs))

    if n_envs > 1:
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)


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


def train_ticker_agent(ticker: str, queue: multiprocessing.Queue, reference_now: Optional[datetime] = None):
    """
    Trains a PPO agent for a specific ticker.
    """
    try:
        queue.put(("status", ticker, "📥 Fetching data..."))
        base_seed = ticker_seed(ticker)
        set_random_seed(base_seed)
        now = reference_now or datetime.utcnow()
        eval_start = now - timedelta(days=TEST_WINDOW_DAYS)
        train_start = now - timedelta(days=365 * 2)
        train_end = eval_start

        # Optimization: Load FULL history to hit the pre-fetched cache
        full_df = load_market_data(ticker, train_start, now)
        
        if full_df.empty:
            queue.put(("error", ticker, "No data found"))
            return

        full_df = sanitize_price_frame(full_df)
        
        # Slice for Training
        df = full_df.loc[(full_df.index >= train_start) & (full_df.index < eval_start)].copy()
        if df.empty:
            queue.put(("error", ticker, "No training samples"))
            return

        df = add_technical_indicators(df)
        queue.put(("status", ticker, f"📊 Data: {len(df)} bars"))

        # Slice for Evaluation
        eval_df = full_df.loc[(full_df.index >= eval_start) & (full_df.index <= now)].copy()
        eval_env = None
        if not eval_df.empty:
            eval_df = add_technical_indicators(eval_df)
            
        # Create Environment(s)
        env = build_vec_env(
            df,
            n_envs=N_ENVS,
            base_seed=base_seed,
            lookback_window=LOOKBACK_WINDOW,
            random_reset=True,
            episode_length=EPISODE_STEPS,
            position_change_penalty=POSITION_CHANGE_PENALTY,
            drawdown_penalty=DRAWDOWN_PENALTY,
            reward_scaling=REWARD_SCALING,
            max_leverage=MAX_LEVERAGE,
        )
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        if not eval_df.empty:
            eval_env = build_vec_env(
                eval_df,
                n_envs=1,
                base_seed=base_seed + 1337,
                lookback_window=LOOKBACK_WINDOW,
                random_reset=False,
                episode_length=None,
                position_change_penalty=POSITION_CHANGE_PENALTY,
                drawdown_penalty=DRAWDOWN_PENALTY,
                reward_scaling=REWARD_SCALING,
                max_leverage=MAX_LEVERAGE,
            )
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
        
        device = "mps" if torch.backends.mps.is_available() else "auto"
        queue.put(("status", ticker, f"🧠 Training ({device})"))
        
        # SOTA: Choose between PPO (MLP) and RecurrentPPO (LSTM)
        if USE_LSTM:
            # RecurrentPPO with LSTM for temporal modeling
            lstm_policy_kwargs = dict(
                lstm_hidden_size=256,  # Hidden units in LSTM
                n_lstm_layers=2,  # 2 LSTM layers
                enable_critic_lstm=True,  # Use LSTM for both actor and critic
                shared_lstm=False,  # Separate LSTMs for actor/critic
            )
            
            model = RecurrentPPO(
                "MlpLstmPolicy",
                env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,  # Standard batch size for stability
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=linear_schedule(0.2, 0.05),
                clip_range_vf=linear_schedule(0.2, 0.05),
                ent_coef=0.05,  # Increased to force exploration away from Cash
                vf_coef=0.5,
                n_epochs=10,
                max_grad_norm=0.5,
                target_kl=0.02,
                tensorboard_log=os.path.join(TB_LOG_DIR, ticker) if ENABLE_TENSORBOARD else None,
                policy_kwargs=lstm_policy_kwargs,
                device=device,
                seed=base_seed,
            )
        else:
            # Standard PPO with MLP
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,  # Standard batch size for stability
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=linear_schedule(0.2, 0.05),
                clip_range_vf=linear_schedule(0.2, 0.05),
                ent_coef=0.05,  # Increased to force exploration away from Cash
                vf_coef=0.5,
                n_epochs=10,
                max_grad_norm=0.5,
                target_kl=0.02,
                tensorboard_log=os.path.join(TB_LOG_DIR, ticker) if ENABLE_TENSORBOARD else None,
                policy_kwargs=policy_kwargs,
                device=device,
                seed=base_seed,
            )
        
        
        callbacks: List[BaseCallback] = []
        
        # Add Rich Reporting Callback
        callbacks.append(RichReportingCallback(queue, ticker, TIMESTEPS))

        if eval_env is not None:
            best_model_dir = os.path.join(MODELS_DIR, "best_models", ticker)
            eval_log_dir = os.path.join(MODELS_DIR, "eval_logs", ticker)
            os.makedirs(best_model_dir, exist_ok=True)
            os.makedirs(eval_log_dir, exist_ok=True)

            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5,
                min_evals=2,
                verbose=0
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

        start_time = time.time()
        model.learn(total_timesteps=TIMESTEPS, callback=callback_list, progress_bar=False)
        duration = time.time() - start_time
        
        # Save Model
        model_path = os.path.join(MODELS_DIR, f"ppo_agent_{ticker}")
        model.save(model_path)

        best_model_zip = os.path.join(MODELS_DIR, "best_models", ticker, "best_model.zip")
        if os.path.exists(best_model_zip):
            shutil.copy(best_model_zip, f"{model_path}.zip")
        
        env.save(os.path.join(MODELS_DIR, f"vec_normalize_{ticker}.pkl"))
        env.close()
        if eval_env is not None:
            eval_env.close()
        
        queue.put(("done", ticker, TIMESTEPS, duration))
        
    except Exception as e:
        queue.put(("error", ticker, str(e)))
        import traceback
        traceback.print_exc()


def prefetch_polygon_history(reference_now: datetime):
    """Warm the Polygon cache sequentially to avoid rate-limit storms during training."""
    if not has_polygon_api_key():
        print("⚠️ Polygon API key missing; skipping prefetch step.")
        return

    prefetch_start = reference_now - timedelta(days=365 * 2)
    print(
        f"📡 Prefetching Polygon history ({prefetch_start.date()} → {reference_now.date()}) for {len(TICKERS)} tickers..."
    )
    download_polygon_intraday_history(
        TICKERS,
        start=prefetch_start,
        end=reference_now,
        interval_minutes=5,
    )

def main():
    console = Console()
    console.print(f"[bold green]🚀 Starting Parallel RL Training on {len(TICKERS)} tickers using {N_CPU} cores[/bold green]")
    
    if torch.backends.mps.is_available():
        console.print("[bold cyan]🍏 MPS (Metal) is available! Using GPU acceleration.[/bold cyan]")
    else:
        console.print("[yellow]⚠️ MPS not detected. Using CPU.[/yellow]")

    console.print(f"🧱 Vec envs per ticker: {N_ENVS}")
    console.print("=" * 60)

    reference_now = REFERENCE_NOW
    prefetch_polygon_history(reference_now)

    # Setup Rich Progress
    progress = Progress(
        TextColumn("[bold blue]{task.fields[ticker]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        TextColumn("[dim]{task.fields[status]}"),
    )

    task_ids = {}
    for ticker in TICKERS:
        task_id = progress.add_task(f"Waiting...", total=TIMESTEPS, ticker=ticker, status="Pending")
        task_ids[ticker] = task_id

    # Manager for Queue
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    # Launch Processes
    # If N_ENVS > 1, we use SubprocVecEnv which creates its own processes.
    # Nesting multiprocessing (Pool -> SubprocVecEnv) causes "daemonic processes" error.
    # So if N_ENVS > 1, we run tickers sequentially (which is fine since each ticker uses all cores).
    
    if N_ENVS > 1:
        # Sequential Execution for High Parallelism per Ticker
        console.print(f"[yellow]Running sequentially because N_ENVS={N_ENVS} (uses all cores per ticker)[/yellow]")
        
        # We need to run this in a way that updates the UI
        # We can't block the UI loop.
        # So we'll use a single separate process for the worker to keep UI responsive,
        # OR just run in loop and update queue? 
        # Actually, simply running them one by one in the main thread might block the UI update loop
        # unless we thread the UI or thread the training.
        
        # Better approach: Use a ThreadPool instead of ProcessPool for the outer loop
        # Threads can spawn processes!
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(processes=1) # Run 1 ticker at a time to maximize resources
        
        for ticker in TICKERS:
            pool.apply_async(train_ticker_agent, args=(ticker, queue, reference_now))
            
        pool.close()
        
    else:
        # Parallel Execution for Single Env per Ticker
        # We use apply_async to launch them and then monitor the queue
        pool = multiprocessing.Pool(processes=min(len(TICKERS), N_CPU))
        
        for ticker in TICKERS:
            pool.apply_async(train_ticker_agent, args=(ticker, queue, reference_now))
        
        pool.close()
    
    # Monitor Loop
    completed_count = 0
    with Live(Panel(progress, title="Kuber RL Training", border_style="green"), refresh_per_second=10) as live:
        while completed_count < len(TICKERS):
            while not queue.empty():
                try:
                    msg = queue.get_nowait()
                    msg_type = msg[0]
                    ticker = msg[1]
                    task_id = task_ids[ticker]
                    
                    if msg_type == "progress":
                        current, total = msg[2], msg[3]
                        progress.update(task_id, completed=current, total=total, status="Training")
                    elif msg_type == "status":
                        status_text = msg[2]
                        progress.update(task_id, status=status_text)
                    elif msg_type == "done":
                        # Ensure 100%
                        progress.update(task_id, completed=TIMESTEPS, status="[bold green]Done[/bold green]")
                        completed_count += 1
                    elif msg_type == "error":
                        error_msg = msg[2]
                        progress.update(task_id, status=f"[bold red]Error: {error_msg}[/bold red]")
                        completed_count += 1 # Mark as done to avoid infinite loop
                        
                except Exception:
                    break
            
            time.sleep(0.1)
            
    pool.join()
    console.print("[bold green]🎉 All agents trained successfully![/bold green]")

if __name__ == "__main__":
    # Fix for macOS multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
