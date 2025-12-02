import os
import sys
import time
import pandas as pd
import numpy as np
import vectorbt as vbt
import ta
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuber.rl.envs import TradingEnv
from kuber.rl.policies import TransformerFeatureExtractor
from kuber.data.providers import download_polygon_intraday_history, has_polygon_api_key
from kuber.data.features import add_technical_indicators

# Configuration
TICKERS = os.environ.get("KUBER_TICKERS", "SPY,QQQ,IWM,NVDA,AAPL,TSLA,AMD,MSFT,AMZN").split(",")
TEST_WINDOW_DAYS = 60
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

console = Console()

def load_test_data(tickers: list) -> dict:
    """Load test data for all tickers."""
    now = pd.Timestamp.utcnow()
    start = now - pd.Timedelta(days=TEST_WINDOW_DAYS)
    
    data = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Fetching data...", total=len(tickers))
        
        # Try Polygon first
        if has_polygon_api_key():
            poly_data = download_polygon_intraday_history(tickers, start, now, interval_minutes=5)
            for ticker, df in poly_data.items():
                if not df.empty:
                    # Ensure datetime is the index
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df = df.set_index('datetime')
                    # Ensure index is DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    #Drop any extra columns
                    cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
                    df = df[[c for c in cols_to_keep if c in df.columns]]
                    data[ticker] = df
                    progress.advance(task)
        
        # Fill missing with yfinance
        remaining = [t for t in tickers if t not in data]
        if remaining:
            import yfinance as yf
            for ticker in remaining:
                df = yf.download(ticker, start=start, end=now, interval="1h", progress=False)
                if not df.empty:
                    # Standardize columns
                    df.columns = [c.lower() for c in df.columns]
                    # Ensure datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    data[ticker] = df
                progress.advance(task)
                
    return data

def run_ppo_inference(ticker: str, df: pd.DataFrame) -> pd.Series:
    """Run PPO agent inference on the dataframe."""
    model_path = os.path.join(MODELS_DIR, f"ppo_agent_{ticker}.zip")
    vec_norm_path = os.path.join(MODELS_DIR, f"vec_normalize_{ticker}.pkl")
    
    if not os.path.exists(model_path):
        return pd.Series(0, index=df.index)
        
    # Prepare data - ensure column names are lowercase
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure we have all required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        console.print(f"[yellow]Warning: {ticker} missing columns {missing}, skipping[/yellow]")
        return pd.Series(0, index=df.index)
    
    # Add technical indicators (this must match training!)
    df = add_technical_indicators(df)
    
    # Load env stats
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    # We need a dummy env to load the stats
    env = TradingEnv(df)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False
    
    # Load model
    model = PPO.load(model_path, env=env)
    
    # Run inference
    obs = env.reset()
    actions = []
    
    # We can't easily use a progress bar here inside the function without passing it in,
    # but inference is usually fast enough.
    for _ in range(len(df)):
        action, _ = model.predict(obs, deterministic=True)
        # Handle both Continuous and Discrete actions
        if isinstance(env.action_space, gym.spaces.Discrete):
            # Discrete: 0->-1, 1->0, 2->1
            # action is array([int])
            discrete_act = action[0]
            pos = float(discrete_act - 1)
            actions.append(pos)
        else:
            # Continuous: action is array([[float]])
            actions.append(action[0][0])
            
        obs, _, done, _ = env.step(action)
        if done:
            break
            
    # Pad if needed (should match length)
    if len(actions) < len(df):
        actions.extend([0.0] * (len(df) - len(actions)))
        
    return pd.Series(actions, index=df.index)

def main():
    console.print("[bold green]Starting Kuber SOTA Benchmark[/bold green]")
    
    # 1. Load Data
    data_map = load_test_data(TICKERS)
    
    # 2. Run Strategies
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Running Inference...", total=len(TICKERS))
        
        for ticker in TICKERS:
            if ticker not in data_map:
                progress.advance(task)
                continue
                
            df = data_map[ticker]
            
            # PPO
            ppo_signals = run_ppo_inference(ticker, df)
            
            # SOTA: Calculate PnL with Transaction Costs
            price = df['close']
            returns = price.pct_change().shift(-1).fillna(0)
            
            # Transaction costs: 0.1% per trade (charged on position changes)
            TRANSACTION_COST_PCT = 0.001
            position_changes = ppo_signals.diff().abs().fillna(0)
            transaction_costs = position_changes * TRANSACTION_COST_PCT
            
            # Net returns = position * market_return - transaction_costs
            ppo_returns = (ppo_signals * returns) - transaction_costs
            
            total_return = ppo_returns.sum()
            sharpe = 0.0
            if ppo_returns.std() > 0:
                sharpe = ppo_returns.mean() / ppo_returns.std() * np.sqrt(252 * 78) # Annualized (5min bars)
            win_rate = (ppo_returns > 0).mean()
            
            results.append({
                "Ticker": ticker,
                "Return": total_return,
                "Sharpe": sharpe,
                "Success Rate": win_rate
            })
            
            progress.advance(task)



    # 3. Display Results
    table = Table(title="PPO Per-Ticker Performance")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Return", justify="right", style="magenta")
    table.add_column("Sharpe", justify="right", style="green")
    table.add_column("Success Rate", justify="right", style="yellow")
    
    for res in results:
        table.add_row(
            res["Ticker"],
            f"{res['Return']:.2%}",
            f"{res['Sharpe']:.2f}",
            f"{res['Success Rate']:.2%}"
        )
        
    console.print(table)

if __name__ == "__main__":
    main()
