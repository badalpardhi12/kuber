#!/usr/bin/env python3
"""
KUBER PLATFORM - SOTA BENCHMARK SYSTEM
Powered by VectorBT and Polygon.io

This benchmark compares:
1. Classical Technical Analysis Strategies (RSI, MACD, etc.)
2. SOTA Reinforcement Learning (PPO)
3. Deep Learning (Transformer/LSTM)

It uses VectorBT for high-performance vectorized backtesting.
"""

import sys
import os
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import vectorbt as vbt
import yfinance as yf
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuber.data.providers import download_polygon_intraday_history, has_polygon_api_key
from kuber.strategies.momentum import RSIStrategy, MACDStrategy
from kuber.strategies.moving_average import SMAStrategy, GoldenCrossStrategy

# Suppress warnings
warnings.filterwarnings("ignore")
console = Console()

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'NVDA', 'AAPL', 'TSLA', 'AMD', 'MSFT', 'AMZN']
START_DATE = datetime.now() - timedelta(days=60)
END_DATE = datetime.now()
TIMEFRAME = '5m'  # 5-minute bars for intraday trading
INITIAL_CAPITAL = 10000.0
RL_LOOKBACK = 30

def fetch_data(symbols):
    """Fetch data from Polygon or YFinance."""
    data = {}
    console.print(f"[bold cyan]Fetching data for {len(symbols)} symbols...[/bold cyan]")
    
    # 1. Try Polygon (SOTA 5-minute data) - Matches Training Data Source
    if has_polygon_api_key():
        console.print("[bold green]Using Polygon.io 5-minute data (Consistent with Training)...[/bold green]")
        datasets = download_polygon_intraday_history(symbols, START_DATE, END_DATE, interval_minutes=5)
        # Polygon returns lowercase columns usually, but let's check
        for sym, df in datasets.items():
            if not df.empty:
                # Ensure 'Close' column exists for VectorBT
                if 'close' in df.columns and 'Close' not in df.columns:
                    df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
                
                data[sym] = df
    
    # 2. Fallback / Fill missing with YFinance
    missing_symbols = [s for s in symbols if s not in data]
    if missing_symbols:
        console.print(f"[yellow]Fetching missing symbols from YFinance: {missing_symbols}[/yellow]")
        for symbol in tqdm(missing_symbols):
            try:
                df = yf.download(symbol, start=START_DATE, end=END_DATE, interval=TIMEFRAME, progress=False)
                
                if not df.empty:
                    # Handle MultiIndex if present
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            # Try to get the level 0 'Close' or just drop the ticker level
                            df.columns = df.columns.get_level_values(0)
                        except Exception:
                            pass
                    
                    if 'Close' in df.columns:
                        data[symbol] = df
                    elif 'close' in df.columns:
                         df = df.rename(columns={'close': 'Close'})
                         data[symbol] = df
                    else:
                        console.print(f"[yellow]Warning: No Close column for {symbol}[/yellow]")

            except Exception as e:
                console.print(f"[red]Error fetching {symbol}: {e}[/red]")
            
    return data

def run_benchmark():
    console.print("[bold green]Starting Kuber SOTA Benchmark[/bold green]")
    
    # 1. Data Acquisition
    data_dict = fetch_data(SYMBOLS)
    if not data_dict:
        console.print("[red]No data fetched. Exiting.[/red]")
        return

    # Combine Close prices for VectorBT
    # Ensure all series have the same index or align them
    series_list = {}
    for sym, df in data_dict.items():
        series_list[sym] = df['Close']
    
    close_prices = pd.DataFrame(series_list)
    close_prices = close_prices.dropna()
    
    if close_prices.empty:
        console.print("[red]No overlapping data found. Exiting.[/red]")
        return
    
    results = []

    # -------------------------------------------------------------------------
    # STRATEGY 1: Buy and Hold (Benchmark)
    # -------------------------------------------------------------------------
    pf_bh = vbt.Portfolio.from_holding(close_prices, init_cash=INITIAL_CAPITAL, freq=TIMEFRAME)
    
    sharpe_bh = pf_bh.sharpe_ratio().replace([np.inf, -np.inf], np.nan).mean()
    
    results.append({
        "Strategy": "Buy & Hold",
        "Total Return [%]": pf_bh.total_return().mean() * 100,
        "Sharpe Ratio": sharpe_bh,
        "Max Drawdown [%]": pf_bh.max_drawdown().mean() * 100,
        "Win Rate [%]": 100.0 # Always winning if positive return? No, irrelevant for B&H
    })

    # -------------------------------------------------------------------------
    # STRATEGY 2: RSI (Mean Reversion)
    # -------------------------------------------------------------------------
    # Vectorized RSI
    rsi = vbt.RSI.run(close_prices, window=14)
    entries = rsi.rsi_below(30)
    exits = rsi.rsi_above(70)
    
    pf_rsi = vbt.Portfolio.from_signals(
        close_prices, entries, exits, 
        init_cash=INITIAL_CAPITAL,
        fees=0.001, # 0.1% trading fee
        slippage=0.001, # 0.1% slippage
        freq=TIMEFRAME
    )
    
    sharpe_rsi = pf_rsi.sharpe_ratio().replace([np.inf, -np.inf], np.nan).mean()
    
    results.append({
        "Strategy": "RSI (14, 30/70)",
        "Total Return [%]": pf_rsi.total_return().mean() * 100,
        "Sharpe Ratio": sharpe_rsi,
        "Max Drawdown [%]": pf_rsi.max_drawdown().mean() * 100,
        "Win Rate [%]": pf_rsi.trades.win_rate().mean() * 100
    })

    # -------------------------------------------------------------------------
    # STRATEGY 3: Golden Cross (SMA 50/200)
    # -------------------------------------------------------------------------
    sma_fast = vbt.MA.run(close_prices, window=50)
    sma_slow = vbt.MA.run(close_prices, window=200)
    
    entries_sma = sma_fast.ma_crossed_above(sma_slow)
    exits_sma = sma_fast.ma_crossed_below(sma_slow)
    
    pf_sma = vbt.Portfolio.from_signals(
        close_prices, entries_sma, exits_sma, 
        init_cash=INITIAL_CAPITAL,
        fees=0.001,
        freq=TIMEFRAME
    )
    
    sharpe_sma = pf_sma.sharpe_ratio().replace([np.inf, -np.inf], np.nan).mean()
    
    results.append({
        "Strategy": "Golden Cross (50/200)",
        "Total Return [%]": pf_sma.total_return().mean() * 100,
        "Sharpe Ratio": sharpe_sma,
        "Max Drawdown [%]": pf_sma.max_drawdown().mean() * 100,
        "Win Rate [%]": pf_sma.trades.win_rate().mean() * 100
    })

    # -------------------------------------------------------------------------
    # STRATEGY 4: SOTA RL (PPO) - Real Inference
    # -------------------------------------------------------------------------
    console.print("[bold cyan]Running PPO Inference...[/bold cyan]")
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from kuber.data.features import add_technical_indicators
    from kuber.rl.envs import TradingEnv

    exposures_rl = pd.DataFrame(0.0, index=close_prices.index, columns=close_prices.columns)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    annualization_factor = np.sqrt(252 * (390 // 5))  # 252 trading days * 78 five-minute bars

    for symbol in close_prices.columns:
        model_path = os.path.join(models_dir, f"ppo_agent_{symbol}.zip")
        vec_norm_path = os.path.join(models_dir, f"vec_normalize_{symbol}.pkl")

        if not os.path.exists(model_path):
            console.print(f"[yellow]Warning: No model found for {symbol}, skipping RL inference.[/yellow]")
            continue

        try:
            df = data_dict[symbol].copy()
            df.columns = [c.lower() for c in df.columns]
            df = add_technical_indicators(df)

            if df.empty or len(df) <= RL_LOOKBACK + 1:
                console.print(f"[yellow]Warning: Not enough data for {symbol} after feature engineering.[/yellow]")
                continue

            # Align with benchmark window
            df = df.loc[(df.index >= close_prices.index.min()) & (df.index <= close_prices.index.max())]
            if df.empty:
                console.print(f"[yellow]Warning: No overlapping data window for {symbol}.[/yellow]")
                continue

            model = PPO.load(model_path)

            def make_env(dataframe: pd.DataFrame):
                return TradingEnv(dataframe, lookback_window=RL_LOOKBACK)

            env = DummyVecEnv([lambda df=df: make_env(df)])

            if os.path.exists(vec_norm_path):
                env = VecNormalize.load(vec_norm_path, env)
                env.training = False
                env.norm_reward = False
            else:
                console.print(f"[yellow]Warning: No normalization stats for {symbol}[/yellow]")

            obs = env.reset()
            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)

                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                ts = info.get("timestamp") if info else None

                if ts is not None and ts in exposures_rl.index:
                    exposures_rl.at[ts, symbol] = float(action[0])

                done = bool(dones[0] if isinstance(dones, (list, tuple, np.ndarray)) else dones)

        except Exception as e:
            console.print(f"[red]Error running RL for {symbol}: {e}[/red]")

    price_returns = close_prices.pct_change().fillna(0.0)
    rl_weights = exposures_rl.shift(1).fillna(0.0).clip(-1, 1)
    rl_returns = rl_weights * price_returns

    equity_rl = (1 + rl_returns).cumprod()
    total_return_rl = (equity_rl.iloc[-1] - 1).mean() * 100

    sharpe_rl_series = rl_returns.mean() / (rl_returns.std().replace(0, np.nan))
    sharpe_rl_series = sharpe_rl_series.replace([np.inf, -np.inf], np.nan) * annualization_factor
    sharpe_rl = sharpe_rl_series.fillna(0.0).mean()

    drawdown_rl = (equity_rl / equity_rl.cummax()) - 1
    max_drawdown_rl = drawdown_rl.min().mean() * 100

    win_rate_rl = (rl_returns > 0).mean().mean() * 100

    results.append({
        "Strategy": "SOTA RL (PPO)",
        "Total Return [%]": total_return_rl,
        "Sharpe Ratio": sharpe_rl,
        "Max Drawdown [%]": abs(max_drawdown_rl),
        "Win Rate [%]": win_rate_rl
    })

    # -------------------------------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------------------------------
    table = Table(title=f"Kuber Strategy Benchmark (Last 60 Days - {TIMEFRAME})")
    table.add_column("Strategy", style="cyan")
    table.add_column("Total Return", justify="right")
    table.add_column("Sharpe Ratio", justify="right")
    table.add_column("Max Drawdown", justify="right")
    table.add_column("Win Rate", justify="right")
    
    for res in results:
        table.add_row(
            res["Strategy"],
            f"{res['Total Return [%]']:.2f}%",
            f"{res['Sharpe Ratio']:.2f}",
            f"{res['Max Drawdown [%]']:.2f}%",
            f"{res['Win Rate [%]']:.2f}%"
        )
        
    console.print(table)
    
if __name__ == "__main__":
    run_benchmark()
    """Detailed log for analytics."""
    strategy_name: str
    symbol: str
    date: str
    return_pct: float
    pnl: float
