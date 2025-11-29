import pandas as pd
import numpy as np
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    SOTA Feature Engineering: Adds technical indicators to the dataframe.
    Matches the training environment exactly.
    """
    df = df.copy()
    # Ensure we have data
    if len(df) < 50:
        return df

    # 1. Trend
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['macd'] = ta.trend.macd_diff(df['close'])
    
    # 2. Momentum
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    
    # 3. Volatility
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['bb_width'] = ta.volatility.bollinger_wband(df['close'])
    
    # 4. Log Returns (Crucial for RL stationarity)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 5. Volume
    df['vol_ma'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # Drop NaNs created by indicators
    df.dropna(inplace=True)
    
    # Normalize features (Z-score) to help the Neural Network
    # We exclude 'close' from normalization if we need it for PnL calculation in Env,
    # but usually Env handles raw price separately. 
    # For observation, we want stationary features.
    cols_to_normalize = ['macd', 'rsi', 'stoch', 'atr', 'bb_width', 'log_ret', 'vol_ma']
    for col in cols_to_normalize:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        
    return df
