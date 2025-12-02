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

    # 1. Trend (Normalized)
    # Instead of raw SMA/EMA, we use the distance from them
    df['dist_sma_20'] = (df['close'] - ta.trend.sma_indicator(df['close'], window=20)) / df['close']
    df['dist_ema_50'] = (df['close'] - ta.trend.ema_indicator(df['close'], window=50)) / df['close']
    df['macd'] = ta.trend.macd_diff(df['close'])
    
    # 2. Momentum
    df['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100.0  # Scale to 0-1
    df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close']) / 100.0 # Scale to 0-1
    
    # 3. Volatility
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']) / df['close'] # Normalize by price
    df['bb_width'] = ta.volatility.bollinger_wband(df['close'])
    
    # [NEW] Rolling Volatility (Standard Deviation of Log Returns)
    # This is crucial for the agent to sense risk regimes
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_20'] = df['log_ret'].rolling(window=20).std()
    df['volatility_5'] = df['log_ret'].rolling(window=5).std()
    
    # [SOTA] Volatility Ratio (Regime Detection)
    # When short-term vol > long-term vol, market is becoming more volatile
    df['vol_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
    
    # [SOTA] Lagged Returns (Momentum Features)
    # Explicitly provide recent returns to help LSTM capture momentum
    df['ret_1'] = df['log_ret'].shift(1)  # Yesterday's return
    df['ret_3'] = df['log_ret'].shift(3)  # 3 periods ago
    df['ret_5'] = df['log_ret'].shift(5)  # 5 periods ago
    
    # 4. Volume
    # Normalize volume by its moving average
    df['vol_ma'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-8)

    # [NEW] Time Features (Cyclic Encoding)
    # Helps agent learn intraday seasonality (open/close volatility)
    if isinstance(df.index, pd.DatetimeIndex):
        # Minute of day (0-1439)
        minutes = df.index.hour * 60 + df.index.minute
        max_minutes = 24 * 60
        df['sin_time'] = np.sin(2 * np.pi * minutes / max_minutes)
        df['cos_time'] = np.cos(2 * np.pi * minutes / max_minutes)
        
        # Day of week (0-6)
        day_of_week = df.index.dayofweek
        df['sin_day'] = np.sin(2 * np.pi * day_of_week / 7)
        df['cos_day'] = np.cos(2 * np.pi * day_of_week / 7)

    # Drop NaNs created by indicators
    df.dropna(inplace=True)
    
    # Normalize features (Z-score) to help the Neural Network
    # We normalize everything that isn't already bounded (like RSI/Stoch/Sin/Cos)
    cols_to_normalize = [
        'macd', 'dist_sma_20', 'dist_ema_50', 'atr', 'bb_width', 
        'log_ret', 'vol_ma', 'volatility_20', 'volatility_5', 'vol_ratio',
        'ret_1', 'ret_3', 'ret_5'
    ]
    for col in cols_to_normalize:
        if col in df.columns:
            # Robust scaling using median/IQR might be better, but mean/std is standard
            # Adding a small epsilon to std to avoid division by zero
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
            # Clip extreme outliers to keep training stable
            df[col] = df[col].clip(-5.0, 5.0)
            
    # Drop raw non-stationary columns that we don't need for the agent
    # We KEEP 'close' because the Environment needs it for PnL calculation, 
    # but the Environment will hide it from the observation.
    cols_to_drop = ['open', 'high', 'low', 'volume']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
    return df
