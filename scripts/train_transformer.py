import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuber.models.transformer import TimeSeriesTransformer

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length] # Predict next step
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_transformer():
    print("Fetching training data...")
    symbol = "SPY"
    start_date = datetime.now() - timedelta(days=365*2)
    end_date = datetime.now()
    df = yf.download(symbol, start=start_date, end=end_date, interval="1h")
    
    if df.empty:
        print("No data fetched.")
        return

    # Use Close price for simplicity
    data = df['Close'].values.reshape(-1, 1)
    
    # Normalize
    mean = data.mean()
    std = data.std()
    data_norm = (data - mean) / std
    
    seq_length = 30
    X, y = create_sequences(data_norm, seq_length)
    
    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(y).float()
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model
    model = TimeSeriesTransformer(input_dim=1, output_dim=1, d_model=64, nhead=4, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    # Save Model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "transformer_forecaster.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_transformer()
