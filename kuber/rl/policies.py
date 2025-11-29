import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # pe: (max_len, 1, d_model) -> slice -> (seq_len, 1, d_model) -> transpose -> (1, seq_len, d_model)
        # We need to align dimensions. 
        # pe is (max_len, 1, d_model).
        # We want (1, seq_len, d_model) to broadcast to (batch, seq_len, d_model)
        pe_slice = self.pe[:x.size(1), :].transpose(0, 1) # (1, seq_len, d_model)
        x = x + pe_slice
        return self.dropout(x)

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    SOTA Transformer-based feature extractor for PPO.
    Reshapes flattened observation back to (Batch, Seq, Features) and applies Transformer Encoder.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, 
                 lookback_window: int = 30, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        
        # We assume the observation is flattened, so we need to know the original shape
        # observation_space.shape[0] should be lookback_window * n_features
        
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.lookback_window = lookback_window
        
        # Calculate n_features
        # If observation_space is (Lookback, Features), then n_features is shape[1]
        # If observation_space is flattened (Lookback * Features,), then n_features is shape[0] // Lookback
        if len(observation_space.shape) > 1:
            self.n_features = observation_space.shape[1]
        else:
            self.n_features = observation_space.shape[0] // lookback_window
            
        self.d_model = d_model
        
        # Layers
        self.input_embedding = nn.Linear(self.n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Flatten and project to features_dim
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(d_model * lookback_window, features_dim)
        
        # Activation
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. Reshape flattened observation back to (Batch, Seq, Features)
        # observations shape: (Batch, lookback * n_features)
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.lookback_window, self.n_features)
        
        # 2. Embed and Encode
        x = self.input_embedding(x) # (Batch, Seq, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # (Batch, Seq, d_model)
        
        # 3. Flatten and Project
        x = self.flatten(x) # (Batch, Seq * d_model)
        x = self.linear(x)  # (Batch, features_dim)
        x = self.relu(x)
        
        return x
