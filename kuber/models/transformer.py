import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for Time Series Forecasting.
    Uses an Encoder-only architecture to predict future price movements.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1, output_dim: int = 1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.bias.data.zero_()
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            output: Tensor of shape (batch_size, output_dim) - prediction for next step
        """
        # Embed and add position encoding
        src = self.input_embedding(src) # (batch, seq, d_model)
        src = self.pos_encoder(src)
        
        # Pass through Transformer
        output = self.transformer_encoder(src) # (batch, seq, d_model)
        
        # We only care about the last time step for forecasting the next step
        last_output = output[:, -1, :] # (batch, d_model)
        
        prediction = self.decoder(last_output) # (batch, output_dim)
        return prediction

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
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
