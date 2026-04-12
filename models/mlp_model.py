# models/mlp_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficPredictorMLP(nn.Module):
    """
    MLP based traffic predictor.

    ✅ Constructor LSTM/GRU compatible:
       TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)

    ✅ Input:
       - (batch, seq_len)
       - (batch, seq_len, input_size)

    ✅ Output:
       - (batch, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,   # kept only for compatibility
        num_layers: int = 3,      # kept only for compatibility
        output_size: int = 1,
        seq_len: int = 10,
        hidden_sizes=None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.seq_len = seq_len

        # Agar hidden_sizes explicitly nahi diya, to hidden_size & num_layers se banao
        if hidden_sizes is None:
            # e.g. hidden_size=128, num_layers=3 -> (128, 128, 128)
            hidden_sizes = (hidden_size,) * max(1, num_layers)

        in_features = input_size * seq_len

        layers = []
        last = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(last, output_size)

    def forward(self, x):
        # x: (batch, seq_len) OR (batch, seq_len, input_size)
        if x.dim() == 2:
            # (batch, seq_len)
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            # (batch, seq_len, input_size)
            x = x.view(x.size(0), -1)
        else:
            raise ValueError(f"Expected x with 2 or 3 dims, got shape {x.shape}")

        x = self.mlp(x)
        x = self.out(x)
        return x


# Alias – taki existing code me `TrafficPredictor` naam same rahe
TrafficPredictor = TrafficPredictorMLP
