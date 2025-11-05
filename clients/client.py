import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import TrafficPredictor

# ----- Sequence preparation -----
def prepare_sequences(series, seq_len=10):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)


def train_local_model(filepath, seq_len=10, epochs=30, lr=0.001, global_model_state=None):
    """
    Trains the local LSTM model on a client's dataset.
    Returns updated state_dict and final loss.
    """
    # Load and normalize column 'down' (for simplicity)
    df = pd.read_csv(filepath)
    series = df['down'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    X_np, y_np = prepare_sequences(series_scaled, seq_len)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    # Initialize model
    model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
    if global_model_state is not None:
        model.load_state_dict(global_model_state)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    return model.state_dict(), loss.item()
