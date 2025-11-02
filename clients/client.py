# clients/client.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.lstm_model import TrafficPredictor
from utils.data_preprocess import load_real_traffic_data, prepare_sequences
import numpy as np


def train_local_model(filepath='data/traffic_data.csv', column='bandwidth_usage',
                      seq_len=10, epochs=5, lr=0.001, device=None):
    """
    Loads data from filepath, prepares sequences, and trains a local LSTM.
    Returns the model state_dict and final loss (float).
    """

    # auto-detect device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess real traffic data
    series, scaler = load_real_traffic_data(filepath, column)
    X, y = prepare_sequences(series, seq_len=seq_len)

    # Convert numpy arrays to torch tensors
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    # Initialize model
    model = TrafficPredictor(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    final_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        # Optional: print training progress
        # print(f"[Client] Epoch {epoch+1}/{epochs}, Loss: {final_loss:.6f}")

    return model.state_dict(), final_loss
