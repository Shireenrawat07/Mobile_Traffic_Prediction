# fl_client.py
# Place this file in your clients/ (or fl-client/) folder.

import sys
import os
# Ensure project root is importable (so `models` and `utils` import works)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# Import your project's model and preprocessing (must exist)
from models.lstm_model import TrafficPredictor
from models.rnn_model import TrafficPredictorRNN
from models.gru_model import TrafficPredictorGRU
from utils.data_preprocess import load_real_traffic_data, prepare_sequences

# -----------------------
# Config (tweak if necessary)
# -----------------------
SEQ_LEN = 10
COLUMN = "down"
BATCH_SIZE = 64           # use batches so we don't allocate giant tensors
LR = 0.005
LOCAL_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map canonical city keys to dataset file names (loader will attempt variants)
DEFAULT_CITY_FILES = {
    "ElBorn": ["Dataset/ElBorn.csv", "Dataset/Elborn.csv", "Dataset/elborn.csv", "Dataset/ElBorn .csv"],
    "LesCorts": ["Dataset/LesCorts.csv", "Dataset/LesCort.csv", "Dataset/lescorts.csv"],
    "PobleSec": ["Dataset/PobleSec.csv", "Dataset/PobleSec.csv", "Dataset/poblesec.csv"],
}

# -----------------------
# Helpers
# -----------------------
def find_city_file(city_name: str):
    # Try provided candidates then direct path
    candidates = DEFAULT_CITY_FILES.get(city_name, [])
    # Also try direct path as provided (in case user passed filename)
    candidates.append(f"Dataset/{city_name}.csv")
    for p in candidates:
        if os.path.exists(p):
            return p
    # try case-insensitive search in Dataset/
    dataset_dir = Path("Dataset")
    if dataset_dir.exists() and dataset_dir.is_dir():
        for f in dataset_dir.iterdir():
            if f.is_file() and f.name.lower().startswith(city_name.lower()):
                return str(f)
    raise FileNotFoundError(f"No dataset CSV found for '{city_name}'. Tried: {candidates}")



def load_client_data(file_path):
    # 1. Load raw series
    series = load_real_traffic_data(file_path, COLUMN)

    # 2. Scale here (client-side)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    # 3. Save this client's scaler (optional)
    torch.save(
        {"min_": scaler.min_, "scale_": scaler.scale_},
        f"{file_path}_scaler.pt"
    )

    # 4. Prepare sequences
    X, y = prepare_sequences(scaled_series, SEQ_LEN)

    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/Val split
    n = len(X)
    split = int(0.8 * n)

    x_train, y_train = X[:split], y[:split]
    x_val, y_val = X[split:], y[split:]

    # Convert to tensors
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float()),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).float()),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, len(x_train), len(x_val)



def get_model_weights(model: nn.Module):
    return [v.cpu().detach().numpy() for v in model.state_dict().values()]

def set_model_weights(model: nn.Module, weights):
    state_dict = model.state_dict()
    for i, key in enumerate(state_dict.keys()):
        # preserve ordering -> keeps Layer 0/1/2 output identical
        state_dict[key] = torch.tensor(weights[i])
    model.load_state_dict(state_dict)

# -----------------------
# Flower client
# -----------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, city_name: str, file_path: str):
        self.city_name = city_name
        self.file_path = file_path

        # instantiate LSTM and move to device
        self.model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        # self.model = TrafficPredictorRNN(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        # self.model = TrafficPredictorGRU(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        self.model.to(DEVICE)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # load data (batches) — this prevents memory explosion
        self.train_loader, self.val_loader, self.train_count, self.val_count = load_client_data(file_path)

        print(f"\n📁 Loaded dataset for {city_name}: train_samples={self.train_count}, val_samples={self.val_count}")

    def get_parameters(self, config=None):
        return get_model_weights(self.model)

    def fit(self, parameters, config):
        # set incoming global parameters
        set_model_weights(self.model, parameters)
        self.model.train()

        epoch_loss = 0.0
        total_examples = 0

        for _ in range(LOCAL_EPOCHS):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                # If model outputs shape (batch, 1) but batch_y may be (batch, seq, 1) or (batch, 1)
                # adjust shapes as your model expects. Here we assume model returns (batch, 1)
                loss = self.criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                total_examples += batch_x.size(0)

        avg_loss = (epoch_loss / total_examples) if total_examples > 0 else float("nan")
        print(f"🏋️ {self.city_name} Training Loss: {avg_loss:.6f}")

        return get_model_weights(self.model), total_examples, {"loss": avg_loss}

    def evaluate(self, parameters, config):
        # set parameters then evaluate on local val set
        set_model_weights(self.model, parameters)
        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y.squeeze())
                total_loss += loss.item() * batch_x.size(0)
                total_examples += batch_x.size(0)

        avg_loss = (total_loss / total_examples) if total_examples > 0 else float("nan")
        print(f"🔎 {self.city_name} Eval Loss: {avg_loss:.6f}")
        return float(avg_loss), total_examples, {"val_loss": avg_loss}


# -----------------------
# CLI / start-up
# -----------------------
if __name__ == "__main__":
    # usage: python fl_client.py ElBorn
    if len(sys.argv) < 2:
        print("Usage: python fl_client.py <CityName> (e.g. ElBorn)")
        sys.exit(1)

    city = sys.argv[1]
    try:
        file_path = find_city_file(city)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    client = FLClient(city, file_path)
    fl.client.start_numpy_client(server_address=os.environ.get("SERVER_ADDRESS", "localhost:8080"), client=client)
