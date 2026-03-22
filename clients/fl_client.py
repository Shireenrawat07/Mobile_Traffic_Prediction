# fl_client.py

import sys
import os

# Ensure project root is importable
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

from models.lstm_model import TrafficPredictor
from models.rnn_model import TrafficPredictorRNN
from models.gru_model import TrafficPredictorGRU
from utils.data_preprocess import load_real_traffic_data, prepare_sequences

# -----------------------
# Config
# -----------------------
SEQ_LEN = 10
COLUMN = "down"
BATCH_SIZE = 64
LR = 0.005
LOCAL_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Dirichlet Loader
# -----------------------
def load_dirichlet_client_data(client_id):
    data_path = f"splits_alpha_1.0/client_{client_id}.pt"   # change alpha if needed

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Split not found: {data_path}")

 
    data = torch.load(data_path, weights_only=False)

    X = data["X"]
    y = data["y"]

    split = int(0.8 * len(X))

    x_train, y_train = X[:split], y[:split]
    x_val, y_val = X[split:], y[split:]

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

# -----------------------
# Model helpers
# -----------------------
def get_model_weights(model: nn.Module):
    return [v.cpu().detach().numpy() for v in model.state_dict().values()]

def set_model_weights(model: nn.Module, weights):
    state_dict = model.state_dict()
    for i, key in enumerate(state_dict.keys()):
        state_dict[key] = torch.tensor(weights[i])
    model.load_state_dict(state_dict)

# -----------------------
# Flower Client
# -----------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, city_name: str, file_path: str):
        self.city_name = city_name
        self.file_path = file_path

        self.model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        # self.model = TrafficPredictorRNN(...)
        # self.model = TrafficPredictorGRU(...)
        self.model.to(DEVICE)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # ✅ Using Dirichlet data (city_name acts as client_id)
        self.train_loader, self.val_loader, self.train_count, self.val_count = load_dirichlet_client_data(city_name)

        print(f"\n📁 Client {city_name}: train_samples={self.train_count}, val_samples={self.val_count}")

    def get_parameters(self, config=None):
        return get_model_weights(self.model)

    def fit(self, parameters, config):
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
                loss = self.criterion(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                total_examples += batch_x.size(0)

        avg_loss = epoch_loss / total_examples
        print(f"🏋️ Client {self.city_name} Training Loss: {avg_loss:.6f}")

        return get_model_weights(self.model), total_examples, {"loss": avg_loss}

    def evaluate(self, parameters, config):
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

        avg_loss = total_loss / total_examples
        print(f"🔎 Client {self.city_name} Eval Loss: {avg_loss:.6f}")

        return float(avg_loss), total_examples, {"val_loss": avg_loss}

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fl_client.py <client_id>")
        sys.exit(1)

    client_id = sys.argv[1]

    # Pass client_id as city_name (no structural change)
    client = FLClient(client_id, None)

    fl.client.start_numpy_client(
        server_address=os.environ.get("SERVER_ADDRESS", "localhost:8080"),
        client=client
    )