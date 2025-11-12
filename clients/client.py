import sys
import os
import torch
import flwr as fl

# Add parent directory so we can import client.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clients.client import train_local_model  # Your existing training function
from models.lstm_model import TrafficPredictor

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🌸 Flower Client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data=None):
        self.model = model.to(DEVICE)
        self.train_data = train_data
        self.val_data = val_data

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, parameters, config):
        # Load global weights
        state_dict = dict(
            zip(self.model.state_dict().keys(), [torch.tensor(p).to(DEVICE) for p in parameters])
        )
        self.model.load_state_dict(state_dict, strict=True)

        # Perform local training using your client.py function
        new_weights, loss = train_local_model(filepath="Dataset/full_dataset.csv", epochs=config.get("epochs", 2))

        return [val.cpu().numpy() for val in new_weights.values()], len(self.train_data), {}

    def evaluate(self, parameters, config):
        if self.val_data is None:
            return 0.0, 0, {}

        # Load global weights
        state_dict = dict(
            zip(self.model.state_dict().keys(), [torch.tensor(p).to(DEVICE) for p in parameters])
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        with torch.no_grad():
            X, y = self.val_data
            output = self.model(X)
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(output, y)

        return float(loss.item()), len(y), {"val_loss": float(loss.item())}


# -------------------------------
# 🚀 Start Flower Client
# -------------------------------
if __name__ == "__main__":
    # Initialize model
    model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1).to(DEVICE)

    # Load validation data if needed (X_val, y_val)
    val_data = None  # Or load your val data: (X_val, y_val)

    # Start client
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FlowerClient(model=model, train_data=None, val_data=val_data)
    )
