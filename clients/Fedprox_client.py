import sys
import os
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.lstm_model import TrafficPredictor


# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 0.001
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
MU = 0.01


# =========================
# LOAD DATA
# =========================
def load_data(client_id, alpha):

    path = f"Alpha_Splits/splits_alpha_{alpha}/client_{client_id}.pt"
    data = torch.load(path, weights_only=False)

    X, y = data["X"], data["y"]

    split = int(0.8 * len(X))

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X[:split]).float(),
            torch.tensor(y[:split]).float()
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X[split:]).float(),
            torch.tensor(y[split:]).float()
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


# =========================
# CLIENT
# =========================
class FedProxClient(fl.client.NumPyClient):

    def __init__(self, cid, alpha):

        self.cid = cid
        self.alpha = alpha

        self.model = TrafficPredictor(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            output_size=1
        ).to(DEVICE)

        self.train_loader, self.val_loader = load_data(cid, alpha)

        self.loss_fn = nn.MSELoss()

    # =========================
    def get_parameters(self, config=None):
        return [
            v.cpu().detach().numpy()
            for _, v in self.model.state_dict().items()
        ]

    # =========================
    def set_parameters(self, parameters):

        state_dict = self.model.state_dict()

        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(parameters[i]).to(DEVICE)

        self.model.load_state_dict(state_dict)

    # =========================
    # TRAIN
    # =========================
    def fit(self, parameters, config):

        self.set_parameters(parameters)

        global_params = [
            p.clone().detach()
            for p in self.model.parameters()
        ]

        optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for _ in range(LOCAL_EPOCHS):
            for x, y in self.train_loader:

                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()

                out = self.model(x)

                mse_loss = self.loss_fn(out.squeeze(), y.squeeze())

                # =========================
                # FEDPROX REGULARIZATION
                # =========================
                prox = 0.0
                for w, w0 in zip(self.model.parameters(), global_params):
                    prox += torch.norm(w - w0, p=2) ** 2

                loss = mse_loss + (MU / 2.0) * prox

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        avg_loss = total_loss / (total_samples + 1e-8)

        return (
            self.get_parameters(),
            total_samples,
            {
                "loss": float(avg_loss),
                "client_name": f"Client_{self.cid}"
            }
        )

    # =========================
    # EVALUATION (STANDARDIZED)
    # =========================
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)
        self.model.eval()

        preds, targets = [], []

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in self.val_loader:

                x, y = x.to(DEVICE), y.to(DEVICE)

                out = self.model(x)
                loss = self.loss_fn(out.squeeze(), y.squeeze())

                preds.extend(out.cpu().numpy().reshape(-1))
                targets.extend(y.cpu().numpy().reshape(-1))

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        preds = np.array(preds)
        targets = np.array(targets)

        # =========================
        # METRICS (STANDARDIZED)
        # =========================
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))
        variance = np.var(preds)

        denom = np.max(targets) - np.min(targets) + 1e-8
        nrmse = rmse / denom

        avg_loss = total_loss / (total_samples + 1e-8)

        # safety
        rmse = float(np.nan_to_num(rmse, nan=1.0))
        mae = float(np.nan_to_num(mae, nan=1.0))
        nrmse = float(np.nan_to_num(nrmse, nan=1.0))
        variance = float(np.nan_to_num(variance, nan=1.0))

        return (
            float(avg_loss),
            total_samples,
            {
                "loss": float(avg_loss),
                "rmse": rmse,
                "mae": mae,
                "nrmse": nrmse,
                "variance": variance,
                "client_name": f"Client_{self.cid}"
            }
        )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    cid = sys.argv[1]
    alpha = float(sys.argv[2])

    client = FedProxClient(cid, alpha)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )