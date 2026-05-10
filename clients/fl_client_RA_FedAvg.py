# fl_client_RA_FedProx_Hybrid.py

import sys
import os

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.append(ROOT)

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from models.lstm_model import TrafficPredictor

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LR = 0.001
LOCAL_EPOCHS = 5
MU = 0.01


# =====================================
# LOAD DATA
# =====================================
def load_data(client_id, alpha):

    path = f"splits_alpha_{alpha}/client_{client_id}.pt"

    data = torch.load(
        path,
        weights_only=False
    )

    X, y = data["X"], data["y"]

    split = int(0.8 * len(X))

    train_x, train_y = X[:split], y[:split]
    val_x, val_y = X[split:], y[split:]

    train_loader = DataLoader(

        TensorDataset(

            torch.tensor(train_x).float(),
            torch.tensor(train_y).float()

        ),

        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(

        TensorDataset(

            torch.tensor(val_x).float(),
            torch.tensor(val_y).float()

        ),

        batch_size=64,
        shuffle=False
    )

    return train_loader, val_loader


# =====================================
# CLIENT
# =====================================
class Client(fl.client.NumPyClient):

    def __init__(self, cid, alpha):

        self.cid = cid
        self.alpha = alpha

        self.model = TrafficPredictor(

            input_size=1,
            hidden_size=128,
            num_layers=3,
            output_size=1

        ).to(DEVICE)

        self.train_loader, self.val_loader = load_data(
            cid,
            alpha
        )

        self.loss_fn = nn.MSELoss()

    # =====================================
    def get_parameters(self, config=None):

        return [

            p.detach().cpu().numpy()

            for p in self.model.state_dict().values()
        ]

    # =====================================
    def fit(self, parameters, config):

        state_dict = self.model.state_dict()

        for i, k in enumerate(state_dict.keys()):

            state_dict[k] = torch.tensor(
                parameters[i]
            ).to(DEVICE)

        self.model.load_state_dict(state_dict)

        global_params = [

            p.clone().detach()

            for p in self.model.parameters()
        ]

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=LR
        )

        self.model.train()

        total_loss = 0
        total_n = 0

        for _ in range(LOCAL_EPOCHS):

            for x, y in self.train_loader:

                x, y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()

                out = self.model(x)

                mse_loss = self.loss_fn(
                    out.squeeze(),
                    y.squeeze()
                )

                prox_term = 0.0

                for w, w_t in zip(
                    self.model.parameters(),
                    global_params
                ):

                    prox_term += torch.norm(
                        w - w_t
                    ) ** 2

                loss = mse_loss + (MU / 2) * prox_term

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    5.0
                )

                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_n += x.size(0)

        avg_loss = total_loss / (
            total_n + 1e-8
        )

        # variance
        self.model.eval()

        preds = []

        with torch.no_grad():

            for x, _ in self.val_loader:

                x = x.to(DEVICE)

                out = self.model(x)

                preds.extend(

                    out.cpu()
                    .numpy()
                    .reshape(-1)

                )

        preds = np.array(preds)

        variance = float(
            np.var(preds) + 1e-6
        )

        divergence = float(
            np.mean([
                torch.norm(
                    w - w_t
                ).item()

                for w, w_t in zip(
                    self.model.parameters(),
                    global_params
                )
            ])
        )

        return (

            self.get_parameters(),

            total_n,

            {

                "loss": avg_loss,
                "variance": variance,
                "divergence": divergence,
                "client_name": f"Client_{self.cid}"

            }

        )

    # =====================================
    def evaluate(self, parameters, config):

        state_dict = self.model.state_dict()

        for i, k in enumerate(state_dict.keys()):

            state_dict[k] = torch.tensor(
                parameters[i]
            ).to(DEVICE)

        self.model.load_state_dict(state_dict)

        self.model.eval()

        preds = []
        targets = []

        total_loss = 0
        total_n = 0

        with torch.no_grad():

            for x, y in self.val_loader:

                x, y = x.to(DEVICE), y.to(DEVICE)

                out = self.model(x)

                preds.extend(
                    out.cpu().numpy().reshape(-1)
                )

                targets.extend(
                    y.cpu().numpy().reshape(-1)
                )

                loss = self.loss_fn(
                    out.squeeze(),
                    y.squeeze()
                )

                total_loss += loss.item() * x.size(0)
                total_n += x.size(0)

        preds = np.array(preds)
        targets = np.array(targets)

        rmse = np.sqrt(
            np.mean((preds - targets) ** 2)
        )

        mae = np.mean(
            np.abs(preds - targets)
        )

        avg_loss = total_loss / (
            total_n + 1e-8
        )

        return (

            float(avg_loss),

            total_n,

            {

                "rmse": float(rmse),
                "mae": float(mae)

            }

        )


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":

    cid = sys.argv[1]
    alpha = float(sys.argv[2])

    client = Client(cid, alpha)

    fl.client.start_numpy_client(

        server_address="localhost:8080",

        client=client
    )