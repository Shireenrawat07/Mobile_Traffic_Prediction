import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from copy import deepcopy
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from models.lstm_model import TrafficPredictor
from utils.data_preprocess import load_real_traffic_data, prepare_sequences

from torch.utils.data import TensorDataset, DataLoader


# ================= CONFIG =================
SEQ_LEN = 10
BATCH_SIZE = 64
LOCAL_EPOCHS = 3
LR = 0.005
ROUNDS = 30
NUM_CLIENTS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FILES = [
    "Dataset/ElBorn.csv",
    "Dataset/LesCorts.csv",
    "Dataset/PobleSec.csv"
]

ALPHAS = [0.1, 0.5, 1.0]

RESULT_DIR = "fednova_results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ================= DATA =================
def load_full_data():
    X_all, y_all = [], []

    for file in DATA_FILES:
        series = load_real_traffic_data(file, "down")

        scaler = MinMaxScaler()
        series = scaler.fit_transform(series)

        X, y = prepare_sequences(series, SEQ_LEN)

        if X.ndim == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))

        X_all.append(X)
        y_all.append(y)

    return np.concatenate(X_all), np.concatenate(y_all)


def dirichlet_split(X, y, alpha):
    N = len(X)

    proportions = np.random.dirichlet([alpha] * NUM_CLIENTS)
    proportions = (proportions * N).astype(int)
    proportions[-1] = N - sum(proportions[:-1])

    indices = np.random.permutation(N)

    client_data = []
    start = 0

    for p in proportions:
        if p == 0:
            p = 1
        idx = indices[start:start + p]
        client_data.append((X[idx], y[idx]))
        start += p

    return client_data


def make_loader(X, y):
    dataset = TensorDataset(
        torch.tensor(X).float(),
        torch.tensor(y).float()
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ================= TRAIN =================
def train_local(model, loader):
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    total_loss = 0
    total_samples = 0

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    tau = LOCAL_EPOCHS * len(loader)

    return deepcopy(model.state_dict()), tau, len(loader.dataset), avg_loss


# ================= FEDNOVA =================
def fednova_aggregate(global_model, client_weights, taus, sizes):
    total = sum(sizes)

    new_state = deepcopy(global_model.state_dict())

    for key in new_state.keys():
        agg = 0
        for k in range(len(client_weights)):
            nk = sizes[k]
            tauk = taus[k]
            agg += (nk / total) * (client_weights[k][key] / tauk)

        new_state[key] = agg

    global_model.load_state_dict(new_state)
    return global_model


# ================= EVAL =================
def evaluate(model, X, y):
    model.eval()

    X_tensor = torch.tensor(X).float().to(DEVICE)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    nrmse = rmse / (y.max() - y.min() + 1e-8)

    return mae, rmse, nrmse


# ================= MAIN =================
def run_experiment():

    X, y = load_full_data()

    for alpha in ALPHAS:

        print(f"\n=== Starting training for alpha={alpha} ===")

        global_model = TrafficPredictor(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            output_size=1
        ).to(DEVICE)

        client_sets = dirichlet_split(X, y, alpha)
        client_loaders = [make_loader(cx, cy) for cx, cy in client_sets]

        # ✅ print client sizes (same as FedProx)
        for i, (cx, cy) in enumerate(client_sets):
            print(f"Client {i+1} initialized with {len(cx)} train samples")

        # -------- Rounds --------
        for r in range(ROUNDS):

            print(f"\n--- Round {r+1}/{ROUNDS} ---")

            weights = []
            taus = []
            sizes = []
            losses = []
            for i, loader in enumerate(client_loaders):
                local_model = deepcopy(global_model)

                w, tau, size, loss = train_local(local_model, loader)

                weights.append(w)
                taus.append(tau)
                sizes.append(size)
                losses.append(loss)

    # ✅ ADD THIS LINE (tau print)
                print(f"Client {i+1} trained (tau={tau})")

            for loader in client_loaders:
                local_model = deepcopy(global_model)

                w, tau, size, loss = train_local(local_model, loader)

                weights.append(w)
                taus.append(tau)
                sizes.append(size)
                losses.append(loss)

            global_model = fednova_aggregate(
                global_model, weights, taus, sizes
            )

            avg_round_loss = np.mean(losses)
            
            print(f"Round {r+1} done. Avg loss: {avg_round_loss:.6f}")
            

        # -------- Evaluation --------
        mae, rmse, nrmse = evaluate(global_model, X, y)

        torch.save(
            global_model.state_dict(),
            f"{RESULT_DIR}/fednova_model_alpha_{alpha}.pt"
        )
        metrics = {}

        for i, (cx, cy) in enumerate(client_sets):
            c_mae, c_rmse, c_nrmse = evaluate(global_model, cx, cy)

            metrics[f"client_{i+1}"] = {
                "MAE": float(c_mae),
                "RMSE": float(c_rmse),
                "NRMSE": float(c_nrmse)
            }

        with open(f"{RESULT_DIR}/metrics_alpha_{alpha}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved model and metrics for alpha={alpha}")
        


if __name__ == "__main__":
    run_experiment()