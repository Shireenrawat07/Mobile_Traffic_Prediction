import os
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.lstm_model import TrafficPredictor
from utils.Fedprox_data    import get_client_loaders
from clients.Fedprox_client import FedProxClient

# ---------------------------
# NRMSE metric
# ---------------------------
def nrmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse) / (y_true.max() - y_true.min())

# ---------------------------
# Settings
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
local_epochs = 1
num_rounds = 30
mu = 0.001
input_size = 5
output_size = 1

results_dir = "fedprox_results"
os.makedirs(results_dir, exist_ok=True)

# ---------------------------
# Train for multiple alphas
# ---------------------------
for alpha in [0.1, 0.5, 1.0]:
    print(f"\n=== Starting training for alpha={alpha} ===")
    client_loaders = get_client_loaders(alpha=alpha, batch_size=batch_size)

    clients = []
    for idx, (trainloader, testloader) in enumerate(client_loaders):
        print(f"Client {idx+1} initialized with {len(trainloader.dataset)} train samples")
        model = TrafficPredictor(input_size=input_size, hidden_size=128, num_layers=3, output_size=output_size).to(device)
        client = FedProxClient(model, trainloader, testloader, device, mu, local_epochs, lr=1e-4)
        clients.append(client)

    # ---------------------------
    # FedProx manual training
    # ---------------------------
    for r in range(num_rounds):
        print(f"\n--- Round {r+1}/{num_rounds} ---")
        client_weights = []
        client_losses = []

        for client in clients:
            weights, loss = client.fit()
            client_weights.append(weights)
            client_losses.append(loss)

        # FedAvg aggregation
        new_state_dict = {}
        for key in client_weights[0].keys():
            new_state_dict[key] = sum([w[key] for w in client_weights]) / len(client_weights)

        for client in clients:
            client.set_weights(new_state_dict)

        print(f"Round {r+1} done. Avg loss: {np.mean(client_losses):.6f}")

    # ---------------------------
    # Evaluate each client
    # ---------------------------
    metrics = {}
    for idx, (trainloader, testloader) in enumerate(client_loaders):
        client_model = clients[idx].model
        client_model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in testloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                out = client_model(X_batch)
                y_true.append(y_batch.cpu().numpy())
                y_pred.append(out.cpu().numpy())
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        mae_val = mean_absolute_error(y_true, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
        nrmse_val = nrmse(y_true, y_pred)
        metrics[f"client_{idx+1}"] = {"MAE": float(mae_val), "RMSE": float(rmse_val), "NRMSE": float(nrmse_val)}

    # ---------------------------
    # Save model & metrics
    # ---------------------------
    model_path = os.path.join(results_dir, f"model_alpha_{alpha}.pth")
    torch.save(clients[0].model.state_dict(), model_path)

    import json
    metrics_path = os.path.join(results_dir, f"metrics_alpha_{alpha}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved model and metrics for alpha={alpha}")