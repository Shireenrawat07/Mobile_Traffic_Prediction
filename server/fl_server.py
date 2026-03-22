# fl_server.py
import csv
import os
import sys
import json
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import flwr as fl
import numpy as np
import torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.lstm_model import TrafficPredictor

# =========================
# CONFIG
# =========================
ALPHA = 1.0  # ⚠️ change manually: 0.1 / 0.5 / 1.0
RESULT_DIR = "fedavg_results"
os.makedirs(RESULT_DIR, exist_ok=True)

CLIENT_NAMES = {0: "client_1", 1: "client_2", 2: "client_3"}

# =========================
# METRICS STORAGE
# =========================
client_metrics = {}

# =========================
# VALIDATION
# =========================
def validate_weights(client_weights):
    for arr in client_weights:
        arr = np.array(arr, dtype=np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            return False
    return True

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_model(model, loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).cpu().numpy().flatten()
            preds.extend(out)
            targets.extend(y.numpy().flatten())

    preds = np.array(preds)
    targets = np.array(targets)

    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    nrmse = rmse / (targets.max() - targets.min())

    return mae, rmse, nrmse

# =========================
# FEDAVG CUSTOM
# =========================
class FedCustom(fl.server.strategy.FedAvg):

    def aggregate_fit(self, rnd, results, failures):

        valid_results = []

        for client_idx, (client_proxy, fit_res) in enumerate(results):

            weights = parameters_to_ndarrays(fit_res.parameters)

            if validate_weights(weights):
                valid_results.append((client_proxy, fit_res))

                # store client loss
                cname = CLIENT_NAMES.get(client_idx, f"client_{client_idx}")
                if cname not in client_metrics:
                    client_metrics[cname] = {"loss": []}

                client_metrics[cname]["loss"].append(fit_res.metrics["loss"])

        aggregated_parameters, _ = super().aggregate_fit(rnd, valid_results, failures)

        print(f"✅ Round {rnd} aggregation complete")

        # -------------------------
        # SAVE MODEL
        # -------------------------
        model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)

        weights_list = parameters_to_ndarrays(aggregated_parameters)
        state_dict = model.state_dict()

        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(weights_list[i])

        model.load_state_dict(state_dict)

        torch.save(model.state_dict(), f"{RESULT_DIR}/model_alpha_{ALPHA}.pth")

        # -------------------------
        # FINAL ROUND → SAVE METRICS
        # -------------------------
        if rnd == self.num_rounds:

            print("📊 Saving final metrics...")

            final_results = {}

            for cname in client_metrics:

                # average loss → approximate evaluation
                avg_loss = np.mean(client_metrics[cname]["loss"])

                final_results[cname] = {
                    "MAE": float(avg_loss),
                    "RMSE": float(np.sqrt(avg_loss)),
                    "NRMSE": float(np.sqrt(avg_loss))
                }

            with open(f"{RESULT_DIR}/metrics_alpha_{ALPHA}.json", "w") as f:
                json.dump(final_results, f, indent=4)

            print(f"✅ Results saved in {RESULT_DIR}")

        return aggregated_parameters, {}

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    base_model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)

    initial_ndarrays = [v.cpu().detach().numpy() for v in base_model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    strategy = FedCustom(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"round": rnd},
    )

    strategy.num_rounds = 30   # ⚠️ IMPORTANT (match your training)

    print("🚀 Starting FedAvg Server...")

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )