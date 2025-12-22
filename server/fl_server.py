# fl_server.py
import csv
import os
import sys
from pathlib import Path

# Ensure project root is importable (so `models` and `utils` import works)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import flwr as fl
import numpy as np
import torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from models.lstm_model import TrafficPredictor
from models.gru_model import TrafficPredictorGRU
from models.rnn_model import TrafficPredictorRNN

# ---------------------------
# Logging FedAvg results
# ---------------------------
FEDAVG_RESULTS = "fedavg_results.csv"

def log_fedavg(round_number, loss):
    file_exists = os.path.isfile(FEDAVG_RESULTS)
    with open(FEDAVG_RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["Round", "Loss"])
        w.writerow([round_number, loss])

# ---------------------------
# Add project root to path
# ---------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ---------------------------
# Client name mapping
# ---------------------------
CLIENT_NAMES = {0: "ElBorn", 1: "LesCorts", 2: "PobleSec"}

# ---------------------------
# Validate weights
# ---------------------------
def validate_weights(client_weights):
    for i, arr in enumerate(client_weights):
        arr = np.array(arr, dtype=np.float32)
        if arr.dtype.kind not in ["f", "i"]:
            print(f"⚠️ Non-numeric dtype at index {i}: {arr.dtype}")
            return False
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"❌ Invalid weight values at index {i}")
            return False
    return True

# ---------------------------
# Custom FedAvg strategy
# ---------------------------
class FedCustom(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            print("❌ No results received from clients.")
            return None, {}

        valid_results = []
        for client_idx, (client_proxy, fit_res) in enumerate(results):
            client_name = CLIENT_NAMES.get(client_idx, f"Client {client_idx}")
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            if validate_weights(client_weights):
                print(f"✅ {client_name} weights validated successfully.")
                valid_results.append((client_proxy, fit_res))
            else:
                print(f"⚠️ Skipping invalid weights from {client_name}.")

        if not valid_results:
            print("❌ No valid client weights. Aggregation aborted.")
            return None, {}

        aggregated_parameters, _ = super().aggregate_fit(rnd, valid_results, failures)
        print(f"✅ Aggregation complete for Round {rnd}.")

        # Log average loss
        avg_loss = sum([fit_res.metrics["loss"] for _, fit_res in results]) / len(results)
        log_fedavg(rnd, avg_loss)

        # ---------------------------
        # Save global model after aggregation
        # ---------------------------
        final_model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        # final_model = TrafficPredictorGRU(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        # final_model = TrafficPredictorRNN(input_size=1, hidden_size=128, num_layers=3, output_size=1)

        weights_list = parameters_to_ndarrays(aggregated_parameters)
        state_dict = final_model.state_dict()
        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(weights_list[i])
        final_model.load_state_dict(state_dict)
        torch.save(final_model.state_dict(), "global_model.pth")
        print(f"✅ Global model saved after round {rnd}.")

        return aggregated_parameters, {}

# ---------------------------
# Main server start
# ---------------------------
if __name__ == "__main__":
    # Initial model for starting parameters
    base_model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
    # base_model = TrafficPredictorRNN(input_size=1, hidden_size=128, num_layers=3, output_size=1)
    # base_model = TrafficPredictorGRU(input_size=1, hidden_size=128, num_layers=3, output_size=1)

    initial_ndarrays = [v.cpu().detach().numpy() for v in base_model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    strategy = FedCustom(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"round": rnd},
        fit_metrics_aggregation_fn=lambda metrics: {},
    )

    print("🚀 Starting Flower server (FedAvg w/ validation)...")
    fl.server.start_server(
        server_address=os.environ.get("SERVER_ADDRESS", "localhost:8080"),
        config=fl.server.ServerConfig(num_rounds=int(os.environ.get("NUM_ROUNDS", 30))),
        strategy=strategy,
    )
