import csv, os

SIMPLE_RESULTS = "medianavg_exp/simple_results.csv"

def log_simple(round_number, loss):
    file_exists = os.path.isfile(SIMPLE_RESULTS)
    with open(SIMPLE_RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["Round", "Loss"])
        w.writerow([round_number, loss])

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# IMPORTANT: load model to get layer names in correct order
from models.lstm_model import TrafficPredictor

model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
state_keys = list(model.state_dict().keys())


# ------------------------------------------------------
# SIMPLE AVERAGE FUNCTION (each client contributes equally)
# ------------------------------------------------------
def simple_average_models(local_models):
    if not local_models:
        print("❌ No models received.")
        return None

    avg_state = {}
    keys = local_models[0].keys()

    for key in keys:
        tensors = torch.stack([state[key] for state in local_models], dim=0)
        avg_state[key] = torch.mean(tensors, dim=0)

    return avg_state


# ------------------------------------------------------
# SIMPLE AVERAGE STRATEGY (parallel to your MedianStrategy)
# ------------------------------------------------------
class SimpleAverageStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            print("❌ No results received.")
            return None, {}

        local_states = []

        # Convert ndarrays → tensor state_dict
        for _, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            state = {}

            for key, arr in zip(state_keys, nds):
                state[key] = torch.tensor(arr)

            local_states.append(state)

        # SIMPLE AVERAGING
        aggregated_state = simple_average_models(local_states)

        # Convert state_dict back to Flower ndarrays
        aggregated_list = [
            aggregated_state[key].cpu().numpy() for key in state_keys
        ]
        aggregated_parameters = ndarrays_to_parameters(aggregated_list)

        print(f"✅ Simple Average aggregation done for round {rnd}")

        # Log loss
        avg_loss = sum([fit_res.metrics["loss"] for _, fit_res in results]) / len(results)
        log_simple(rnd, avg_loss)

        return aggregated_parameters, {}


# ------------------------------------------------------
# START SERVER
# ------------------------------------------------------
if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8090",   # you chose this port
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=SimpleAverageStrategy()
    )
