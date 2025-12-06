import csv
import os
import sys
import torch
import flwr as fl
from simple_aggregator import simple_average_models as simple_average


# --------------------------
# 1. Logging Setup
# --------------------------
SIMPLEAVG_RESULTS = "simpleavg_exp/simpleavg_results.csv"

def log_simpleavg(rnd, loss):
    file_exists = os.path.isfile(SIMPLEAVG_RESULTS)
    os.makedirs("simpleavg_exp", exist_ok=True)
    with open(SIMPLEAVG_RESULTS, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Round", "Loss"])
        writer.writerow([rnd, loss])


# --------------------------
# 2. Import Root and Model
# --------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from simple_aggregator import simple_average_models
from models.lstm_model import TrafficPredictor


# --------------------------
# 3. Load model to obtain layer keys
# --------------------------
model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
state_keys = list(model.state_dict().keys())


# --------------------------
# 4. Strategy Definition
# --------------------------
class SimpleAverageStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, rnd, results, failures):

        if not results:
            print("❌ No client results received.")
            return None, {}

        print(f"\n🔄 Performing SIMPLE AVERAGE aggregation for round {rnd}")

        local_states = []

        # Convert Flower parameters → PyTorch state_dict
        for _, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)

            state = {}
            for key, arr in zip(state_keys, nds):
                state[key] = torch.tensor(arr)

            local_states.append(state)

        # Apply simple average algorithm
        aggregated_state = simple_average(local_states)

        # Convert state_dict back → list of ndarrays for Flower
        aggregated_list = [aggregated_state[k].cpu().numpy() for k in state_keys]
        aggregated_parameters = ndarrays_to_parameters(aggregated_list)

        # Log loss
        avg_loss = sum([fit_res.metrics["loss"] for _, fit_res in results]) / len(results)
        log_simpleavg(rnd, avg_loss)

        print(f"✅ Round {rnd} complete | Avg Loss Logged: {avg_loss:.4f}")

        return aggregated_parameters, {}


# --------------------------
# 5. Start Server
# --------------------------
if __name__ == "__main__":
    print("🚀 Starting SIMPLE AVERAGE server on port 8090...")
    fl.server.start_server(
        server_address="localhost:8090",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=SimpleAverageStrategy()
    )
