import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays

CLIENT_NAMES = {0: "Elborn", 1: "LesCorts", 2: "PobleSec"}
# -------------------------------
# Validate weights
# -------------------------------
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

# -------------------------------
# Custom FedAvg with validation
# -------------------------------

class FedCustom(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        """Aggregate weights after validating them."""
        if not results:
            print("❌ No results received from clients.")
            return None, {}

        valid_results = []

        for client_idx, (client_proxy, fit_res) in enumerate(results):
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            client_name = CLIENT_NAMES.get(client_idx, f"Client {client_idx}")

            if validate_weights(client_weights):
                print(f"✅ {client_name} weights validated successfully.")
                valid_results.append((client_proxy, fit_res))
            else:
                print(f"⚠️ Skipping invalid weights from {client_name}.")

        if not valid_results:
            print("❌ No valid client weights. Aggregation aborted.")
            return None, {}

        aggregated_parameters, _ = super().aggregate_fit(rnd, valid_results, failures)
        print(f"✅ Aggregation complete for Round {rnd}.\n")
        return aggregated_parameters, {}


# -------------------------------
# Start server
# -------------------------------
if __name__ == "__main__":
    strategy = FedCustom(
    fraction_fit=1.0,
    min_fit_clients=3,         
    min_available_clients=3,   
    on_fit_config_fn=lambda rnd: {"round": rnd},
    fit_metrics_aggregation_fn=lambda metrics: {},  # resolves warning
)


    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy
    )
