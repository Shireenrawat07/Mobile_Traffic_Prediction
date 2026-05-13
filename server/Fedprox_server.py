import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import flwr as fl

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
import os
import sys

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.lstm_model import TrafficPredictor



# =========================
# CONFIG
# =========================
if len(sys.argv) < 2:
    print("Usage: python fl_server_fedprox.py <alpha>")
    sys.exit(1)

ALPHA = float(sys.argv[1])
TOTAL_ROUNDS = 30

RESULT_DIR = "results/fedprox_results"
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# STRATEGY
# =========================
class FedProxStrategy(fl.server.strategy.FedAvg):

    def __init__(self, split_value=0.1, total_rounds=30, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.split_value = split_value
        self.total_rounds = total_rounds

        # ===== STANDARD LOG STORAGE =====
        self.all_data = []              # full history (CSV)
        self.round_buffer = []          # per-round temp storage
        self.final_client_metrics = {}  # final JSON

        self.global_weights = None

    # =========================
    # AGGREGATE FIT (UNCHANGED LOGIC)
    # =========================
    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        weights_list = []
        sizes = []
        client_names = []
        losses = []

        for client, fit_res in results:

            weights = parameters_to_ndarrays(fit_res.parameters)

            # keep only valid updates
            if any(np.isnan(w).any() or np.isinf(w).any() for w in weights):
                continue

            weights_list.append(weights)
            sizes.append(fit_res.num_examples)
            losses.append(float(fit_res.metrics.get("loss", 0.0)))
            client_names.append(
                fit_res.metrics.get("client_name", f"Client_{client.cid}")
            )

        if len(weights_list) == 0:
            return None, {}

        total_samples = sum(sizes)

        # FedAvg aggregation (FedProx server side unchanged)
        aggregated = None

        for i, weights in enumerate(weights_list):

            weight = sizes[i] / (total_samples + 1e-8)

            if aggregated is None:
                aggregated = [w * weight for w in weights]
            else:
                aggregated = [
                    a + w * weight
                    for a, w in zip(aggregated, weights)
                ]

        self.global_weights = aggregated

        # =========================
        # ROUND LOGGING (STANDARDIZED)
        # =========================
        self.round_buffer = []

        for i in range(len(client_names)):

            self.round_buffer.append([
                self.split_value,          # alpha
                rnd,                       # round
                client_names[i],          # client
                losses[i],                # loss
                sizes[i],                 # samples
                None,                     # rmse (filled in eval)
                None,                     # mae (filled in eval)
                None                      # nrmse (optional)
            ])

        print(f"\nRound {rnd} completed")

        return ndarrays_to_parameters(aggregated), {}

    # =========================
    # AGGREGATE EVALUATE (STANDARDIZED)
    # =========================
    def aggregate_evaluate(self, rnd, results, failures):

        if not results:
            return 0.0, {}

        total_loss = 0.0
        total_n = 0

        print(f"\n--- Round {rnd} Evaluation ---")

        for i, (_, res) in enumerate(results):

            n = res.num_examples
            total_loss += res.loss * n
            total_n += n

            metrics = res.metrics

            rmse = float(metrics.get("rmse", 0.0))
            mae = float(metrics.get("mae", 0.0))
            nrmse = float(metrics.get("nrmse", rmse))
            client_name = metrics.get("client_name", f"Client_{i+1}")

            print(f"{client_name} -> RMSE={rmse:.6f}, MAE={mae:.6f}")

            # =========================
            # UPDATE ROUND BUFFER
            # =========================
            if i < len(self.round_buffer):
                self.round_buffer[i][5] = rmse
                self.round_buffer[i][6] = mae
                self.round_buffer[i][7] = nrmse

                self.all_data.append(self.round_buffer[i])

            # =========================
            # FINAL ROUND JSON
            # =========================
            if rnd == self.total_rounds:
                self.final_client_metrics[client_name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "NRMSE": nrmse
                }

        avg_loss = total_loss / (total_n + 1e-8)

        print(f"Round {rnd} Loss: {avg_loss:.6f}")

        # =========================
        # SAVE FINAL OUTPUTS
        # =========================
        if rnd == self.total_rounds:

            # -------- CSV (FULL HISTORY) --------
            df = pd.DataFrame(
                self.all_data,
                columns=[
                    "Alpha",
                    "Round",
                    "Client",
                    "Loss",
                    "Samples",
                    "RMSE",
                    "MAE",
                    "NRMSE"
                ]
            )

            df["Algorithm"] = "FedProx"

            csv_path = os.path.join(
                RESULT_DIR,
                "FedProx_ClientWise.csv"
            )

            df.to_csv(
                csv_path,
                index=False
            )

            # -------- JSON (FINAL ONLY) --------
            json_path = os.path.join(
                RESULT_DIR,
                f"metrics_alpha_{self.split_value}.json"
            )

            with open(json_path, "w") as f:
                json.dump(self.final_client_metrics, f, indent=4)

            print(f"\n✅ FedProx results saved for alpha={self.split_value}")

        return avg_loss, {"rmse": float(np.sqrt(avg_loss))}


# =========================
# MAIN
# =========================
def main():

    strategy = FedProxStrategy(
        split_value=ALPHA,
        total_rounds=TOTAL_ROUNDS,

        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,

        fraction_evaluate=1.0,
        min_evaluate_clients=3,
    )

    print(f"\n🚀 Starting FedProx (alpha={ALPHA})")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        strategy=strategy
    )


if __name__ == "__main__":
    main()