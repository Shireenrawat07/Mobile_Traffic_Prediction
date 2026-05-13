# fednova_server.py (FINAL FIXED + UNIFIED LOGGING)

import flwr as fl
import numpy as np
import torch
import pandas as pd
import os
import json
import sys

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import os
import sys

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.append(ROOT)
from models.lstm_model import TrafficPredictor


class FedNovaStrategy(fl.server.strategy.FedAvg):

    def __init__(self, split_value=0.1, total_rounds=30, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.split_value = split_value
        self.total_rounds = total_rounds

        # ===== UNIFIED LOGGING STORAGE =====
        self.all_data = []
        self.round_buffer = []
        self.final_client_metrics = {}
        self.global_weights = None

    # =========================
    # FIT AGGREGATION (UNCHANGED LOGIC)
    # =========================
    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        weights_list, taus, sizes, names, losses = [], [], [], [], []

        for client, fit_res in results:

            weights_list.append(
                parameters_to_ndarrays(fit_res.parameters)
            )

            taus.append(float(fit_res.metrics.get("tau", 1.0)))
            losses.append(float(fit_res.metrics.get("loss", 0.0)))
            sizes.append(fit_res.num_examples)

            names.append(
                fit_res.metrics.get("client_name", f"Client_{client.cid}")
            )

        total_samples = sum(sizes)

        if self.global_weights is None:
            self.global_weights = [w.copy() for w in weights_list[0]]

        aggregated_delta = [
            np.zeros_like(w) for w in self.global_weights
        ]

        # ===== FEDNOVA CORE =====
        for i, local_w in enumerate(weights_list):

            delta = [
                gw - lw
                for gw, lw in zip(self.global_weights, local_w)
            ]

            normalized_delta = [
                d / (taus[i] + 1e-8) for d in delta
            ]

            weight = sizes[i] / (total_samples + 1e-8)

            aggregated_delta = [
                ad + weight * nd
                for ad, nd in zip(aggregated_delta, normalized_delta)
            ]

        new_global = [
            gw - ad
            for gw, ad in zip(self.global_weights, aggregated_delta)
        ]

        self.global_weights = new_global

        # =========================
        # ROUND BUFFER (FIXED STRUCTURE)
        # =========================
        self.round_buffer = []

        for i in range(len(names)):

            self.round_buffer.append([
                self.split_value,   # Alpha
                rnd,                # Round ✔ FIXED
                names[i],          # Client
                losses[i],         # Loss
                sizes[i],          # Samples
                None,              # RMSE
                None,              # MAE
                None,              # NRMSE
                "FedNova"          # Algorithm ✔ FIXED
            ])

        return ndarrays_to_parameters(new_global), {}

    # =========================
    # EVALUATION (STANDARDIZED)
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

            rmse = float(res.metrics.get("rmse", 0.0))
            mae = float(res.metrics.get("mae", 0.0))
            nrmse = float(res.metrics.get("nrmse", rmse))
            client_name = res.metrics.get("client_name", f"Client_{i+1}")

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

            # ---------- CSV (UNIFIED FORMAT) ----------
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
                    "NRMSE",
                    "Algorithm"
                ]
            )

            csv_path = "results/fednova_results/FedNova_ClientWise.csv"
            df.to_csv(csv_path, index=False)

            # ---------- JSON ----------
            json_path = f"results/fednova_results/metrics_alpha_{self.split_value}.json"

            with open(json_path, "w") as f:
                json.dump(self.final_client_metrics, f, indent=4)

            print(f"\n✅ FedNova results saved for alpha={self.split_value}")

        return avg_loss, {"rmse": float(np.sqrt(avg_loss))}
        # =========================
# MAIN
# =========================
def main():

    if len(sys.argv) < 2:
        print("Usage: python Fednova_server.py <alpha>")
        sys.exit(1)

    alpha = float(sys.argv[1])

    # create result folders
    os.makedirs("results/fednova_results", exist_ok=True)

    # initial model
    base_model = TrafficPredictor(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        output_size=1
    )

    initial_parameters = ndarrays_to_parameters([
        v.cpu().detach().numpy()
        for v in base_model.state_dict().values()
    ])

    strategy = FedNovaStrategy(
        split_value=alpha,
        total_rounds=30,

        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,

        fraction_evaluate=1.0,
        min_evaluate_clients=3,

        initial_parameters=initial_parameters
    )

    print(f"\nStarting FedNova Server for alpha={alpha}")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy
    )


if __name__ == "__main__":
    main()