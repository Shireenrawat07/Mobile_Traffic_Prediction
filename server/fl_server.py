import os
import sys
import json
import numpy as np
import torch
import flwr as fl
import pandas as pd

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# =========================
# PROJECT PATH
# =========================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.lstm_model import TrafficPredictor


# =========================
# CONFIG
# =========================
if len(sys.argv) < 2:
    print("Usage: python fl_server.py <alpha>")
    sys.exit(1)

ALPHA = float(sys.argv[1])

RESULT_DIR = "results/fedavg_results"
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# FEDAVG STRATEGY
# =========================
class FedCustom(fl.server.strategy.FedAvg):

    def __init__(self, alpha=0.1, total_rounds=30, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.total_rounds = total_rounds

        # unified storage
        self.all_data = []
        self.final_metrics = {}

    # =========================
    # FIT
    # =========================
    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        aggregated, _ = super().aggregate_fit(rnd, results, failures)

        print(f"\nRound {rnd} aggregation complete")

        # save model final round
        if rnd == self.total_rounds:

            model = TrafficPredictor(
                input_size=1,
                hidden_size=128,
                num_layers=3,
                output_size=1
            )

            weights = parameters_to_ndarrays(aggregated)
            state = model.state_dict()

            for i, key in enumerate(state.keys()):
                state[key] = torch.tensor(weights[i]).float()

            model.load_state_dict(state)

            torch.save(
                model.state_dict(),
                f"{RESULT_DIR}/model_alpha_{self.alpha}.pth"
            )

            print(f"Model saved for alpha={self.alpha}")

        return aggregated, {}

    # =========================
    # EVALUATION + LOGGING
    # =========================
    def aggregate_evaluate(self, rnd, results, failures):

        if not results:
            return 0.0, {}

        total_loss = 0.0
        total_n = 0

        print(f"\n--- Round {rnd} Evaluation ---")

        for idx, (_, res) in enumerate(results):

            n = res.num_examples
            total_loss += res.loss * n
            total_n += n

            m = res.metrics

            client_name = m.get("client_name", f"client_{idx+1}")
            mae = float(m.get("mae", 0.0))
            rmse = float(m.get("rmse", 0.0))
            nrmse = float(m.get("nrmse", rmse))

            print(f"{client_name} -> MAE={mae:.6f}, RMSE={rmse:.6f}")

            # =========================
            # CLEAN LOG FORMAT (IMPORTANT)
            # =========================
            self.all_data.append([
                self.alpha,      # alpha (heterogeneity)
                client_name,
                float(res.loss),
                mae,
                rmse,
                nrmse,
                "FedAvg",
                rnd
            ])

            # final metrics
            if rnd == self.total_rounds:
                self.final_metrics[client_name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "NRMSE": nrmse
                }

        avg_loss = total_loss / (total_n + 1e-8)

        print(f"Round {rnd} Loss: {avg_loss:.6f}")

        # =========================
        # SAVE FINAL RESULTS
        # =========================
        if rnd == self.total_rounds:

            df = pd.DataFrame(self.all_data, columns=[
                "alpha",
                "client",
                "loss",
                "mae",
                "rmse",
                "nrmse",
                "algorithm",
                "round"
            ])

            csv_path = f"{RESULT_DIR}/FedAvg_clientwise.csv"
            df.to_csv(csv_path, index=False)

            json_path = f"{RESULT_DIR}/metrics_alpha_{self.alpha}.json"
            with open(json_path, "w") as f:
                json.dump(self.final_metrics, f, indent=4)

            print(f"Saved CSV + JSON for alpha={self.alpha}")

        return avg_loss, {"loss": float(avg_loss)}


# =========================
# MAIN
# =========================
def main():

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

    strategy = FedCustom(
        alpha=ALPHA,
        total_rounds=30,
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        fraction_evaluate=1.0,
        min_evaluate_clients=3,
        initial_parameters=initial_parameters
    )

    print(f"\nStarting FedAvg Server for alpha={ALPHA}")

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()