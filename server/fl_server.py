# fl_server.py

import os
import sys
import json

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT not in sys.path:
    sys.path.append(ROOT)

import flwr as fl
import numpy as np
import torch

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters
)

from models.lstm_model import TrafficPredictor

# =========================
# ALPHA FROM TERMINAL
# =========================
if len(sys.argv) < 2:

    print("Usage: python fl_server.py <alpha>")
    sys.exit(1)

ALPHA = sys.argv[1]

# =========================
# RESULT DIRECTORY
# =========================
RESULT_DIR = "results/fedavg_results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# =========================
# CLIENT METRICS STORAGE
# =========================
client_metrics = {}

# =========================
# VALIDATE WEIGHTS
# =========================
def validate_weights(client_weights):

    for arr in client_weights:

        arr = np.array(
            arr,
            dtype=np.float32
        )

        if np.isnan(arr).any() or np.isinf(arr).any():
            return False

    return True


# =========================
# CUSTOM FEDAVG
# =========================
class FedCustom(fl.server.strategy.FedAvg):

    # -------------------------
    # AGGREGATE FIT
    # -------------------------
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):

        valid_results = []

        for client_proxy, fit_res in results:

            weights = parameters_to_ndarrays(
                fit_res.parameters
            )

            if validate_weights(weights):

                valid_results.append(
                    (client_proxy, fit_res)
                )

        if len(valid_results) == 0:
            return None, {}

        aggregated_parameters, _ = super().aggregate_fit(
            rnd,
            valid_results,
            failures
        )

        print(
            f"\n✅ Round {rnd} aggregation complete"
        )

        # -------------------------
        # SAVE MODEL ONLY FINAL ROUND
        # -------------------------
        if rnd == 30:

            model = TrafficPredictor(
                input_size=1,
                hidden_size=128,
                num_layers=3,
                output_size=1
            )

            weights_list = parameters_to_ndarrays(
                aggregated_parameters
            )

            state_dict = model.state_dict()

            for i, key in enumerate(state_dict.keys()):

                state_dict[key] = torch.tensor(
                    weights_list[i]
                ).float()

            model.load_state_dict(state_dict)

            torch.save(
                model.state_dict(),
                f"{RESULT_DIR}/model_alpha_{ALPHA}.pth"
            )

            print(
                f"✅ Model saved for alpha={ALPHA}"
            )

        return aggregated_parameters, {}

    # -------------------------
    # AGGREGATE EVALUATE
    # -------------------------
    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures
    ):

        if len(results) == 0:

            print("❌ No evaluation results received")

            return 0.0, {}

        total_loss = 0
        total_examples = 0

        for idx, (client_proxy, eval_res) in enumerate(results):

            num_examples = eval_res.num_examples

            total_loss += (
                eval_res.loss * num_examples
            )

            total_examples += num_examples

            # =========================
            # CLIENT NAME
            # =========================
            client_name = eval_res.metrics.get(
                "client_name",
                f"client_{idx+1}"
            )

            # =========================
            # INIT STORAGE
            # =========================
            if client_name not in client_metrics:

                client_metrics[client_name] = {
                    "MAE": [],
                    "RMSE": [],
                    "NRMSE": []
                }

            # =========================
            # READ METRICS
            # =========================
            mae = float(
                eval_res.metrics.get(
                    "mae",
                    0.0
                )
            )

            rmse = float(
                eval_res.metrics.get(
                    "rmse",
                    0.0
                )
            )

            nrmse = float(
                eval_res.metrics.get(
                    "nrmse",
                    rmse
                )
            )

            # =========================
            # STORE
            # =========================
            client_metrics[client_name]["MAE"].append(
                mae
            )

            client_metrics[client_name]["RMSE"].append(
                rmse
            )

            client_metrics[client_name]["NRMSE"].append(
                nrmse
            )

            print(
                f"{client_name} -> "
                f"MAE={mae:.6f}, "
                f"RMSE={rmse:.6f}"
            )

        avg_loss = total_loss / total_examples

        print(
            f"📊 Round {rnd} Evaluation Loss: "
            f"{avg_loss:.6f}"
        )

        # =========================
        # FINAL ROUND SAVE
        # =========================
        if rnd == 30:

            print("\n📁 Saving final metrics...")

            final_results = {}

            for client_name in client_metrics:

                final_results[client_name] = {

                    "MAE": float(
                        np.mean(
                            client_metrics[client_name]["MAE"]
                        )
                    ),

                    "RMSE": float(
                        np.mean(
                            client_metrics[client_name]["RMSE"]
                        )
                    ),

                    "NRMSE": float(
                        np.mean(
                            client_metrics[client_name]["NRMSE"]
                        )
                    )
                }

            with open(
                f"{RESULT_DIR}/metrics_alpha_{ALPHA}.json",
                "w"
            ) as f:

                json.dump(
                    final_results,
                    f,
                    indent=4
                )

            print(
                f"✅ Metrics saved for alpha={ALPHA}"
            )

        return avg_loss, {
            "loss": avg_loss
        }


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # -------------------------
    # INITIAL MODEL
    # -------------------------
    base_model = TrafficPredictor(
        input_size=1,
        hidden_size=128,
        num_layers=3,
        output_size=1
    )

    initial_ndarrays = [

        v.cpu().detach().numpy()

        for v in base_model.state_dict().values()
    ]

    initial_parameters = ndarrays_to_parameters(
        initial_ndarrays
    )

    # -------------------------
    # STRATEGY
    # -------------------------
    strategy = FedCustom(

        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,

        fraction_evaluate=1.0,
        min_evaluate_clients=3,

        initial_parameters=initial_parameters,

        on_fit_config_fn=lambda rnd: {
            "round": rnd
        }
    )

    print(
        f"\n🚀 Starting FedAvg Server "
        f"for alpha={ALPHA}"
    )

    fl.server.start_server(

        server_address="localhost:8080",

        config=fl.server.ServerConfig(
            num_rounds=30
        ),

        strategy=strategy,
    )