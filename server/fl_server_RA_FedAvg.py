
# 1. Adaptive Heterogeneity Weighting
# 2. Hybrid RA-FedAvg + FedAvg Aggregation
# =========================================================

import os
import sys
import json
import numpy as np
import flwr as fl
import pandas as pd

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters
)

# =========================================================
# ROOT PATH
# =========================================================
ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.append(ROOT)

# =========================================================
# RA-FEDAVG STRATEGY
# =========================================================
class RAFedAvg(fl.server.strategy.FedAvg):

    def __init__(
        self,
        split_value=0.1,
        total_rounds=30,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.split_value = split_value
        self.total_rounds = total_rounds

        self.all_data = []
        self.final_client_metrics = {}

    # =====================================================
    # NORMALIZATION
    # =====================================================
    def normalize(self, values):

        values = np.array(
            values,
            dtype=np.float64
        )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        if np.max(values) - np.min(values) < 1e-8:
            return np.ones_like(values)

        return (
            (values - np.min(values))
            /
            (np.max(values) - np.min(values) + 1e-8)
        )

    # =====================================================
    # AGGREGATE FIT
    # =====================================================
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):

        if not results:
            return None, {}

        weights_list = []

        losses = []
        variances = []
        divergences = []
        sizes = []

        # =================================================
        # EXTRACT CLIENT DATA
        # =================================================
        for client, fit_res in results:

            weights = parameters_to_ndarrays(
                fit_res.parameters
            )

            weights_list.append(weights)

            losses.append(
                float(
                    fit_res.metrics.get(
                        "loss",
                        1.0
                    )
                )
            )

            variances.append(
                float(
                    fit_res.metrics.get(
                        "variance",
                        1.0
                    )
                )
            )

            divergences.append(
                float(
                    fit_res.metrics.get(
                        "divergence",
                        0.0
                    )
                )
            )

            sizes.append(
                fit_res.num_examples
            )

        # =================================================
        # NORMALIZE VALUES
        # =================================================
        loss_n = self.normalize(losses)

        var_n = self.normalize(variances)

        div_n = self.normalize(divergences)

        # =================================================
        # ADAPTIVE HETEROGENEITY FACTOR
        # =================================================
        heterogeneity_factor = (
            1.0 /
            (self.split_value + 1e-8)
        )

        lambda1 = 1.0 * heterogeneity_factor
        lambda2 = 0.5 * heterogeneity_factor
        lambda3 = 0.5 * heterogeneity_factor

        # =================================================
        # RELIABILITY SCORE
        # =================================================
        reliability_scores = []

        for i in range(len(weights_list)):

            score = (
                sizes[i]
                *
                np.exp(
                    -(
                        lambda1 * var_n[i]
                        +
                        lambda2 * loss_n[i]
                        +
                        lambda3 * div_n[i]
                    )
                )
            )

            reliability_scores.append(score)

        reliability_scores = np.clip(
            reliability_scores,
            1e-8,
            1e8
        )

        reliability_scores = np.array(
            reliability_scores
        )

        # =================================================
        # RA-FEDAVG WEIGHTS
        # =================================================
        ra_alpha = (
            reliability_scores
            /
            (
                np.sum(reliability_scores)
                + 1e-8
            )
        )

        # =================================================
        # FEDAVG WEIGHTS
        # =================================================
        fedavg_alpha = (
            np.array(sizes)
            /
            (
                np.sum(sizes)
                + 1e-8
            )
        )

        # =================================================
        # HYBRID AGGREGATION
        # =================================================
        beta = (
            1.0
            /
            (1.0 + self.split_value)
        )

        alpha = (
            beta * ra_alpha
            +
            (1 - beta) * fedavg_alpha
        )

        alpha = (
            alpha
            /
            (
                np.sum(alpha)
                + 1e-8
            )
        )

        # =================================================
        # PRINT INFO
        # =================================================
        print(f"\nRound {rnd}")

        print(
            f"Heterogeneity Factor: "
            f"{heterogeneity_factor:.4f}"
        )

        print(
            f"Aggregation Weights: "
            f"{np.round(alpha, 4)}"
        )

        # =================================================
        # MODEL AGGREGATION
        # =================================================
        aggregated = None

        for i, weights in enumerate(weights_list):

            if aggregated is None:

                aggregated = [
                    layer * alpha[i]
                    for layer in weights
                ]

            else:

                aggregated = [
                    a + layer * alpha[i]
                    for a, layer in zip(
                        aggregated,
                        weights
                    )
                ]

        return (
            ndarrays_to_parameters(
                aggregated
            ),
            {}
        )

    # =====================================================
    # AGGREGATE EVALUATE
    # =====================================================
    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures
    ):

        if not results:
            return 0.0, {}

        total_loss = 0.0
        total_n = 0

        print(f"\n--- Round {rnd} Evaluation ---")

        for _, evaluate_res in results:

            n = evaluate_res.num_examples

            total_loss += (
                evaluate_res.loss * n
            )

            total_n += n

            metrics = evaluate_res.metrics

            client_name = metrics.get(
                "client_name",
                "client"
            )

            mae = float(
                metrics.get("mae", 0.0)
            )

            rmse = float(
                metrics.get("rmse", 0.0)
            )

            nrmse = float(
                metrics.get("nrmse", rmse)
            )

            print(
                f"{client_name} -> "
                f"MAE={mae:.6f}, "
                f"RMSE={rmse:.6f}"
            )

            self.all_data.append([
                self.split_value,
                client_name,
                float(evaluate_res.loss),
                mae,
                rmse,
                nrmse,
                "RAFedAvg",
                rnd
            ])

            if rnd == self.total_rounds:

                self.final_client_metrics[
                    client_name
                ] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "NRMSE": nrmse
                }

        avg_loss = (
            total_loss
            /
            (total_n + 1e-8)
        )

        print(
            f"Round {rnd} "
            f"Loss: {avg_loss:.6f}"
        )

        # =================================================
        # SAVE RESULTS
        # =================================================
        if rnd == self.total_rounds:

            os.makedirs(
                "results/RA_Fedavg_results",
                exist_ok=True
            )

            df = pd.DataFrame(
                self.all_data,
                columns=[
                    "split",
                    "client_name",
                    "loss",
                    "mae",
                    "rmse",
                    "nrmse",
                    "algorithm",
                    "round"
                ]
            )

            df.to_csv(
                "results/RA_Fedavg_results/RAFedAvg_clientwise.csv",
                index=False
            )

            with open(
                f"results/RA_Fedavg_results/metrics_alpha_{self.split_value}.json",
                "w"
            ) as f:

                json.dump(
                    self.final_client_metrics,
                    f,
                    indent=4
                )

            print("\n✅ Results Saved")

        return avg_loss, {
            "loss": float(avg_loss)
        }

# =========================================================
# MAIN
# =========================================================
def main():

    if len(sys.argv) < 2:

        print(
            "Usage: python "
            "fl_server_RA_FedAvg.py <alpha>"
        )

        sys.exit(1)

    alpha = float(sys.argv[1])

    strategy = RAFedAvg(

        split_value=alpha,
        total_rounds=30,

        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,

        fraction_evaluate=1.0,
        min_evaluate_clients=3
    )

    print(
        f"\nStarting RA-FedAvg "
        f"alpha={alpha}"
    )

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=30
        ),

        strategy=strategy
    )

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()