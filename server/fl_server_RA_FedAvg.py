# fl_server_RA_FedProx_Hybrid.py

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import flwr as fl

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT not in sys.path:
    sys.path.append(ROOT)


class RAFedProxHybrid(fl.server.strategy.FedAvg):

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

        self.global_weights = None

        self.round_data = []
        self.all_data = []

        self.final_client_metrics = {}

    # =====================================
    # NORMALIZATION
    # =====================================
    def normalize(self, values):

        values = np.array(values, dtype=np.float64)

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        vmin = np.min(values)
        vmax = np.max(values)

        if abs(vmax - vmin) < 1e-8:
            return np.ones_like(values) * 0.5

        return (
            (values - vmin)
            / (vmax - vmin + 1e-8)
        )

    # =====================================
    # FIT AGGREGATION
    # =====================================
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):

        if len(results) == 0:
            return None, {}

        weights_list = []

        losses = []
        variances = []
        divergences = []

        n_clients = []
        client_names = []

        # =====================================
        # COLLECT CLIENT INFO
        # =====================================
        for client, fit_res in results:

            weights = fl.common.parameters_to_ndarrays(
                fit_res.parameters
            )

            loss = float(
                fit_res.metrics.get("loss", 1.0)
            )

            variance = float(
                fit_res.metrics.get("variance", 1.0)
            )

            divergence = float(
                fit_res.metrics.get("divergence", 0.0)
            )

            client_name = fit_res.metrics.get(
                "client_name",
                client.cid
            )

            n_k = fit_res.num_examples

            if np.isnan(loss):
                loss = 1.0

            if np.isnan(variance):
                variance = 1.0

            if np.isnan(divergence):
                divergence = 0.0

            weights_list.append(weights)

            losses.append(loss)
            variances.append(variance)
            divergences.append(divergence)

            n_clients.append(n_k)
            client_names.append(client_name)

        # =====================================
        # NORMALIZE
        # =====================================
        loss_norm = self.normalize(losses)
        var_norm = self.normalize(variances)
        div_norm = self.normalize(divergences)

        # =====================================
        # ADAPTIVE LAMBDA
        # =====================================
        heterogeneity_factor = 1.0 / max(
            self.split_value,
            0.1
        )

        lambda1 = 0.8 * heterogeneity_factor
        lambda2 = 0.4 * heterogeneity_factor
        lambda3 = 0.4 * heterogeneity_factor

        # =====================================
        # RA-FedAvg RELIABILITY
        # =====================================
        R = []

        for i in range(len(results)):

            score = n_clients[i] * np.exp(

                -lambda1 * var_norm[i]
                -lambda2 * loss_norm[i]
                -lambda3 * div_norm[i]

            )

            R.append(score)

        R = np.array(R)

        R = np.clip(
            R,
            1e-6,
            1e6
        )

        ra_alpha = R / (
            np.sum(R) + 1e-8
        )

        # =====================================
        # FedProx/FedAvg Weight
        # =====================================
        fedprox_alpha = np.array(n_clients) / (
            np.sum(n_clients) + 1e-8
        )

        # =====================================
        # HYBRID FACTOR
        # =====================================
        beta = 1.0 / (
            1.0 + self.split_value
        )

        alpha = (

            beta * ra_alpha

            +

            (1 - beta) * fedprox_alpha

        )

        alpha = alpha / (
            np.sum(alpha) + 1e-8
        )

        print(f"\nRound {rnd}")
        print(f"Hybrid Beta = {beta:.4f}")
        print(f"Aggregation Weights = {alpha}")

        # =====================================
        # SAVE ROUND DATA
        # =====================================
        self.round_data = []

        for i in range(len(client_names)):

            self.round_data.append([

                self.split_value,
                client_names[i],
                losses[i],
                variances[i],
                divergences[i],
                R[i],
                alpha[i],
                None,
                None

            ])

        # =====================================
        # AGGREGATION
        # =====================================
        aggregated = None

        for i, weights in enumerate(weights_list):

            if aggregated is None:

                aggregated = [
                    w * alpha[i]
                    for w in weights
                ]

            else:

                aggregated = [

                    a + w * alpha[i]

                    for a, w in zip(
                        aggregated,
                        weights
                    )
                ]

        # =====================================
        # GLOBAL SMOOTHING
        # =====================================
        if self.global_weights is None:

            self.global_weights = aggregated

        else:

            gamma = 0.3

            self.global_weights = [

                (1 - gamma) * gw
                +
                gamma * aw

                for gw, aw in zip(
                    self.global_weights,
                    aggregated
                )
            ]

        return (

            fl.common.ndarrays_to_parameters(
                self.global_weights
            ),

            {}

        )

    # =====================================
    # EVALUATE
    # =====================================
    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures
    ):

        if len(results) == 0:
            return None, {}

        total_loss = 0.0
        total_rmse = 0.0
        total_examples = 0

        for i, (client, res) in enumerate(results):

            n = res.num_examples

            rmse = float(
                res.metrics.get("rmse", 0.0)
            )

            mae = float(
                res.metrics.get("mae", 0.0)
            )

            total_loss += res.loss * n
            total_rmse += rmse * n
            total_examples += n

            self.round_data[i][7] = rmse
            self.round_data[i][8] = mae

            self.all_data.append(
                self.round_data[i]
            )

            if rnd == self.total_rounds:

                cname = self.round_data[i][1]

                cid = int(
                    cname.replace(
                        "Client_",
                        ""
                    )
                )

                self.final_client_metrics[
                    f"client_{cid}"
                ] = {

                    "MAE": mae,
                    "RMSE": rmse,
                    "NRMSE": rmse

                }

        avg_loss = total_loss / (
            total_examples + 1e-8
        )

        avg_rmse = total_rmse / (
            total_examples + 1e-8
        )

        print(
            f"Round {rnd} RMSE: "
            f"{avg_rmse:.6f}"
        )

        # =====================================
        # SAVE RESULTS
        # =====================================
        if rnd == self.total_rounds:

            os.makedirs(
                "results/RA_Fedavg_results",
                exist_ok=True
            )

            df = pd.DataFrame(

                self.all_data,

                columns=[

                    "Split",
                    "Client",
                    "Loss",
                    "Variance",
                    "Divergence",
                    "R_k",
                    "Alpha",
                    "RMSE",
                    "MAE"

                ]
            )

            df.to_csv(

                "results/RA_Fedavg_results/clientwise.csv",

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

            torch.save(

                {

                    f"layer_{i}": torch.tensor(w)

                    for i, w in enumerate(
                        self.global_weights
                    )

                },

                f"results/RA_Fedavg_results/model_alpha_{self.split_value}.pth"

            )

            print("\n✅ Results Saved")

        return avg_loss, {
            "rmse": avg_rmse
        }


# =====================================
# MAIN
# =====================================
def main():

    if len(sys.argv) < 2:

        print(
            "Usage: python fl_server_RA_FedProx_Hybrid.py <alpha>"
        )

        sys.exit(1)

    split_value = float(sys.argv[1])

    strategy = RAFedProxHybrid(

        split_value=split_value,

        total_rounds=30,

        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,

        fraction_evaluate=1.0,
        min_evaluate_clients=3
    )

    print(
        f"\n🚀 Starting RA-FedProx Hybrid Server "
        f"(alpha={split_value})"
    )

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=30
        ),

        strategy=strategy
    )


if __name__ == "__main__":
    main()