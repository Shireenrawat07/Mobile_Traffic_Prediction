import flwr as fl
import numpy as np
import torch
import pandas as pd
import os
import json
import sys


class FedNovaStrategy(fl.server.strategy.FedAvg):

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

        self.round_data = []
        self.all_data = []

        self.global_weights = None
        self.final_client_metrics = {}

    # =========================
    # AGGREGATE FIT
    # =========================
    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        weights_list = []
        taus = []
        sizes = []
        client_names = []
        losses = []

        for client, fit_res in results:

            weights = fl.common.parameters_to_ndarrays(
                fit_res.parameters
            )

            tau = fit_res.metrics.get("tau", 1.0)
            loss = fit_res.metrics.get("loss", 1.0)
            n_k = fit_res.num_examples

            client_name = fit_res.metrics.get(
                "client_name",
                client.cid
            )

            weights_list.append(weights)
            taus.append(tau)
            sizes.append(n_k)
            client_names.append(client_name)
            losses.append(loss)

        total_samples = sum(sizes)

        # =========================
        # FEDNOVA AGGREGATION
        # =========================
        aggregated = None

        for i, weights in enumerate(weights_list):

            alpha = sizes[i] / (total_samples * taus[i])

            if aggregated is None:
                aggregated = [w * alpha for w in weights]
            else:
                aggregated = [
                    a + w * alpha
                    for a, w in zip(aggregated, weights)
                ]

        self.global_weights = aggregated

        # =========================
        # STORE ROUND DATA
        # =========================
        self.round_data = []

        for i in range(len(client_names)):

            self.round_data.append([
                self.split_value,
                client_names[i],
                losses[i],
                taus[i],
                sizes[i],
                None,
                None,
                None
            ])

        return (
            fl.common.ndarrays_to_parameters(aggregated),
            {}
        )

    # =========================
    # AGGREGATE EVALUATE
    # =========================
    def aggregate_evaluate(self, rnd, results, failures):

        total_loss = 0
        total_rmse = 0
        total_n = 0

        for client, res in results:

            n = res.num_examples
            loss = res.loss
            rmse = res.metrics.get("rmse", 0)

            total_loss += loss * n
            total_rmse += rmse * n
            total_n += n

        avg_loss = total_loss / total_n
        avg_rmse = total_rmse / total_n

        print(f"Round {rnd}: RMSE={avg_rmse:.6f}")

        # =========================
        # STORE METRICS
        # =========================
        for i, (client, res) in enumerate(results):

            rmse = res.metrics.get("rmse", 0)
            mae = res.metrics.get("mae", 0)

            self.round_data[i][5] = res.loss
            self.round_data[i][6] = rmse
            self.round_data[i][7] = mae

            self.all_data.append(self.round_data[i])

            if rnd == self.total_rounds:

                client_name = self.round_data[i][1]
                client_id = int(client_name.replace("Client_", ""))

                self.final_client_metrics[f"client_{client_id}"] = {
                    "MAE": float(mae),
                    "RMSE": float(rmse),
                    "NRMSE": float(rmse)
                }

        # =========================
        # SAVE RESULTS (FIXED ONLY HERE)
        # =========================
        if rnd == self.total_rounds:

            os.makedirs("results/fednova_results", exist_ok=True)

            df = pd.DataFrame(
                self.all_data,
                columns=[
                    "split",
                    "client_name",
                    "loss",
                    "tau",
                    "samples",
                    "global_loss",
                    "rmse",
                    "mae"
                ]
            )

            df_client = (
                df.groupby(["split", "client_name"])
                .mean()
                .reset_index()
            )

            df_client.columns = [
                "Split",
                "Client",
                "Loss",
                "Tau",
                "Samples",
                "Global_Loss",
                "RMSE",
                "MAE"
            ]

            df_client["Client"] = (
                df_client["Client"]
                .str.replace("Client_", "")
                .astype(int)
            )

            df_client["Algorithm"] = "FedNova"
            df_client["NRMSE"] = df_client["RMSE"]

            # =========================
            # ONLY FIX: SAFE APPEND CSV
            # =========================
            df_client["Alpha"] = self.split_value

            file_path = "results/fednova_results/FedNova_ClientWise.csv"
            file_exists = os.path.isfile(file_path)

            df_client.to_csv(
                file_path,
                mode="a",
                header=not file_exists,
                index=False
            )

            # =========================
            # JSON SAVE
            # =========================
            with open(
                f"results/fednova_results/metrics_alpha_{self.split_value}.json",
                "w"
            ) as f:
                json.dump(self.final_client_metrics, f, indent=4)

            # =========================
            # MODEL SAVE
            # =========================
            model_weights = {
                f"layer_{i}": torch.tensor(w)
                for i, w in enumerate(self.global_weights)
            }

            torch.save(
                model_weights,
                f"results/fednova_results/model_alpha_{self.split_value}.pth"
            )

            print(f"Saved results for alpha={self.split_value}")

        return avg_loss, {"rmse": avg_rmse}


# =========================
# MAIN
# =========================
def main():

    split_value = float(sys.argv[1])

    strategy = FedNovaStrategy(
        split_value=split_value,
        total_rounds=30,
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        fraction_evaluate=1.0,
        min_evaluate_clients=3
    )

    print(f"FedNova server started for alpha={split_value}")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=30)
    )


if __name__ == "__main__":
    main()