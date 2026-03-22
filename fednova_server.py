# fl_server_fednova.py
import flwr as fl
import numpy as np
from typing import List, Tuple


class FedNova(fl.server.strategy.FedAvg):
    """Custom FedNova Strategy"""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures,
    ):
        if not results:
            return None, {}

        print(f"\n🌐 Aggregating round {server_round} using FedNova")

        # ---- Extract data ----
        weights = []
        num_examples = []
        taus = []

        for _, fit_res in results:
            w = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weights.append(w)

            n = fit_res.num_examples
            num_examples.append(n)

            # Approximate tau = local steps (epochs × batches)
            tau = fit_res.metrics.get("tau", 1)
            taus.append(tau)

        total_examples = sum(num_examples)

        # ---- Compute FedNova update ----
        aggregated = []

        for layer_i in range(len(weights[0])):
            layer_sum = 0

            for k in range(len(weights)):
                nk = num_examples[k]
                tauk = taus[k]

                # Normalized contribution
                layer_sum += (nk / total_examples) * (weights[k][layer_i] / tauk)

            aggregated.append(layer_sum)

        parameters = fl.common.ndarrays_to_parameters(aggregated)

        return parameters, {}
    

if __name__ == "__main__":
    strategy = FedNova(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

    print("🚀 Starting FedNova Server...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )