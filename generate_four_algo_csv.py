import os
import json
import pandas as pd

# =========================
# ALL ALGORITHMS
# =========================
folders = {
    'FedAvg': 'results/fedavg_results',
    'FedProx': 'results/fedprox_results',
    'FedNova': 'results/fednova_results',
    'RAFedAvg': 'results/RA_Fedavg_results'
}

rows = []

# =========================
# LOAD JSON FILES
# =========================
for algo, folder in folders.items():

    if not os.path.exists(folder):
        continue

    files = sorted(os.listdir(folder))

    for file in files:

        if (
            file.startswith("metrics_alpha_")
            and file.endswith(".json")
        ):

            try:

                # =========================
                # EXTRACT ALPHA
                # =========================
                alpha = float(
                    file.replace(
                        "metrics_alpha_",
                        ""
                    ).replace(
                        ".json",
                        ""
                    )
                )

                file_path = os.path.join(
                    folder,
                    file
                )

                with open(file_path, "r") as f:
                    data = json.load(f)

                # =========================
                # CLIENT LOOP
                # =========================
                for client_key, metrics in data.items():

                    client_num = int(
                        client_key.split("_")[-1]
                    )

                    rows.append({

                        "Client": client_num,

                        "Split": alpha,

                        "Algorithm": algo,

                        "MAE": float(
                            metrics.get("MAE", 0)
                        ),

                        "RMSE": float(
                            metrics.get("RMSE", 0)
                        ),

                        "NRMSE": float(
                            metrics.get("NRMSE", 0)
                        )
                    })

            except Exception as e:

                print(
                    f"Skipping {file}: {e}"
                )

# =========================
# CREATE DATAFRAME
# =========================
df_results = pd.DataFrame(rows)

# =========================
# REMOVE DUPLICATES
# =========================
df_results = df_results.drop_duplicates(
    subset=[
        "Client",
        "Split",
        "Algorithm"
    ]
)

# =========================
# SORT VALUES
# =========================
df_results = df_results.sort_values(
    by=[
        "Split",
        "Algorithm",
        "Client"
    ]
)

# =========================
# SAVE FINAL CSV
# =========================
os.makedirs(
    "results",
    exist_ok=True
)

output_path = "results/results_all_four.csv"

df_results.to_csv(
    output_path,
    index=False
)

print(
    f"\n✅ CSV file created successfully!"
)

print(
    f"Saved at: {output_path}"
)

print(
    f"Total rows: {len(df_results)}"
)