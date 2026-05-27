import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Folder containing CSV result files
RESULTS_FOLDER = "results/Models_results"

# Folder to save plots
PLOTS_FOLDER = "plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Lists to store values
model_names = []
mae_values = []
nrmse_values = []

# Read all CSV files
for file in sorted(os.listdir(RESULTS_FOLDER)):

    if file.endswith(".csv"):

        file_path = os.path.join(RESULTS_FOLDER, file)

        try:
            # Read CSV
            df = pd.read_csv(file_path)

            # Extract values
            model_name = str(df["Model"].iloc[0]).strip()
            mae = float(df["MAE"].iloc[0])
            nrmse = float(df["NRMSE"].iloc[0])

            # Store values
            model_names.append(model_name)
            mae_values.append(mae)
            nrmse_values.append(nrmse)

        except Exception as e:
            print(f"Error reading {file}: {e}")

# X locations
x = np.arange(len(model_names))
width = 0.35

# Figure size like your example
plt.figure(figsize=(10, 6))

# Bars
bars1 = plt.bar(x - width/2, mae_values, width, label='MAE')
bars2 = plt.bar(x + width/2, nrmse_values, width, label='NRMSE')

# Value labels on bars
for bar in bars1:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.0002,
        str(height),
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )

for bar in bars2:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.0002,
        str(height),
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )

# Labels and title
plt.xlabel("Models", fontsize=13)
plt.ylabel("Error Metrics", fontsize=13)
plt.title("Performance Comparison of Deep Models", fontsize=16)

# X-axis labels
plt.xticks(x, model_names, fontsize=11)

# Legend
plt.legend(fontsize=11)

# Grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Reduce side spacing
plt.margins(x=0.05)

# Tight layout
plt.tight_layout()

# Save graph
save_path = os.path.join(PLOTS_FOLDER, "model_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show graph
plt.show()

print(f"Graph saved at: {save_path}")