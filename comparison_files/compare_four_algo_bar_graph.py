import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===========================
# Load SINGLE CSV
# ===========================
df = pd.read_csv("results/final_algo_split_rmse.csv")

# ===========================
# Clean column names
# ===========================
df.columns = df.columns.str.strip().str.lower()

# ===========================
# Validate columns
# ===========================
required = {'algorithm', 'split', 'rmse'}
if not required.issubset(df.columns):
    raise Exception(f"Missing columns: {required - set(df.columns)}")

# ===========================
# Pivot table
# ===========================
pivot = df.pivot_table(index='split', columns='algorithm', values='rmse', aggfunc='mean')
pivot = pivot.sort_index()

# ===========================
# Best algorithm per split
# ===========================
best_per_split = pivot.idxmin(axis=1)

print("\n===== BEST PER SPLIT =====\n")
print(best_per_split)

# ===========================
# Plot BAR GRAPH
# ===========================
os.makedirs("plots", exist_ok=True)

x = np.arange(len(pivot.index))
width = 0.2

plt.figure(figsize=(12, 6))

algorithms = pivot.columns.tolist()

for i, algo in enumerate(algorithms):
    values = pivot[algo].values
    
    bars = plt.bar(x + i*width, values, width, label=algo)

    # Highlight RAFedAvg wins
    if algo == "RAFedAvg":
        for j, val in enumerate(values):
            if best_per_split.iloc[j] == "RAFedAvg":
                bars[j].set_edgecolor('black')
                bars[j].set_linewidth(2)

# X-axis labels
plt.xticks(x + width, pivot.index)

plt.xlabel("Alpha (Split)")
plt.ylabel("RMSE")
plt.title("Algorithm Comparison across Alpha")
plt.legend()
plt.grid(axis='y')

plot_path = "plots/bargraph_algo_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print(f"\n📊 Plot saved at: {plot_path}")

plt.show()