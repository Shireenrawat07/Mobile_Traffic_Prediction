import pandas as pd
import matplotlib.pyplot as plt
import os

# ===========================
# Step 1: Load CSV
# ===========================
df = pd.read_csv('results_fedprox_fedavg.csv')

# Quick check
print("First few rows of CSV:")
print(df.head())

# ===========================
# Step 2: Define parameters
# ===========================
algorithms = ['FedAvg', 'FedProx']
metric = 'RMSE'
splits = sorted(df['Split'].unique())

# ===========================
# Step 3: Compute mean RMSE per split
# ===========================
summary = {algo: [] for algo in algorithms}

for split in splits:
    for algo in algorithms:
        mean_val = df[(df['Algorithm']==algo) & (df['Split']==split)][metric].mean()
        summary[algo].append(mean_val)

# ===========================
# Step 4: Compute % improvement FedProx vs FedAvg
# ===========================
print("\n=== FedProx vs FedAvg RMSE Improvement ===")
for i, split in enumerate(splits):
    diff = summary['FedAvg'][i] - summary['FedProx'][i]
    perc = diff / summary['FedAvg'][i] * 100
    better = "FedProx" if diff > 0 else "FedAvg"
    print(f"Split {split}: {better} better by {abs(diff):.4f} ({abs(perc):.2f}%)")

# ===========================
# Step 5: Plot RMSE vs Dirichlet split
# ===========================
plt.figure(figsize=(8,5))
plt.plot(splits, summary['FedAvg'], marker='o', label='FedAvg RMSE')
plt.plot(splits, summary['FedProx'], marker='x', label='FedProx RMSE')

plt.xlabel('Dirichlet Split α')
plt.ylabel('RMSE')
plt.title('FedAvg vs FedProx: RMSE Comparison')
plt.xticks(splits)  # show only 0.1, 0.5, 1.0
plt.grid(True)
plt.legend()

# ===========================
# Step 6: Save the plot
# ===========================
# Create 'plots' folder if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Save figure
plot_path = os.path.join('plots', 'rmse_fedavg_fedprox.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully at '{plot_path}'")

# Show the plot
plt.show()