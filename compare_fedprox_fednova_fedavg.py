import pandas as pd
import matplotlib.pyplot as plt
import os

# ===========================
# Step 1: Load CSV
# ===========================
df = pd.read_csv('results_all_three.csv')
print(df.head())

# ===========================
# Step 2: Define parameters
# ===========================
algorithms = ['FedAvg', 'FedProx', 'FedNova']
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
# Step 4: Compute % improvement vs FedAvg
# ===========================
print("\n=== Improvement vs FedAvg ===")
for i, split in enumerate(splits):
    print(f"\nSplit {split}:")
    for algo in ['FedProx', 'FedNova']:
        diff = summary['FedAvg'][i] - summary[algo][i]
        perc = diff / summary['FedAvg'][i] * 100
        better = algo if diff > 0 else 'FedAvg'
        print(f"{algo} vs FedAvg: {better} better by {abs(diff):.4f} ({abs(perc):.2f}%)")

# ===========================
# Step 5: Plot RMSE vs Dirichlet split
# ===========================
plt.figure(figsize=(8,5))
markers = ['o', 'x', 's']

for i, algo in enumerate(algorithms):
    plt.plot(splits, summary[algo], marker=markers[i], label=f'{algo} RMSE')

plt.xlabel('Dirichlet Split α')
plt.ylabel('RMSE')
plt.title('RMSE Comparison: FedAvg vs FedProx vs FedNova')
plt.xticks(splits)
plt.grid(True)
plt.legend()

# ===========================
# Step 6: Save the plot
# ===========================
os.makedirs('plots', exist_ok=True)
plot_path = os.path.join('plots', 'rmse_all_three.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully at '{plot_path}'")

plt.show()