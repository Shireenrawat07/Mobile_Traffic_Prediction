import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('default')

# ===========================
# Step 1: Load CSVs
# ===========================
fedavg = pd.read_csv('results/fedavg_results.csv')
simpleavg = pd.read_csv('results/simpleavg_results.csv')
medianavg = pd.read_csv('results/medianavg_results.csv')
import pandas as pd
import os

# ===========================
# Step 1: Load CSV files
# ===========================
fedavg = pd.read_csv('results/fedavg_results.csv')
simpleavg = pd.read_csv('results/simpleavg_results.csv')
medianavg = pd.read_csv('results/medianavg_results.csv')

# ===========================
# Step 2: Add Algorithm column
# ===========================
fedavg['Algorithm'] = 'FedAvg'
simpleavg['Algorithm'] = 'SimpleAvg'
medianavg['Algorithm'] = 'MedianAvg'

# ===========================
# Step 3: Combine all
# ===========================
combined_df = pd.concat([fedavg, simpleavg, medianavg], ignore_index=True)

# ===========================
# Step 4: Save combined CSV
# ===========================
os.makedirs('results', exist_ok=True)

combined_path = os.path.join('results', 'fedavg_simple_median.csv')
combined_df.to_csv(combined_path, index=False)

print(f"✅ Combined CSV saved at: {combined_path}")
# ===========================
# Step 3: Plot CLEAN graph
# ===========================
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/fedavg_simple_median.csv')

plt.figure(figsize=(10,6))

for algo in df['Algorithm'].unique():
    subset = df[df['Algorithm'] == algo]
    mean_curve = subset.groupby('Round')['Loss'].mean().reset_index()
    plt.plot(mean_curve['Round'], mean_curve['Loss'], label=algo)

plt.xlabel('Round')
plt.ylabel('Loss')
plt.title('FedAvg vs SimpleAvg vs MedianAvg')
plt.grid(True)
plt.legend()




# Fix Y-axis like your earlier graph
plt.ylim(0.002, 0.012)
plt.yticks([0.002, 0.004, 0.006, 0.008, 0.010, 0.012])

plt.grid(True)
plt.legend()

# ===========================
# Step 4: Save
# ===========================
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/fedavg_median_simple.png', dpi=300, bbox_inches='tight')

print("✅ Clean graph saved at plots/fedavg_median_simple.png")

plt.show()