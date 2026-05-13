import pandas as pd
import matplotlib.pyplot as plt
import os

# ===========================
# LOAD CSV
# ===========================
df = pd.read_csv("results/results_all_four.csv")

# ===========================
# CLEAN COLUMN NAMES
# ===========================
df.columns = df.columns.str.strip().str.lower()

# ===========================
# REQUIRED COLUMNS CHECK
# ===========================
required_cols = {'client', 'algorithm', 'split', 'rmse'}

if not required_cols.issubset(df.columns):
    raise Exception(f"Missing columns: {required_cols - set(df.columns)}")

# ===========================
# CLEAN DATA
# ===========================
df = df[['client', 'algorithm', 'split', 'rmse']].dropna()

# ===========================
# STEP 1: CLIENT → ALGORITHM AVG PER SPLIT
# ===========================
algo_split_df = (
    df.groupby(['split', 'algorithm'])['rmse']
    .mean()
    .reset_index()
)

# ===========================
# STEP 2: OVERALL PERFORMANCE (ALL SPLITS)
# ===========================
overall = (
    algo_split_df.groupby('algorithm')['rmse']
    .mean()
    .reset_index()
    .sort_values('rmse')
)

print("\n===== OVERALL ALGORITHM PERFORMANCE =====\n")
print(overall.to_string(index=False))

best_algo = overall.iloc[0]
print(f"\n🏆 BEST OVERALL: {best_algo['algorithm']} | RMSE: {best_algo['rmse']:.6f}")

# ===========================
# STEP 3: BEST PER SPLIT
# ===========================
best_per_split = algo_split_df.loc[
    algo_split_df.groupby('split')['rmse'].idxmin()
].sort_values('split')

print("\n===== BEST PER SPLIT =====\n")
print(best_per_split.to_string(index=False))

for _, row in best_per_split.iterrows():
    print(f"Split {row['split']} → {row['algorithm']} best (RMSE={row['rmse']:.5f})")

# ===========================
# STEP 4: SAVE CLEAN SUMMARY
# ===========================
os.makedirs("results", exist_ok=True)
algo_split_df.to_csv("results/final_algo_split_rmse.csv", index=False)

# ===========================
# STEP 5: PLOT (CLEAN + FAIR)
# ===========================
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(10, 6))

algos = algo_split_df['algorithm'].unique()

for algo in algos:
    temp = algo_split_df[algo_split_df['algorithm'] == algo]
    temp = temp.sort_values('split')

    plt.plot(
        temp['split'],
        temp['rmse'],
        marker='o',
        label=algo,
        linewidth=2
    )

plt.xlabel("Alpha (Split)")
plt.ylabel("Avg RMSE")
plt.title("Algorithm Comparison across Data Heterogeneity (Alpha)")
plt.grid(True)
plt.legend()

plt.savefig("plots/four_algo_final_comparison.png", dpi=300, bbox_inches='tight')
plt.show() 