import pandas as pd
import matplotlib.pyplot as plt
import os

# ===========================
# LOAD CSV
# ===========================
df = pd.read_csv(
    "results/results_all_four.csv"
)

# ===========================
# CLEAN COLUMN NAMES
# ===========================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

# ===========================
# VALIDATION
# ===========================
required_cols = {
    'split',
    'algorithm',
    'rmse'
}

if not required_cols.issubset(df.columns):

    raise Exception(
        f"CSV missing columns: "
        f"{required_cols - set(df.columns)}"
    )

# ===========================
# CLEAN DATA
# ===========================
df = df[
    ['client', 'algorithm', 'split', 'rmse']
].dropna()

# ===========================
# REMOVE DUPLICATES
# ===========================
df = df.drop_duplicates()

# ===========================
# SORT
# ===========================
df = df.sort_values(
    by=['split', 'algorithm']
)

# ===========================
# SCALE RMSE FOR PLOT
# ===========================
df['rmse_scaled'] = (
    df['rmse'] * 100
)

# ===========================
# OVERALL PERFORMANCE
# ===========================
algo_performance = (

    df.groupby('algorithm')['rmse']
    .mean()
    .reset_index()
    .sort_values('rmse')
)

print("\n===== OVERALL PERFORMANCE =====\n")

print(
    algo_performance.to_string(index=False)
)

best_algo = algo_performance.iloc[0]

print(
    f"\n🏆 BEST OVERALL: "
    f"{best_algo['algorithm']}"
)

print(
    f"👉 Avg RMSE: "
    f"{best_algo['rmse']:.6f}"
)

# ===========================
# DETAILED SUMMARY
# ===========================
summary = (

    df.groupby(
        ['split', 'algorithm']
    )['rmse']

    .agg(['mean', 'std'])

    .reset_index()
)

print("\n===== DETAILED SUMMARY =====\n")

print(summary.to_string(index=False))

# ===========================
# BEST PER SPLIT
# ===========================
print(
    "\n===== BEST PER SPLIT =====\n"
)

mean_df = (

    df.groupby(
        ['split', 'algorithm']
    )['rmse']

    .mean()

    .reset_index()
)

best_per_split = mean_df.loc[
    mean_df.groupby('split')['rmse'].idxmin()
]

best_per_split = best_per_split.sort_values(
    'split'
)

print(
    best_per_split.to_string(index=False)
)

print("\n📊 INTERPRETATION:")

for _, row in best_per_split.iterrows():

    print(
        f"Split {row['split']} "
        f"→ {row['algorithm']} best "
        f"(Avg RMSE={row['rmse']:.5f})"
    )

# ===========================
# SAVE SUMMARY CSV
# ===========================
os.makedirs(
    "results",
    exist_ok=True
)

csv_path = (
    "results/four_algo_split_rmse.csv"
)

mean_df.to_csv(
    csv_path,
    index=False
)

print(
    f"\nCSV saved at: {csv_path}"
)

# ===========================
# PLOT
# ===========================
os.makedirs(
    "plots",
    exist_ok=True
)

plt.figure(figsize=(10, 6))

algorithms = sorted(
    df['algorithm'].unique()
)

for algo in algorithms:

    temp = df[
        df['algorithm'] == algo
    ]

    grouped = (

        temp.groupby('split')['rmse_scaled']
        .mean()
        .sort_index()
    )

    if algo.lower() == "rafedavg":

        plt.plot(
            grouped.index,
            grouped.values,
            marker='o',
            linewidth=2.5,
            label=algo
        )

    else:

        plt.plot(
            grouped.index,
            grouped.values,
            marker='o',
            linestyle='--',
            alpha=0.8,
            label=algo
        )

# ===========================
# AXIS
# ===========================
plt.xlabel("Split")
plt.ylabel("RMSE ×100")

plt.title(
    "Split vs RMSE Comparison"
)

plt.grid(True)

plt.legend()

# Better auto scaling
ymin = df['rmse_scaled'].min()
ymax = df['rmse_scaled'].max()

plt.ylim(
    ymin - 0.2,
    ymax + 0.2
)

plot_path = (
    "plots/four_algo_split_vs_rmse.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches='tight'
)

print(
    f"Plot saved at: {plot_path}"
)

plt.show()