import os
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.data_preprocess import load_real_traffic_data, prepare_sequences

SEQ_LEN = 10
NUM_CLIENTS = 3
ALPHA = 0.05 # change to 0.1 / 0.5 / 1.0

DATA_FILES = [
    "Dataset/ElBorn.csv",
    "Dataset/LesCorts.csv",
    "Dataset/PobleSec.csv"
]

SAVE_DIR = f"splits_alpha_{ALPHA}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- Combine all data ----
X_all, y_all = [], []

for file in DATA_FILES:
    series = load_real_traffic_data(file, "down")

    scaler = MinMaxScaler()
    series = scaler.fit_transform(series)

    X, y = prepare_sequences(series, SEQ_LEN)

    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    X_all.append(X)
    y_all.append(y)

X = np.concatenate(X_all)
y = np.concatenate(y_all)

# ---- Dirichlet split ----
# ---- Dirichlet split (FIXED) ----
N = len(X)

while True:
    proportions = np.random.dirichlet([ALPHA] * NUM_CLIENTS)
    proportions = (proportions * N).astype(int)

    # fix sum
    proportions[-1] = N - sum(proportions[:-1])

    # ✅ ensure NO client gets 0 data
    if min(proportions) > 0:
        break

indices = np.random.permutation(N)

start = 0
for i, p in enumerate(proportions):
    idx = indices[start:start + p]

    torch.save({
        "X": X[idx],
        "y": y[idx]
    }, f"{SAVE_DIR}/client_{i+1}.pt")

    print(f"Client {i+1}: {len(idx)} samples")

    start += p

print(f"\n✅ Splits saved in {SAVE_DIR}")