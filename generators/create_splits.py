import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from utils.data_preprocess import load_real_traffic_data, prepare_sequences

# =========================
# CONFIG
# =========================
SEQ_LEN = 10
NUM_CLIENTS = 3
ALPHA = 0.3# change for experiments: 0.1 / 0.5 / 1.0

DATA_FILES = [
    "Dataset/ElBorn.csv",
    "Dataset/LesCorts.csv",
    "Dataset/PobleSec.csv"
]

SAVE_DIR = f"splits_alpha_{ALPHA}"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# STEP 1: LOAD ALL DATA (NO SCALING YET)
# =========================
all_series = []

for file in DATA_FILES:
    series = load_real_traffic_data(file, "down")
    all_series.append(series)

all_series = np.concatenate(all_series)
all_series = all_series.reshape(-1, 1)

# =========================
# STEP 2: GLOBAL SCALING (IMPORTANT FIX)
# =========================
scaler = MinMaxScaler()
all_series = scaler.fit_transform(all_series)

# =========================
# STEP 3: CREATE SEQUENCES
# =========================
X, y = prepare_sequences(all_series, SEQ_LEN)

if X.ndim == 2:
    X = X.reshape((X.shape[0], X.shape[1], 1))

N = len(X)

# =========================
# STEP 4: DIRICHLET SPLIT (HETEROGENEITY)
# =========================
MIN_SAMPLES = 500   # IMPORTANT FIX

while True:

    proportions = np.random.dirichlet([ALPHA] * NUM_CLIENTS)
    proportions = proportions * N

    # enforce minimum size
    proportions = np.maximum(proportions, MIN_SAMPLES)

    # normalize back to N
    proportions = proportions / np.sum(proportions) * N
    proportions = proportions.astype(int)

    # fix rounding
    proportions[-1] = N - np.sum(proportions[:-1])

    # safety check
    if min(proportions) > MIN_SAMPLES:
        break

# =========================
# STEP 5: SHUFFLE INDICES
# =========================
indices = np.random.permutation(N)

start = 0

# =========================
# STEP 6: SAVE CLIENT DATA
# =========================
for i, size in enumerate(proportions):

    idx = indices[start:start + size]

    torch.save(
        {
            "X": X[idx],
            "y": y[idx]
        },
        f"{SAVE_DIR}/client_{i+1}.pt"
    )

    print(f"Client {i+1}: {len(idx)} samples")

    start += size

print(f"\n✅ Correct splits saved in {SAVE_DIR}")