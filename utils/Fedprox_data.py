# utils/fedprox_data.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# PyTorch Dataset for LSTM
# -----------------------------
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        # LSTM expects (batch_size, seq_len=1, features)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# Load city CSV
# -----------------------------
def load_city_csv(filepath):
    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by=["time"])
    df["traffic"] = df["down"] + df["up"]

    # Remove NaNs and negative values
    df.dropna(inplace=True)
    df = df[(df["rnti_count"] >= 0) & (df["rb_down"] >= 0) & (df["rb_up"] >= 0)]

    X = df[["rnti_count", "mcs_down", "mcs_up", "rb_down", "rb_up"]].values
    y = df[["traffic"]].values

    # Scale features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    return X_scaled, y_scaled

# -----------------------------
# Dirichlet split for non-IID
# -----------------------------
def dirichlet_split(data_list, alpha=0.5):
    all_X = np.concatenate([d[0] for d in data_list], axis=0)
    all_y = np.concatenate([d[1] for d in data_list], axis=0)

    num_clients = len(data_list)
    n_samples = len(all_X)

    # Draw proportions and normalize
    proportions = np.random.dirichlet([alpha]*num_clients)
    proportions = np.maximum(proportions, 0.01)
    proportions = proportions / proportions.sum()

    client_data = []
    start = 0
    for i, p in enumerate(proportions):
        end = start + max(1, int(p * n_samples))
        if i == num_clients - 1:
            end = n_samples
        client_data.append((all_X[start:end], all_y[start:end]))
        start = end

    return client_data

# -----------------------------
# Create PyTorch DataLoaders per client with balancing
# -----------------------------
def get_client_loaders(alpha=0.5, batch_size=32, max_samples_per_client=2000):
    # Load city datasets
    elborn_X, elborn_y = load_city_csv("dataset/ElBorn.csv")
    lescorts_X, lescorts_y = load_city_csv("dataset/LesCorts.csv")
    poblesec_X, poblesec_y = load_city_csv("dataset/PobleSec.csv")

    raw_data = [(elborn_X, elborn_y), (lescorts_X, lescorts_y), (poblesec_X, poblesec_y)]
    client_data = dirichlet_split(raw_data, alpha)

    # Balance clients
    client_data_balanced = []
    for X, y in client_data:
        if len(X) > max_samples_per_client:
            idx = np.random.choice(len(X), max_samples_per_client, replace=False)
            X, y = X[idx], y[idx]
        client_data_balanced.append((X, y))

    client_loaders = []
    for X, y in client_data_balanced:
        if len(X) == 0:
            print("Warning: Client received 0 samples, skipping")
            continue

        dataset = TrafficDataset(X, y)

        train_size = max(1, int(0.8 * len(dataset)))
        test_size = len(dataset) - train_size
        if test_size == 0:
            test_size = 1
            train_size = len(dataset) - 1

        train_set, test_set = random_split(dataset, [train_size, test_size])
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        testloader  = DataLoader(test_set, batch_size=batch_size)
        client_loaders.append((trainloader, testloader))

    return client_loaders