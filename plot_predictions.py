import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import load_model_from_checkpoint
from clients.client import prepare_sequences
import numpy as np

# ========== SETTINGS ==========
MODEL_PATH = "global_model.pth"
DATA_PATH = "data/traffic_data.csv"
SEQ_LEN = 10
COLUMN = "down"   # change to "up" for uplink
WINDOW = 300      # number of points to visualize
# ==============================

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)
if COLUMN not in df.columns:
    raise ValueError(f"Column '{COLUMN}' not found. Available columns: {list(df.columns)}")

# Convert to Mbps if large numeric values
if "down" in df.columns:
    df["down"] = df["down"] / 1e6
if "up" in df.columns:
    df["up"] = df["up"] / 1e6

if "timestamp" in df.columns:
    df = df.sort_values("timestamp")

series = df[COLUMN].values.reshape(-1, 1)

# --- Normalize ---
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# --- Prepare sequences ---
X_np, y_np = prepare_sequences(series_scaled, SEQ_LEN)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# --- Load model adaptively ---
model = load_model_from_checkpoint(MODEL_PATH, input_size=1, output_size=1)
model.eval()

# --- Predict ---
with torch.no_grad():
    y_pred = model(X).numpy()

# --- Inverse transform ---
y_true = scaler.inverse_transform(y.numpy())
y_pred = scaler.inverse_transform(y_pred)

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(y_true[:WINDOW], label=f'Actual {COLUMN.capitalize()}', color='blue', linewidth=1)
plt.plot(y_pred[:WINDOW], label=f'Predicted {COLUMN.capitalize()}', color='orange', linewidth=1.5)
plt.title(f"Actual vs Predicted {COLUMN.capitalize()} Traffic (First {WINDOW} Samples)")
plt.xlabel("Time Steps")
plt.ylabel("Traffic (Mbps)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
