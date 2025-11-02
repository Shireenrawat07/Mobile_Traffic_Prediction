# test_global_model.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.lstm_model import TrafficPredictor
from utils.data_preprocess import load_real_traffic_data, prepare_sequences

MODEL_PATH = "global_model.pth"
CSV_PATH = "data/traffic_data.csv"
COLUMN = "bandwidth_usage"
SEQ_LEN = 10

# Load data
series, scaler = load_real_traffic_data(CSV_PATH, COLUMN)
X, y = prepare_sequences(series, seq_len=SEQ_LEN)

X = torch.tensor(X).float()
y = torch.tensor(y).float()

# Load model
model = TrafficPredictor(input_size=1, hidden_size=50, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Predict
with torch.no_grad():
    preds = model(X).numpy()

# Reverse scaling
y_true = scaler.inverse_transform(y.numpy().reshape(-1, 1))
y_pred = scaler.inverse_transform(preds.reshape(-1, 1))

# --- Metrics ---
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"📊 Model Evaluation:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(y_true, label="Actual", color="blue")
plt.plot(y_pred, label="Predicted", color="orange")
plt.title("Traffic Prediction (Federated LSTM)")
plt.xlabel("Time Step")
plt.ylabel(COLUMN)
plt.legend()
plt.show()
