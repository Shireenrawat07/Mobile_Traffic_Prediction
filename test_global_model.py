import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ✅ Import functions and model
from utils.data_preprocess import load_real_traffic_data, prepare_sequences
from models.lstm_model import TrafficPredictor  # ← make sure this file exists

# ====== CONFIG ======
DATA_PATH = "Dataset/full_dataset.csv"
MODEL_PATH = "global_model.pth"
SCALER_PATH = "scaling_params.pt"
SEQ_LEN = 10
COLUMN = 'down'
# ====================


def load_and_scale_data(data_path, scaler_path, column):
    print("📂 Loading dataset and scaler...")

    # load normalized series from your utils/data_preprocess.py
    scaled_series, _ = load_real_traffic_data(data_path, column=column)

    # Load scaler params safely (handles both old/new PyTorch versions)
    params = torch.load(scaler_path, weights_only=False)

    scaler = MinMaxScaler()
    # ✅ handle both key styles ("min"/"scale" or "min_"/"scale_")
    scaler.min_ = params.get("min", params.get("min_", None))
    scaler.scale_ = params.get("scale", params.get("scale_", None))

    if scaler.min_ is None or scaler.scale_ is None:
        raise KeyError("Scaler parameters missing from scaling_params.pt")

    return scaled_series


def create_sequences(values, seq_len):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len])
    return (
      
    torch.tensor(X).float(),   # Already 3D (samples, seq_len, 1)
    torch.tensor(y).float().unsqueeze(-1)
)

    


def evaluate_model():
    print("🔍 Evaluating Global Model...")
    values = load_and_scale_data(DATA_PATH, SCALER_PATH, COLUMN)
    X, y = create_sequences(values, SEQ_LEN)

    # ✅ Load trained model
    model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        preds = model(X).squeeze().numpy()
        actual = y.squeeze().numpy()

    # ✅ Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:200], label="Actual", linewidth=2)
    plt.plot(preds[:200], label="Predicted", linestyle="--", linewidth=2)
    plt.title("Global Model Prediction vs Actual (First 200 samples)")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Traffic Flow")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    evaluate_model()
