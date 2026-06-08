import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import argparse

# ✅ Import models and utilities
from utils.data_preprocess import load_real_traffic_data
from models.lstm_model import TrafficPredictor
from models.rnn_model import TrafficPredictorRNN
from models.lstm_model import TrafficPredictor as LSTMTrafficPredictor
from models.gru_model import TrafficPredictorGRU
from models.mlp_model import TrafficPredictorMLP

# ===== CONFIG =====
DATA_PATH = "Dataset/full_dataset.csv"
MODEL_PATH = "saved_models/global_model.pth"
SCALER_PATH = "saved_models/scaling_params.pt"
SEQ_LEN = 10
COLUMN = "down"

MODEL_PATH_LSTM = "saved_models/global_model.pth"
MODEL_PATH_GRU = "saved_models/global_model_gru.pth"
MODEL_PATH_MLP = "saved_models/global_model_mlp.pth"

# ===== Command-line argument for model type =====
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="lstm",
    choices=["lstm", "gru","rnn", "mlp"],
    help="Choose model to evaluate",
)

args = parser.parse_args()


def load_and_scale_data(data_path, scaler_path, column):
    # Load raw series
    series = load_real_traffic_data(data_path, column=column)

    # Load scaler params saved during TRAINING
    scaler_dict = torch.load(scaler_path, weights_only=False)

    min_ = scaler_dict["min_"]
    scale_ = scaler_dict["scale_"]

    # Apply SAME scaling used during training
    scaled_series = (series - min_) * scale_

    return scaled_series


def create_sequences(values, seq_len):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])

    X = torch.tensor(np.array(X)).float()
    y = torch.tensor(np.array(y)).float().unsqueeze(-1)
    return X, y


def evaluate_model():
    print("🔍 Evaluating Global Model...")

    # Load data
    values = load_and_scale_data(DATA_PATH, SCALER_PATH, COLUMN)
    X, y = create_sequences(values, SEQ_LEN)

    # ===== Choose model =====
    if args.model.lower() == "rnn":
        print("🟢 Evaluating RNN model")
        model = TrafficPredictorRNN(input_size=1, hidden_size=128, num_layers=3)
    # ===== Choose model & checkpoint =====
    model_name = args.model.lower()

    if model_name == "gru":
        print("🟢 Evaluating GRU Model")
        model = TrafficPredictorGRU(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        state = torch.load(MODEL_PATH_GRU, weights_only=True)

    elif model_name == "mlp":
        print("🟣 Evaluating MLP Model")
        model = TrafficPredictorMLP(
            input_size=1,
            hidden_size=128,
            num_layers=3,
            output_size=1,
            seq_len=SEQ_LEN,
        )
        state = torch.load(MODEL_PATH_MLP, weights_only=True)

    else:
        print("🔵 Evaluating LSTM Model")
        model = LSTMTrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
        state = torch.load(MODEL_PATH_LSTM, weights_only=True)

    # Load weights
    model.load_state_dict(state)
    model.eval()

    # If needed, reshape X to (batch, seq_len, 1)
    if X.ndim == 2:
        X_in = X.unsqueeze(-1)  # (N, seq_len, 1)
    else:
        X_in = X

    # ===== Predict =====
    with torch.no_grad():
        preds = model(X_in).squeeze().numpy()
        actual = y.squeeze().numpy()

    # ===== Metrics =====
    # (Optionally ignore last 200 if required – same as your old code)
    actual_eval = actual[:-200]
    preds_eval = preds[:-200]

    mae = mean_absolute_error(actual_eval, preds_eval)
    rmse = np.sqrt(np.mean((actual_eval - preds_eval) ** 2))
    nrmse = rmse / (actual_eval.max() - actual_eval.min())

    print("\n📊 Model Evaluation Metrics:")
    print(f"MAE   = {mae:.6f}")
    print(f"NRMSE = {nrmse:.6f}")
    # ===========================
# SAVE METRICS TO CSV
# ===========================
    import pandas as pd
    import os

    model_name = args.model.upper()

    new_row = pd.DataFrame([{
        "Model": model_name,
        "MAE": mae,
        "NRMSE": nrmse
    }])

    file_path = "results/Models_Results/GRU_MODEL_RESULTS.csv"

    if os.path.exists(file_path):
        new_row.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(file_path, mode='w', header=True, index=False)

    print("✅ Metrics saved to results/Model_Results/GRU_MODEL_RESULT.csv")

    # ===== Plot predictions =====
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:-200],label="Actual", linewidth=2)
    plt.plot(preds[:-200],label="Predicted", linestyle="--", linewidth=2)
    plt.title(f"{args.model.upper()} Model Predictions vs Actual")
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Traffic Flow")
    plt.legend()
    plt.grid(True)

    text = f"MAE: {mae:.6f}\nNRMSE: {nrmse:.6f}"
    plt.gcf().text(0.02, 0.90, text,
               fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8),
               transform=plt.gca().transAxes)
    plt.savefig("plots/gru_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
