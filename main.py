import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils.data_preprocess import load_real_traffic_data, prepare_sequences
from models.lstm_model import TrafficPredictor

# =============================
# CONFIG
# =============================
FILE_PATH = "Dataset/full_dataset.csv"
SEQ_LEN = 10
COLUMN = "down"
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001


def train_model():
    # --- Load and preprocess data ---
    scaled_series, _ = load_real_traffic_data(FILE_PATH, COLUMN)
    X, y = prepare_sequences(scaled_series, SEQ_LEN)

    X = torch.tensor(X).float()
    y = torch.tensor(y).float()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --- Model, Loss, Optimizer ---
    model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 Training started...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "global_model.pth")
    print("✅ Model saved as global_model.pth")
    print("✅ Scaler saved as scaling_params.pt (from data_preprocess.py)")


if __name__ == "__main__":
    train_model()
