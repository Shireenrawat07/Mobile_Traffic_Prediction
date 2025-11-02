# main.py
import torch
import os
import copy
import pandas as pd
from models.lstm_model import TrafficPredictor
from server.aggregator import average_models
from clients.client import train_local_model


# ------------------ USER SETTINGS ------------------
NUM_CLIENTS = 3                # number of clients
ROUNDS = 10                     # number of federated rounds
EPOCHS_PER_CLIENT = 5          # local training epochs per client
SEQ_LEN = 10                   # sequence length for LSTM
LR = 0.005                    # learning rate
DATA_FOLDER = "data"           # folder containing traffic_data.csv
BASE_CSV = os.path.join(DATA_FOLDER, "traffic_data.csv")
# ---------------------------------------------------


def split_csv_for_clients(base_csv, num_clients, out_folder="data"):
    """
    Splits a single CSV into multiple smaller client files
    to simulate local datasets (non-IID).
    """
    df = pd.read_csv(base_csv)
    n = len(df)
    chunk_size = n // num_clients
    client_files = []

    for i in range(num_clients):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_clients - 1 else n
        sub = df.iloc[start:end]
        fname = os.path.join(out_folder, f"client_{i + 1}.csv")
        sub.to_csv(fname, index=False)
        client_files.append(fname)

    return client_files


def run_federated_training():
    """Run full federated learning simulation."""
    # 1️⃣ Split dataset among clients
    client_files = split_csv_for_clients(BASE_CSV, NUM_CLIENTS, out_folder=DATA_FOLDER)
    print("Client data files:", client_files)

    # 2️⃣ Initialize the global LSTM model (2 layers)
   
    global_model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=2, output_size=1)

    global_state = global_model.state_dict()

    # 3️⃣ Federated Training Rounds
    for r in range(ROUNDS):
        print(f"\n*** Federated Round {r + 1}/{ROUNDS} ***")
        local_states, local_losses, weights = [], [], []

        # Train each client
        for i, cfile in enumerate(client_files):
            print(f"-> Training client {i + 1} on {cfile}")
            state_dict, loss = train_local_model(
                filepath=cfile,
                seq_len=SEQ_LEN,
                epochs=EPOCHS_PER_CLIENT,
                lr=LR
            )
            local_states.append(state_dict)
            local_losses.append(loss)
            data_len = len(pd.read_csv(cfile))
            weights.append(data_len)
            print(f"   Client {i + 1} final loss: {loss:.6f}")

        # 4️⃣ Aggregate local weights into global model
        avg_state = average_models(local_states, weights)
        global_model.load_state_dict(avg_state)

        avg_loss = sum(local_losses) / len(local_losses)
        print(f"Round {r + 1} aggregation done. Avg client loss: {avg_loss:.6f}")

    # 5️⃣ Save trained global model
    torch.save(global_model.state_dict(), "global_model.pth")
    print("\n✅ Federated training complete. Global model saved to 'global_model.pth'.")


if __name__ == "__main__":
    run_federated_training()
