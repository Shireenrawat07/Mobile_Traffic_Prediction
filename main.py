import torch
import os
import pandas as pd
from models.lstm_model import TrafficPredictor, load_model_from_checkpoint
from clients.client import train_local_model

# ------------------ USER SETTINGS ------------------
NUM_CLIENTS = 3
ROUNDS = 10
EPOCHS_PER_CLIENT = 30
SEQ_LEN = 10
LR = 0.001
DATA_FOLDER = "data"
BASE_CSV = os.path.join(DATA_FOLDER, "traffic_data.csv")
GLOBAL_MODEL_PATH = "global_model.pth"
# ---------------------------------------------------

# --- Split CSV among clients ---
def split_csv_for_clients(base_csv, num_clients, out_folder="data"):
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


# --- Federated training ---
def average_models(state_dicts, weights):
    avg_state = {}
    total_weight = sum(weights)
    for key in state_dicts[0].keys():
        avg_state[key] = sum([state_dicts[i][key] * weights[i] for i in range(len(state_dicts))]) / total_weight
    return avg_state


def run_federated_training():
    client_files = split_csv_for_clients(BASE_CSV, NUM_CLIENTS, out_folder=DATA_FOLDER)
    print("📂 Client data files:", client_files)

    # Initialize or load global model
    if os.path.exists(GLOBAL_MODEL_PATH):
        print("🔄 Loading existing global model checkpoint...")
        global_model = load_model_from_checkpoint(GLOBAL_MODEL_PATH, input_size=1, output_size=1)
    else:
        print("⚡ Initializing new global model...")
        global_model = TrafficPredictor(input_size=1, hidden_size=128, num_layers=3, output_size=1)

    global_state = global_model.state_dict()

    for r in range(ROUNDS):
        print(f"\n==============================")
        print(f"🌍 Federated Round {r + 1}/{ROUNDS}")
        print(f"==============================")
        local_states, local_losses, weights = [], [], []

        for i, cfile in enumerate(client_files):
            print(f"🧠 Training client {i + 1} on {cfile}")
            state_dict, loss = train_local_model(
                filepath=cfile,
                seq_len=SEQ_LEN,
                epochs=EPOCHS_PER_CLIENT,
                lr=LR,
                global_model_state=global_state
            )
            local_states.append(state_dict)
            local_losses.append(loss)
            data_len = len(pd.read_csv(cfile))
            weights.append(data_len)
            print(f"✅ Client {i + 1} final loss: {loss:.6f}")

        avg_state = average_models(local_states, weights)
        global_model.load_state_dict(avg_state)
        global_state = avg_state

        avg_loss = sum(local_losses) / len(local_losses)
        print(f"📊 Round {r + 1} completed. Avg client loss: {avg_loss:.6f}")

    torch.save(global_model.state_dict(), GLOBAL_MODEL_PATH)
    print(f"\n✅ Federated training complete. Global model saved to '{GLOBAL_MODEL_PATH}'.\n")


if __name__ == "__main__":
    run_federated_training()
