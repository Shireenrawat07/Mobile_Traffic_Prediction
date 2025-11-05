import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_all_client_data(base_path='data/csv/', columns=['down', 'up', 'rnti_count']):
    """
    Loads and normalizes all train/test CSV files from the specified folder.
    
    Returns:
        train_data_dict: dict of client_name -> normalized training DataFrame
        test_data_dict: dict of client_name -> normalized testing DataFrame
        scalers: dict of client_name -> fitted MinMaxScaler
    """
    train_data_dict = {}
    test_data_dict = {}
    scalers = {}

    # --- Loop through all files ---
    for filename in os.listdir(base_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(base_path, filename)
            df = pd.read_csv(filepath)

            # Identify client and dataset type
            parts = filename.replace(".csv", "").split("_")
            if len(parts) >= 2:
                data_type, client_name = parts[0], "_".join(parts[1:])
            else:
                continue  # skip unexpected files

            # Only keep relevant columns (if specified)
            if all(col in df.columns for col in columns):
                df = df[columns]
            else:
                print(f"⚠️ Skipping {filename} - missing columns {columns}")
                continue

            # Normalize each client's data
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=columns)

            # Store in dicts
            if "train" in data_type.lower():
                train_data_dict[client_name] = df_scaled
            elif "test" in data_type.lower():
                test_data_dict[client_name] = df_scaled

            scalers[client_name] = scaler
            print(f"✅ Loaded {filename} ({data_type}, {client_name}) with shape {df.shape}")

    return train_data_dict, test_data_dict, scalers


# --- Example usage ---
if __name__ == "__main__":
    train_data, test_data, scalers = load_all_client_data()

    print("\nClients loaded:")
    for client, df in train_data.items():
        print(f"🧠 {client}: {df.shape}")
