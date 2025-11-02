import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath="data/traffic_data.csv"):
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Sort data by timestamp
    df = df.sort_values(by=["cell_id", "timestamp"])
    
    # Features and target
    X = df[["user_count", "bandwidth_usage", "latency"]]
    y = df["throughput"]
    
    # Normalize data for better LSTM performance
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    return X_train, y_train, X_test, y_test, df
def get_client_datasets(filepath="data/traffic_data.csv"):
    df = pd.read_csv(filepath)
    df = df.sort_values(by=["cell_id", "timestamp"])

    clients_data = {}
    cell_ids = df["cell_id"].unique()

    for cid in cell_ids:
        df_client = df[df["cell_id"] == cid]
        X = df_client[["user_count", "bandwidth_usage", "latency"]]
        y = df_client["throughput"]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        clients_data[cid] = (X_train, y_train, X_test, y_test)

    return clients_data
