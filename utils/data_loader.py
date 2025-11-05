import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath="data/traffic_data.csv"):
    """
    Loads and preprocesses the mobile traffic dataset for model training.
    - Reads CSV
    - Sorts by timestamp
    - Scales numeric features
    - Splits into train/test sets
    """

    # Read the CSV file
    df = pd.read_csv(filepath)

    # Convert timestamp to datetime and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["timestamp"])

    # --- Feature Engineering ---
    # Create total traffic as target variable (down + up)
    df["traffic"] = df["down"] + df["up"]

    # Define features (independent variables)
    X = df[["rnti_count", "mcs_down", "mcs_up", "rb_down", "rb_up"]]

    # Define target variable
    y = df[["traffic"]]

    # --- Normalization ---
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test, df


def get_client_datasets(filepath="data/traffic_data.csv", group_by="timestamp"):
    """
    Splits dataset into smaller 'clients' (optional grouping, e.g., by day/hour).
    Currently groups by time slices, can be customized for federated setups.
    """

    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["timestamp"])

    # Create total traffic column
    df["traffic"] = df["down"] + df["up"]

    clients_data = {}

    # Example grouping: by day
    df["day"] = df["timestamp"].dt.date
    days = df["day"].unique()

    for day in days:
        df_client = df[df["day"] == day]

        X = df_client[["rnti_count", "mcs_down", "mcs_up", "rb_down", "rb_up"]]
        y = df_client[["traffic"]]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        clients_data[str(day)] = (X_train, y_train, X_test, y_test)

    return clients_data
