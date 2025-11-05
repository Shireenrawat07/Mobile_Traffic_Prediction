import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_real_traffic_data(filepath, column='down'):
    """
    Loads one numeric column (like 'down' or 'up') from dataset,
    normalizes it, and returns (scaled_series, scaler).
    """
    df = pd.read_csv(filepath)

    # Handle timestamp column if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset. Available: {list(df.columns)}")

    # Extract and scale the selected column
    series = df[column].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    return scaled_series, scaler


def prepare_sequences(series, seq_len=10):
    """
    Converts a time series (1D) into sequences for LSTM.
    Input: scaled time series (numpy array)
    Output: (X, y) numpy arrays ready for training
    """
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    return X, y
