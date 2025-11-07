import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def load_real_traffic_data(filepath, column='down'):
    """
    Loads CSV, normalizes one numeric column, and saves scaler parameters.
    """
    df = pd.read_csv(filepath)

    # Handle timestamp column if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    # Scale data
    series = df[column].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    # ✅ Save scaling parameters correctly for evaluation
    torch.save({
        'min_': scaler.min_,
        'scale_': scaler.scale_
    }, 'scaling_params.pt')

    print("✅ Scaler parameters saved to scaling_params.pt")

    return scaled_series, scaler


def prepare_sequences(series, seq_len=10):
    """
    Converts scaled time series into sequences for LSTM.
    """
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)
