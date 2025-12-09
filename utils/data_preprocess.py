import pandas as pd
import numpy as np

def load_real_traffic_data(filepath, column='down'):
    """
    Loads CSV and returns RAW (unscaled) data.
    Scaling must be done ONLY in training/evaluation scripts.
    """
    df = pd.read_csv(filepath)

    # Handle timestamp column if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    series = df[column].values.astype(float).reshape(-1, 1)

 

    return series


def prepare_sequences(series, seq_len=10):
    """
    Converts time series into sequences for LSTM.
    """
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)
