# utils/data_preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_real_traffic_data(filepath='data/traffic_data.csv', column='traffic_volume'):
    """
    Loads CSV, cleans, normalizes the `column`. Returns df and scaler.
    """
    df = pd.read_csv(filepath)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV. Available columns: {list(df.columns)}")
    # keep only the time-series column for simplicity
    series = df[[column]].copy()
    series = series.fillna(method='ffill').fillna(method='bfill')
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    return series, scaler

def prepare_sequences(series_df, seq_len=10):
    """
    Converts normalized series into (X, y) sequences for LSTM training.
    X shape: (num_samples, seq_len, 1)
    y shape: (num_samples, 1)
    """
    data = series_df['normalized'].values
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y
