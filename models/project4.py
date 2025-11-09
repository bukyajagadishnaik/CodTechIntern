"""
Project 4: Stock Price Predictor (Time-series toy example)
Creates synthetic stock-like series and trains a simple LinearRegression to predict next value from last 5.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model = None
metrics = {}
lag = 5


def load_data(n_points=1000):
    # Synthetic time series: sine wave + trend + noise
    t = np.arange(n_points)
    series = 0.1 * t + 5 * np.sin(0.02 * t) + np.random.normal(scale=0.5, size=n_points)
    return pd.Series(series)


def preprocess_data(series: pd.Series):
    # Build lag features
    df = pd.DataFrame({'y': series})
    for i in range(1, lag+1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df = df.dropna()
    X = df[[f'lag_{i}' for i in range(1, lag+1)]]
    y = df['y']
    return X, y


def train_model():
    global model, metrics
    series = load_data()
    X, y = preprocess_data(series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    metrics = {'rmse': float(rmse)}


def predict(input_data):
    if model is None:
        raise RuntimeError('Model not trained')
    # Expect a comma-separated list of last `lag` prices under key 'prices'
    s = input_data.get('prices')
    if s is None:
        raise ValueError('Missing field prices')
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    if len(parts) != lag:
        raise ValueError(f'Provide exactly {lag} comma-separated values')
    vals = [float(x) for x in parts]
    X = np.array(vals).reshape(1, -1)
    pred = model.predict(X)[0]
    return {'prediction': float(pred), 'metrics': metrics}
