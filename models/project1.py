"""
Project 1: Customer Churn Prediction (Classification)

This module uses a synthetic or generated telecom-style churn dataset for a quick demo.
It trains a RandomForestClassifier to predict whether a customer will churn. The module
exposes load_data(), preprocess_data(), train_model(), and predict(input_dict).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = None
metrics = {}
feature_names = []
categorical_maps = {}


def load_data(n_samples=2000, random_state=42):
    """Generate a small synthetic churn dataset."""
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame()
    # Numerical features
    df['tenure'] = rng.randint(0, 72, size=n_samples)
    df['monthly_charges'] = rng.uniform(20, 120, size=n_samples).round(2)
    df['total_charges'] = (df['tenure'] * df['monthly_charges']).round(2)
    df['num_services'] = rng.randint(1, 6, size=n_samples)
    # Categorical features
    df['contract'] = rng.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.6,0.25,0.15])
    df['internet_service'] = rng.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.4,0.45,0.15])
    df['payment_method'] = rng.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], size=n_samples)
    # churn label with some heuristics
    churn_prob = (
        0.3 * (df['contract'] == 'Month-to-month').astype(float)
        + 0.25 * (df['internet_service'] == 'Fiber optic').astype(float)
        + 0.2 * (df['tenure'] < 12).astype(float)
        + 0.05 * (df['num_services'] <= 1).astype(float)
    )
    churn = rng.rand(n_samples) < churn_prob
    # make churn label slightly more separable by boosting weights
    churn = rng.rand(n_samples) < (churn_prob * 0.9 + 0.05 * (df['monthly_charges'] > 80).astype(float))
    df['churn'] = churn.astype(int)
    return df


def preprocess_data(df, fit_maps=False):
    """Encode categoricals and fill missing; when fit_maps=True capture mappings for later use."""
    df = df.copy()
    # Fill numerics
    num_cols = ['tenure','monthly_charges','total_charges','num_services']
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # Encode categoricals
    cat_cols = ['contract','internet_service','payment_method']
    global categorical_maps
    if fit_maps:
        categorical_maps = {}
        for c in cat_cols:
            uniques = sorted(df[c].unique())
            mapping = {v:i for i,v in enumerate(uniques)}
            categorical_maps[c] = mapping
            df[c] = df[c].map(mapping)
    else:
        for c in cat_cols:
            mapping = categorical_maps.get(c,{})
            df[c] = df[c].map(lambda x: mapping.get(x, -1))
    return df


def train_model():
    global model, metrics, feature_names, categorical_maps
    df = load_data()
    y = df['churn']
    X = df.drop(columns=['churn'])
    X = preprocess_data(X, fit_maps=True)
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # simple feature importances
    importances = getattr(model, 'feature_importances_', None)
    if importances is not None:
        top_idx = np.argsort(importances)[::-1][:8]
        top = [(feature_names[i], float(importances[i])) for i in top_idx]
    else:
        top = []
    metrics = {'accuracy': float(acc), 'top_features': top}


def _build_row_from_input(input_data):
    # Build a full dataframe row from input dict; allow missing fields (fill with medians/defaults)
    # Use one sample from load_data to get reasonable defaults
    sample = load_data(n_samples=1)
    row = sample.drop(columns=['churn']).iloc[0].to_dict()
    # override with provided inputs
    for k,v in input_data.items():
        if k in ['tenure','num_services']:
            row[k] = int(v)
        elif k in ['monthly_charges','total_charges']:
            row[k] = float(v)
        else:
            row[k] = v
    return pd.DataFrame([row])


def predict(input_data):
    if model is None:
        raise RuntimeError('Model not trained')
    X = _build_row_from_input(input_data)
    X = preprocess_data(X, fit_maps=False)
    pred = model.predict(X)[0]
    label = 'Churn' if int(pred) == 1 else 'No churn'
    return {'prediction': label, 'metrics': metrics}
