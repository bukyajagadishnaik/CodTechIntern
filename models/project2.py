"""
Project 2: Breast Cancer Classifier

Uses the sklearn breast cancer dataset and trains a GradientBoostingClassifier with
standard scaling. The model is trained on startup (or lazy when first accessed) and
returns a binary prediction (Malignant / Benign) plus training metrics and top feature importances.

The prediction form accepts a small set of representative features. Missing features are filled with
the dataset mean so we only require a few inputs for a quick prediction demo.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

model = None
scaler = None
metrics = {}
feature_names = []
feature_means = None

# We'll expose a short list of fields for the UI (representative features)
ui_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness'
]


def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=['target'])
    y = data.frame['target']
    return X, y


def preprocess_data(X, fit_scaler=False):
    global scaler
    X = X.copy()
    X = X.fillna(X.median())
    if fit_scaler:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        return pd.DataFrame(Xs, columns=X.columns)
    else:
        if scaler is None:
            return X
        Xs = scaler.transform(X)
        return pd.DataFrame(Xs, columns=X.columns)


def train_model():
    global model, metrics, feature_names, feature_means, scaler
    X, y = load_data()
    feature_names = list(X.columns)
    feature_means = X.mean()
    Xs = preprocess_data(X, fit_scaler=True)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # top features
    importances = getattr(model, 'feature_importances_', None)
    if importances is not None:
        top_idx = np.argsort(importances)[::-1][:8]
        top = [(feature_names[i], float(importances[i])) for i in top_idx]
    else:
        top = []
    metrics = {'accuracy': float(acc), 'top_features': top}


def _build_input_row(input_data):
    """Construct a full feature vector using provided ui_features; fill others with mean."""
    if feature_means is None:
        raise RuntimeError('Model not trained')
    row = feature_means.copy()
    # map provided UI features into row
    for f in ui_features:
        v = input_data.get(f)
        if v is not None and v != '':
            try:
                row[f] = float(v)
            except Exception:
                raise ValueError(f'Invalid value for {f}: {v}')
    return pd.DataFrame([row])


def predict(input_data):
    if model is None:
        raise RuntimeError('Model not trained')
    X = _build_input_row(input_data)
    Xs = preprocess_data(X, fit_scaler=False)
    pred = model.predict(Xs)[0]
    label = 'Malignant' if int(pred) == 0 else 'Benign'  # note: sklearn uses 0=malignant,1=benign
    return {'prediction': label, 'metrics': metrics}
