"""
Project 3: Sentiment Analyzer (NLP)
Small example using a tiny in-memory dataset and TF-IDF + LogisticRegression.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = None
vectorizer = None
metrics = {}


def load_data():
    # Tiny toy dataset for demonstration (positive=1, negative=0)
    texts = [
        'I love this product, it is amazing and works great',
        'This is the worst purchase I have ever made',
        'Absolutely fantastic, highly recommend',
        'Terrible, do not buy',
        'I am very happy with the quality',
        'Not good, very disappointing',
        'Exceeded my expectations',
        'It broke after a day, awful'
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    return texts, labels


def preprocess_data(texts):
    # minimal preprocessing done by TfidfVectorizer
    return texts


def train_model():
    global model, vectorizer, metrics
    texts, labels = load_data()
    texts = preprocess_data(texts)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500)
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=7)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    metrics = {'accuracy': float(acc)}


def predict(input_data):
    if model is None or vectorizer is None:
        raise RuntimeError('Model not trained')
    text = input_data.get('text')
    if text is None:
        raise ValueError('Missing field text')
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = float(model.predict_proba(X).max()) if hasattr(model, 'predict_proba') else None
    label = 'Positive' if int(pred) == 1 else 'Negative'
    return {'prediction': label, 'probability': proba, 'metrics': metrics}
