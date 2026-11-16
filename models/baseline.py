from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def create_baseline():
    return Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])