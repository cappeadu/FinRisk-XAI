import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

RSEED = 182026


class LogisticBaseline:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
        )
        self.mlb = MultiLabelBinarizer()
        self.models = {}  # One model per class

    def prepare_data(self, df):
        """Prepare data for training"""

        # Parse labels
        df["label_list"] = df["labels"].apply(
            lambda x: [label.strip() for label in x.split(",") if label.strip()]
        )

        # Remove ambiguous or unlabeled
        df = df[df["label_list"].apply(lambda x: len(x) > 0 and "Ambiguous" not in x)]

        X = df["paragraph_cleaned"].values
        y_list = df["label_list"].tolist()

        # Transform labels
        y = self.mlb.fit_transform(y_list)

        return X, y

    def train(self, X_train, y_train):
        """Train one-vs-rest classifiers"""
        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Train one model per class
        for i, class_name in enumerate(self.mlb.classes_):
            print(f"Training {class_name}...")
            model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            model.fit(X_train_vec, y_train[:, i])
            self.models[class_name] = model

    def predict(self, X, threshold=0.5):
        """Predict labels for text"""
        X_vec = self.vectorizer.transform(X)

        predictions = np.zeros((len(X), len(self.mlb.classes_)))
        for i, class_name in enumerate(self.mlb.classes_):
            probs = self.models[class_name].predict_proba(X_vec)[:, 1]
            predictions[:, i] = (probs >= threshold).astype(int)

        return predictions

    def predict_proba(self, X):
        """Get probability predictions"""
        X_vec = self.vectorizer.transform(X)

        probas = np.zeros((len(X), len(self.mlb.classes_)))
        for i, class_name in enumerate(self.mlb.classes_):
            probas[:, i] = self.models[class_name].predict_proba(X_vec)[:, 1]

        return probas

    def save(self, path):
        """Save model"""
        model_data = {
            "vectorizer": self.vectorizer,
            "mlb": self.mlb,
            "models": self.models,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, path):
        """Load model"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        self.vectorizer = model_data["vectorizer"]
        self.mlb = model_data["mlb"]
        self.models = model_data["models"]


def evaluate_model(model, X, y, mlb):
    """Evaluate model performance"""
    y_pred = model.predict(X)

    # Overall metrics
    f1_micro = f1_score(y, y_pred, average="micro")
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_samples = f1_score(y, y_pred, average="samples")

    print("\nOverall Metrics:")
    print(f"F1 (micro): {f1_micro:.3f}")
    print(f"F1 (macro): {f1_macro:.3f}")
    print(f"F1 (samples): {f1_samples:.3f}")

    # Per-class metrics
    print("\nPer-Class Metrics:")
    for i, class_name in enumerate(mlb.classes_):
        f1 = f1_score(y[:, i], y_pred[:, i])
        support = y[:, i].sum()
        print(f"{class_name:25s}: F1={f1:.3f}, Support={int(support)}")

    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "f1_samples": f1_samples}


# Training script
if __name__ == "__main__":
    # Load data
    link = Path(__name__).resolve().parent / "data/risk_data.csv"
    df = pd.read_csv(link)

    # Prepare
    model = LogisticBaseline(max_features=5000)
    X, y = model.prepare_data(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RSEED
    )

    print(f"Training set: {len(X_train)} examples")
    print(f"Test set: {len(X_test)} examples")
    print(f"Classes: {model.mlb.classes_}")

    # Train
    model.train(X_train, y_train)

    # Evaluate

    metrics = evaluate_model(model, X_test, y_test, model.mlb)

    # Save
    artifact_folder = Path() / "results/model_artifacts"
    model.save(artifact_folder / "logistic_baseline.pkl")
    with open(artifact_folder / "logistic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved!")
