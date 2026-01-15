# src/explainers/shap_explainer.py
from pathlib import Path

import numpy as np
import shap


class SHAPExplainer:
    def __init__(self, model, model_type="logistic"):
        """
        model_type: 'logistic', 'xgboost'
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None

    def fit(self, X_train, max_samples=100):
        """Initialize explainer with background data"""
        if self.model_type == "logistic":
            # For sklearn models - use one class as example
            X_train_vec = self.model.vectorizer.transform(X_train[:max_samples])
            # Create explainer for first class
            first_class = list(self.model.models.keys())[0]
            self.explainer = shap.LinearExplainer(
                self.model.models[first_class], X_train_vec
            )
            self.explainer_type = "single"

        elif self.model_type == "xgboost":
            # Use TreeExplainer for each class
            self.explainer = {}
            for class_name in self.model.models:
                self.explainer[class_name] = shap.TreeExplainer(
                    self.model.models[class_name]
                )
            self.explainer_type = "multi"

        else:  # neural
            # Use KernelExplainer (slower but model-agnostic)
            def predict_fn(texts):
                return self.model.predict_proba(texts)

            self.explainer = shap.KernelExplainer(predict_fn, X_train[:max_samples])
            self.explainer_type = "single"

    def explain(self, text, class_name):
        X_vec = self.model.vectorizer.transform([text])
        feature_names = self.model.vectorizer.get_feature_names_out()

        if self.model_type == "xgboost":
            explainer = self.explainer[class_name]
            shap_values = explainer.shap_values(X_vec)

            # XGBoost binary classifier : shape (1, n_features)
            values = shap_values[0]

        else:
            shap_values = self.explainer.shap_values(X_vec)
            values = shap_values[0]

        word_weights = [
            (feature_names[i], float(values[i]))
            for i in range(len(values))
            if abs(values[i]) > 0.001
        ]

        word_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        return word_weights

    def get_top_words(self, explanation, k=10):
        """Get top-k words"""
        return explanation[:k]


# Usage
if __name__ == "__main__":
    import pickle

    # Load model
    artifact_folder = Path(__name__).resolve().parent
    with open(artifact_folder / "results/model_artifacts/xgboost.pkl", "rb") as f:
        model_data = pickle.load(f)

    class ModelWrapper:
        def __init__(self, model_data):
            self.vectorizer = model_data["vectorizer"]
            self.models = model_data["models"]
            self.mlb = model_data["mlb"]

        def predict_proba(self, texts):
            X_vec = self.vectorizer.transform(texts)
            probas = np.zeros((len(texts), len(self.mlb.classes_)))
            for i, class_name in enumerate(self.mlb.classes_):
                probas[:, i] = self.models[class_name].predict_proba(X_vec)[:, 1]
            return probas

    model = ModelWrapper(model_data)

    # Need some training data for background
    import pandas as pd

    df = pd.read_csv(artifact_folder / "data/risk_data.csv")
    X_train = df["paragraph_cleaned"].values[:100]

    explainer = SHAPExplainer(model, model_type="xgboost")
    explainer.fit(X_train, max_samples=50)

    # test
    test_text = df["paragraph"].iloc[-1]

    explanation = explainer.explain(test_text, class_name="Legal/Regulatory Risk")

    print("\n")
    print("SHAP Explanation:")
    for word, weight in explanation[:10]:
        print(f"  {word}: {weight:.4f}")
