# src/evaluation/sparsity.py
from pathlib import Path

import numpy as np


class SparsityEvaluator:
    def evaluate_explanation(self, explanation, threshold=0.01):
        """
        Count number of non-zero features
        """
        if isinstance(explanation, dict):
            # For LIME
            class_name = list(explanation.keys())[0]
            word_weights = explanation[class_name]["word_weights"]
        else:
            # For others
            word_weights = explanation

        # Count features above threshold
        significant_features = sum(
            1 for _, weight in word_weights if abs(weight) > threshold
        )
        total_features = len(word_weights)

        return {
            "significant_features": significant_features,
            "total_features": total_features,
            "sparsity_ratio": significant_features / total_features
            if total_features > 0
            else 0,
        }

    def evaluate_dataset(self, explanations):
        """Evaluate sparsity across multiple explanations"""
        results = [self.evaluate_explanation(exp) for exp in explanations]

        return {
            "mean_significant_features": np.mean(
                [r["significant_features"] for r in results]
            ),
            "mean_sparsity_ratio": np.mean([r["sparsity_ratio"] for r in results]),
            "std_sparsity_ratio": np.std([r["sparsity_ratio"] for r in results]),
        }


# Usage
if __name__ == "__main__":
    import pickle

    from src.explainers.lime_explainer import LIMEExplainer

    # Load model
    artifact_folder = Path(__name__).resolve().parent / "results/model_artifacts"
    with open(artifact_folder / "logistic_baseline.pkl", "rb") as f:
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
    explainer = LIMEExplainer(model, model_data["mlb"].classes_)
    evaluator = SparsityEvaluator()

    # Test
    test_texts = [
        "Market volatility may affect our business",
        "We face significant credit risk from counterparties",
    ]

    explanations = [explainer.explain(text) for text in test_texts]
    results = evaluator.evaluate_dataset(explanations)

    print("Sparsity Results:")
    print(f"Mean significant features: {results['mean_significant_features']:.1f}")
    print(f"Mean sparsity ratio: {results['mean_sparsity_ratio']:.3f}")
