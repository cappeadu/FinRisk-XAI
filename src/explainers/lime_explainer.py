# src/explainers/lime_explainer.py
from pathlib import Path

import numpy as np
from lime.lime_text import LimeTextExplainer


class LIMEExplainer:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names)

    def explain(self, text, num_features=10):
        def predict_proba_fn(texts):
            return self.model.predict_proba(texts)

        probs = self.model.predict_proba([text])[0]

        exp = self.explainer.explain_instance(
            text,
            predict_proba_fn,
            labels=list(np.where(probs > 0.0)[0]),
            num_features=num_features,
            num_samples=500,
        )

        explanations = {}

        for class_idx in exp.local_exp:
            class_name = self.class_names[class_idx]
            explanations[class_name] = {
                "word_weights": exp.as_list(label=class_idx),
                "score": probs[class_idx],
            }

        return explanations

    def get_top_words(self, explanation, class_name, k=10):
        """Get top-k words for a class"""
        word_weights = explanation[class_name]["word_weights"]
        sorted_weights = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)
        return sorted_weights[:k]


# Usage example
if __name__ == "__main__":
    import pickle

    # Load model
    artifact_folder = Path(__name__).resolve().parent / "results/model_artifacts"
    with open(artifact_folder / "logistic_baseline.pkl", "rb") as f:
        model_data = pickle.load(f)

    # Mock model wrapper
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

    # Test
    test_text = "intense competition in our industry could result in pricing pressure and \
        loss of market share, which may adversely affect our revenues and profitability."
    print(f"{test_text}\n")
    explanation = explainer.explain(test_text)

    print("LIME Explanation:")
    for class_name in explanation:
        print(f"\n{class_name}:")
        top_words = explainer.get_top_words(explanation, class_name, k=6)
        for word, weight in top_words:
            print(f"  {word}: {weight:.3f}")
    print(model.predict_proba([test_text]))
