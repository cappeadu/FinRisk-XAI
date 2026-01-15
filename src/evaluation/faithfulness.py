# src/evaluation/faithfulness.py
import pickle

import numpy as np
from tqdm import tqdm

from src.explainers.lime_explainer import LIMEExplainer


class FaithfulnessEvaluator:
    def __init__(self, model):
        self.model = model

    def sufficiency(self, text, explanation, k=10):
        """
        Sufficiency: Can top-k words alone produce similar prediction?
        """
        # Get original prediction
        orig_proba = self.model.predict_proba([text])[0]

        # Extract top-k words from explanation
        if isinstance(explanation, dict):
            # For LIME - get first class explanation
            class_name = list(explanation.keys())[0]
            word_weights = explanation[class_name]["word_weights"]
        else:
            # For SHAP- already list of tuples
            word_weights = explanation

        top_words = [word for word, _ in word_weights[:k]]

        # Create text with only top-k words
        words = text.split()
        kept_words = [w for w in words if any(tw in w.lower() for tw in top_words)]
        sufficient_text = " ".join(kept_words) if kept_words else text

        # Get prediction on sufficient text
        try:
            sufficient_proba = self.model.predict_proba([sufficient_text])[0]
        except:
            return 0.0  # If text too short, score 0

        # Calculate similarity (correlation)
        similarity = np.corrcoef(orig_proba, sufficient_proba)[0, 1]
        return similarity if not np.isnan(similarity) else 0.0

    def comprehensiveness(self, text, explanation, k=10):
        """
        Comprehensiveness: Does removing top-k words change prediction?
        """
        # Get original prediction
        orig_proba = self.model.predict_proba([text])[0]

        # Extract top-k words
        if isinstance(explanation, dict):
            class_name = list(explanation.keys())[0]
            word_weights = explanation[class_name]["word_weights"]
        else:
            word_weights = explanation

        top_words = [word for word, _ in word_weights[:k]]

        # Create text without top-k words
        words = text.split()
        removed_words = [
            w for w in words if not any(tw in w.lower() for tw in top_words)
        ]
        removed_text = " ".join(removed_words) if removed_words else "placeholder"

        # Get prediction on removed text
        try:
            removed_proba = self.model.predict_proba([removed_text])[0]
        except:
            return 1.0  # If text too short, assume max change

        # Calculate change (want large change = comprehensive)
        change = np.abs(orig_proba - removed_proba).mean()
        return change

    def evaluate_sample(self, text, explanation, k=10):
        """Evaluate both metrics for one sample"""
        suff = self.sufficiency(text, explanation, k)
        comp = self.comprehensiveness(text, explanation, k)
        return {"sufficiency": suff, "comprehensiveness": comp}

    def evaluate_dataset(self, texts, explanations, k=10):
        """Evaluate on multiple samples"""
        results = []
        for text, explanation in tqdm(zip(texts, explanations), total=len(texts)):
            result = self.evaluate_sample(text, explanation, k)
            results.append(result)

        # Aggregate
        avg_suff = np.mean([r["sufficiency"] for r in results])
        avg_comp = np.mean([r["comprehensiveness"] for r in results])

        return {
            "sufficiency_mean": avg_suff,
            "comprehensiveness_mean": avg_comp,
            "sufficiency_std": np.std([r["sufficiency"] for r in results]),
            "comprehensiveness_std": np.std([r["comprehensiveness"] for r in results]),
            "n_samples": len(results),
        }


# Usage
if __name__ == "__main__":
    # Load model
    with open("results/model_artifacts/logistic_baseline.pkl", "rb") as f:
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
    evaluator = FaithfulnessEvaluator(model)

    # Test on sample
    test_texts = [
        "Market volatility may affect our business",
        "We face significant credit risk from counterparties",
        "We may not be able to attract top talents and this may increase our operational risk",
    ]

    explanations = [explainer.explain(text) for text in test_texts]
    results = evaluator.evaluate_dataset(test_texts, explanations, k=10)

    print("Faithfulness Results:")
    print(
        f"Sufficiency: {results['sufficiency_mean']:.3f} ± {results['sufficiency_std']:.3f}"
    )
    print(
        f"Comprehensiveness: {results['comprehensiveness_mean']:.3f} ± {results['comprehensiveness_std']:.3f}"
    )
