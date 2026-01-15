import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.explainers.lime_explainer import LIMEExplainer
from src.explainers.shap_explainer import SHAPExplainer


class EfficiencyEvaluator:
    def evaluate_explainer(self, explainer, texts, method_name):
        """Measure time to generate explanations"""
        times = []

        for text in tqdm(texts, desc=f"Evaluating {method_name}"):
            start = time.time()
            try:
                if method_name == "LIME":
                    explanation = explainer.explain(text)
                    elapsed = time.time() - start
                    times.append(elapsed)
                elif method_name == "SHAP":
                    explanation = explainer.explain(
                        text, class_name="Legal/Regulatory Risk"
                    )
                    elapsed = time.time() - start
                    times.append(elapsed)
            except Exception as e:
                print(f"Error: {e}")
                times.append(np.nan)

        valid_times = [t for t in times if not np.isnan(t)]

        return {
            "method": method_name,
            "mean_time": np.mean(valid_times),
            "median_time": np.median(valid_times),
            "std_time": np.std(valid_times),
            "min_time": np.min(valid_times),
            "max_time": np.max(valid_times),
            "success_rate": len(valid_times) / len(texts),
        }


# Usage
if __name__ == "__main__":
    # Load model
    import pickle

    with open("results/model_artifacts/xgboost.pkl", "rb") as f:
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
    lime_explainer = LIMEExplainer(model, model_data["mlb"].classes_)

    # SHAP
    # Need some training data for background
    import pandas as pd

    artifact_folder = Path(__name__).resolve().parent
    df = pd.read_csv(artifact_folder / "data/risk_data.csv")
    X_train = df["paragraph_cleaned"].values[:100]

    shap_explainer = SHAPExplainer(model, model_type="xgboost")
    shap_explainer.fit(X_train, max_samples=50)

    evaluator = EfficiencyEvaluator()

    # test
    test_texts = ["Sample text..."] * 50  # 50 samples
    # Evaluate each explainer
    results = []
    results.append(evaluator.evaluate_explainer(lime_explainer, test_texts, "LIME"))
    results.append(evaluator.evaluate_explainer(shap_explainer, test_texts, "SHAP"))

    # Print comparison
    print("\nEfficiency Comparison:")
    print(f"{'Method':<20} {'Mean Time (s)':<15} {'Median Time (s)':<15}")
    print("-" * 50)
    for result in results:
        print(
            f"{result['method']:<20} {result['mean_time']:<15.3f} {result['median_time']:<15.3f}"
        )
