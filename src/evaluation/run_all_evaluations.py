# src/evaluation/run_all_evaluations.py
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.efficiency import EfficiencyEvaluator
from src.evaluation.faithfulness import FaithfulnessEvaluator
from src.evaluation.sparsity import SparsityEvaluator
from src.explainers.lime_explainer import LIMEExplainer
from src.explainers.shap_explainer import SHAPExplainer

home_path = Path(__name__).resolve().parent
RSEED = 182026


def load_logistic_model():
    """Load logistic regression model"""
    with open(home_path / "results/model_artifacts/logistic_baseline.pkl", "rb") as f:
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

    return ModelWrapper(model_data), model_data["mlb"].classes_


def load_xgboost_model():
    """Load XGBoost model"""
    with open(home_path / "results/model_artifacts/xgboost.pkl", "rb") as f:
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

    return ModelWrapper(model_data), model_data["mlb"].classes_


def run_evaluation():
    """Run complete evaluation pipeline"""

    # Load test data
    df = pd.read_csv(home_path / "data/risk_data.csv")
    df["label_list"] = df["labels"].apply(
        lambda x: [l.strip() for l in x.split(",") if l.strip()]
    )
    df = df[df["label_list"].apply(lambda x: len(x) > 0 and "Ambiguous" not in x)]

    # Get test set (same split as training)
    from sklearn.model_selection import train_test_split

    X = df["paragraph_cleaned"].values
    X_train_temp, X_test = train_test_split(X, test_size=0.2, random_state=RSEED)

    # Sample 200 for evaluation
    test_texts = X_test[:200]

    print(f"Evaluating on {len(test_texts)} test examples")

    # Results storage
    all_results = {}

    # ========================================
    # LOGISTIC REGRESSION + LIME & SHAP
    # ========================================
    print("\n" + "=" * 60)
    print("EVALUATING: Logistic Regression")
    print("=" * 60)

    logistic_model, class_names = load_logistic_model()

    # LIME
    print("\n--- LIME ---")
    lime_explainer = LIMEExplainer(logistic_model, class_names)

    print("Generating explanations...")
    lime_explanations = [lime_explainer.explain(text) for text in test_texts[:50]]

    # Faithfulness
    faith_eval = FaithfulnessEvaluator(logistic_model)
    lime_faith = faith_eval.evaluate_dataset(test_texts[:50], lime_explanations, k=10)
    print(f"Faithfulness - Sufficiency: {lime_faith['sufficiency_mean']:.3f}")
    print(
        f"Faithfulness - Comprehensiveness: {lime_faith['comprehensiveness_mean']:.3f}"
    )

    # Efficiency
    eff_eval = EfficiencyEvaluator()
    lime_eff = eff_eval.evaluate_explainer(lime_explainer, test_texts[:50], "LIME")
    print(f"Efficiency - Mean time: {lime_eff['mean_time']:.3f}s")

    # Sparsity
    sparse_eval = SparsityEvaluator()
    lime_sparse = sparse_eval.evaluate_dataset(lime_explanations)
    print(
        f"Sparsity - Mean significant features: {lime_sparse['mean_significant_features']:.1f}"
    )

    all_results["logistic_lime"] = {
        "faithfulness": lime_faith,
        "efficiency": lime_eff,
        "sparsity": lime_sparse,
    }

    # SHAP
    print("\n--- SHAP ---")
    shap_explainer = SHAPExplainer(logistic_model, model_type="logistic")
    shap_explainer.fit(X_train_temp[:100], max_samples=50)

    print("Generating explanations...")
    shap_explanations = [
        shap_explainer.explain(text, class_name="Legal/Regulatory Risk")
        for text in test_texts[:50]
    ]

    # Faithfulness
    shap_faith = faith_eval.evaluate_dataset(test_texts[:50], shap_explanations, k=10)
    print(f"Faithfulness - Sufficiency: {shap_faith['sufficiency_mean']:.3f}")
    print(
        f"Faithfulness - Comprehensiveness: {shap_faith['comprehensiveness_mean']:.3f}"
    )

    # Efficiency
    shap_eff = eff_eval.evaluate_explainer(shap_explainer, test_texts[:50], "SHAP")
    print(f"Efficiency - Mean time: {shap_eff['mean_time']:.3f}s")

    # Sparsity
    shap_sparse = sparse_eval.evaluate_dataset(shap_explanations)
    print(
        f"Sparsity - Mean significant features: {shap_sparse['mean_significant_features']:.1f}"
    )

    all_results["logistic_shap"] = {
        "faithfulness": shap_faith,
        "efficiency": shap_eff,
        "sparsity": shap_sparse,
    }

    # ========================================
    # XGBOOST + LIME & SHAP
    # ========================================
    print("\n" + "=" * 60)
    print("EVALUATING: XGBoost")
    print("=" * 60)

    xgb_model, class_names = load_xgboost_model()

    # LIME
    print("\n--- LIME ---")
    lime_explainer_xgb = LIMEExplainer(xgb_model, class_names)

    print("Generating explanations...")
    lime_explanations_xgb = [
        lime_explainer_xgb.explain(text) for text in test_texts[:50]
    ]

    faith_eval_xgb = FaithfulnessEvaluator(xgb_model)
    lime_faith_xgb = faith_eval_xgb.evaluate_dataset(
        test_texts[:50], lime_explanations_xgb, k=10
    )
    print(f"Faithfulness - Sufficiency: {lime_faith_xgb['sufficiency_mean']:.3f}")
    print(
        f"Faithfulness - Comprehensiveness: {lime_faith_xgb['comprehensiveness_mean']:.3f}"
    )

    lime_eff_xgb = eff_eval.evaluate_explainer(
        lime_explainer_xgb, test_texts[:50], "LIME"
    )
    print(f"Efficiency - Mean time: {lime_eff_xgb['mean_time']:.3f}s")

    lime_sparse_xgb = sparse_eval.evaluate_dataset(lime_explanations_xgb)
    print(
        f"Sparsity - Mean significant features: {lime_sparse_xgb['mean_significant_features']:.1f}"
    )

    all_results["xgboost_lime"] = {
        "faithfulness": lime_faith_xgb,
        "efficiency": lime_eff_xgb,
        "sparsity": lime_sparse_xgb,
    }

    # SHAP
    print("\n--- SHAP ---")
    shap_explainer_xgb = SHAPExplainer(xgb_model, model_type="xgboost")
    shap_explainer_xgb.fit(X_train_temp[:100], max_samples=50)

    print("Generating explanations...")
    shap_explanations_xgb = [
        shap_explainer_xgb.explain(text, class_name="Legal/Regulatory Risk")
        for text in test_texts[:50]
    ]

    shap_faith_xgb = faith_eval_xgb.evaluate_dataset(
        test_texts[:50], shap_explanations_xgb, k=10
    )
    print(f"Faithfulness - Sufficiency: {shap_faith_xgb['sufficiency_mean']:.3f}")
    print(
        f"Faithfulness - Comprehensiveness: {shap_faith_xgb['comprehensiveness_mean']:.3f}"
    )

    shap_eff_xgb = eff_eval.evaluate_explainer(
        shap_explainer_xgb, test_texts[:50], "SHAP"
    )
    print(f"Efficiency - Mean time: {shap_eff_xgb['mean_time']:.3f}s")

    shap_sparse_xgb = sparse_eval.evaluate_dataset(shap_explanations_xgb)
    print(
        f"Sparsity - Mean significant features: {shap_sparse_xgb['mean_significant_features']:.1f}"
    )

    all_results["xgboost_shap"] = {
        "faithfulness": shap_faith_xgb,
        "efficiency": shap_eff_xgb,
        "sparsity": shap_sparse_xgb,
    }

    # ========================================
    # SAVE RESULTS
    # ========================================
    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "quantitative_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to {output_dir / 'quantitative_results.json'}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    results = run_evaluation()
