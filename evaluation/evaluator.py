import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from config import RESULTS_DIR


def evaluate(model, texts, labels):
    """
    Runs prediction and returns PR-AUC, macro F1, weighted F1, and accuracy.
    PR-AUC is the primary metric for imbalanced classification like ToxicChat.
    """
    predictions = model.predict(texts)
    proba = model.predict_proba(texts)
    pos_proba = proba[:, 1]

    return {
        "prauc": average_precision_score(labels, pos_proba),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "accuracy": accuracy_score(labels, predictions),
    }


def aggregate_across_seeds(results_per_seed):
    """
    Takes a list of dicts like [{"prauc": 0.91, "f1_macro": 0.90, ...}, ...]
    and returns mean and std for each metric.
    """
    metrics = list(results_per_seed[0].keys())
    aggregated = {}

    for metric in metrics:
        values = [r[metric] for r in results_per_seed]
        aggregated[f"{metric}_mean"] = round(float(np.mean(values)), 4)
        aggregated[f"{metric}_std"] = round(float(np.std(values)), 4)

    return aggregated


def save_results(results, filename):
    """Dumps results as JSON into the results directory."""
    path = RESULTS_DIR / f"{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved → {path}")


def load_results(filename):
    path = RESULTS_DIR / f"{filename}.json"
    with open(path, "r") as f:
        return json.load(f)
