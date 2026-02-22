import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from config import SEEDS, NOISE_LEVELS
from data.loader import load_sst2
from noise.injector import inject_label_noise
from training.trainer import train
from evaluation.evaluator import evaluate, aggregate_across_seeds, save_results
from cleaning.strategies import confidence_filter, loss_filter, heuristic_filter


CLEANING_STRATEGIES = {
    "confidence": lambda texts, labels, model: confidence_filter(texts, labels, model),
    "loss": lambda texts, labels, model: loss_filter(texts, labels, model),
    "heuristic": lambda texts, labels, _model: heuristic_filter(texts, labels),
}

NOISE_LEVELS_TO_STUDY = [nl for nl in NOISE_LEVELS if nl > 0.0]


def run_cleaning(model_name):
    print(f"\nrunning cleaning experiment — {model_name}")
    print(f"noise levels: {NOISE_LEVELS_TO_STUDY}")
    print(f"strategies: {list(CLEANING_STRATEGIES.keys())}\n")

    splits = load_sst2()
    test_texts = splits["test"]["texts"]
    test_labels = splits["test"]["labels"]
    train_texts = splits["train"]["texts"]
    train_labels = splits["train"]["labels"]

    results = {}

    for noise_level in tqdm(NOISE_LEVELS_TO_STUDY, desc="noise levels"):
        results[str(noise_level)] = {}

        noisy_baseline_results = []
        cleaned_results = {strategy: [] for strategy in CLEANING_STRATEGIES}

        for seed in SEEDS:
            noisy_texts, noisy_labels = inject_label_noise(
                train_texts, train_labels, noise_level, seed=seed
            )

            noisy_model = train(model_name, noisy_texts, noisy_labels, seed=seed)
            noisy_metrics = evaluate(noisy_model, test_texts, test_labels)
            noisy_baseline_results.append(noisy_metrics)

            for strategy_name, strategy_fn in CLEANING_STRATEGIES.items():
                cleaned_texts, cleaned_labels = strategy_fn(
                    noisy_texts, noisy_labels, noisy_model
                )

                cleaned_model = train(
                    model_name, cleaned_texts, cleaned_labels, seed=seed
                )
                cleaned_metrics = evaluate(cleaned_model, test_texts, test_labels)
                cleaned_results[strategy_name].append(cleaned_metrics)

        results[str(noise_level)]["noisy_baseline"] = aggregate_across_seeds(
            noisy_baseline_results
        )

        for strategy_name, seed_results in cleaned_results.items():
            results[str(noise_level)][strategy_name] = aggregate_across_seeds(
                seed_results
            )

        _print_noise_level_summary(noise_level, results[str(noise_level)])

    save_results(results, f"cleaning_{model_name}")
    print(f"\ndone. results saved to results/cleaning_{model_name}.json")
    return results


def _print_noise_level_summary(noise_level, level_results):
    print(f"\n  noise={noise_level:.0%}")
    for strategy, metrics in level_results.items():
        print(
            f"    {strategy:<20} | "
            f"acc={metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f} | "
            f"f1={metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}"
        )


if __name__ == "__main__":
    for model_name in ["logreg", "distilbert"]:
        run_cleaning(model_name)
