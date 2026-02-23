import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm
from config import SEEDS, TIPPING_POINT_NOISE_LEVELS
from data.loader import load_toxicchat
from noise.injector import inject_label_noise_conditional
from training.trainer import train
from evaluation.evaluator import evaluate, aggregate_across_seeds, save_results


def fit_piecewise_linear(noise_levels, prauc_means):
    """
    Tries every possible breakpoint and picks the one that minimizes
    total squared error across both linear segments.
    Returns the breakpoint noise level and the fitted values for plotting.
    """
    best_breakpoint = None
    best_error = float("inf")
    best_fitted = None

    for i in range(1, len(noise_levels) - 1):
        left_x = noise_levels[: i + 1]
        left_y = prauc_means[: i + 1]
        right_x = noise_levels[i:]
        right_y = prauc_means[i:]

        left_fit = np.polyfit(left_x, left_y, 1)
        right_fit = np.polyfit(right_x, right_y, 1)

        left_pred = np.polyval(left_fit, left_x)
        right_pred = np.polyval(right_fit, right_x)

        error = np.sum((left_y - left_pred) ** 2) + np.sum((right_y - right_pred) ** 2)

        if error < best_error:
            best_error = error
            best_breakpoint = noise_levels[i]
            fitted_left = list(zip(left_x, left_pred))
            fitted_right = list(zip(right_x, right_pred))
            best_fitted = {"left": fitted_left, "right": fitted_right}

    return best_breakpoint, best_fitted


def run_tipping_point():
    print("\nfinding tipping point -- logreg on toxicchat")
    print(f"noise levels: {TIPPING_POINT_NOISE_LEVELS}")
    print(f"seeds: {SEEDS}\n")

    splits = load_toxicchat()
    test_texts = splits["test"]["texts"]
    test_labels = splits["test"]["labels"]
    train_texts = splits["train"]["texts"]
    train_labels = splits["train"]["labels"]

    results = {}
    prauc_means = []
    prauc_stds = []

    for noise_level in tqdm(TIPPING_POINT_NOISE_LEVELS, desc="noise levels"):
        seed_results = []

        for seed in SEEDS:
            noisy_texts, noisy_labels = inject_label_noise_conditional(
                train_texts, train_labels, noise_level, seed=seed
            )

            model = train("logreg", noisy_texts, noisy_labels, seed=seed)
            metrics = evaluate(model, test_texts, test_labels)
            seed_results.append(metrics)

        aggregated = aggregate_across_seeds(seed_results)
        results[str(noise_level)] = aggregated

        prauc_means.append(aggregated["prauc_mean"])
        prauc_stds.append(aggregated["prauc_std"])

        print(
            f"  noise={noise_level:.0%} | "
            f"prauc={aggregated['prauc_mean']:.4f} ± {aggregated['prauc_std']:.4f} | "
            f"f1_macro={aggregated['f1_macro_mean']:.4f} ± {aggregated['f1_macro_std']:.4f}"
        )

    breakpoint_level, fitted = fit_piecewise_linear(
        TIPPING_POINT_NOISE_LEVELS, prauc_means
    )

    breakpoint_idx = TIPPING_POINT_NOISE_LEVELS.index(breakpoint_level)
    breakpoint_std = prauc_stds[breakpoint_idx]

    print(f"\n{'—' * 50}")
    print(f"  tipping point: {breakpoint_level:.0%} noise")
    print(
        f"  prauc at breakpoint: {prauc_means[breakpoint_idx]:.4f} ± {breakpoint_std:.4f}"
    )
    print(
        f"  finding: classifier degrades nonlinearly past {breakpoint_level:.0%} noise (±{breakpoint_std:.4f})"
    )
    print(f"{'—' * 50}")

    output = {
        "noise_levels": TIPPING_POINT_NOISE_LEVELS,
        "results": results,
        "tipping_point": {
            "noise_level": breakpoint_level,
            "prauc_mean": prauc_means[breakpoint_idx],
            "prauc_std": breakpoint_std,
            "finding": f"classifier degrades nonlinearly past {breakpoint_level:.0%} noise (std ±{breakpoint_std:.4f})",
        },
        "piecewise_fit": fitted,
    }

    save_results(output, "tipping_point_toxicchat")
    print("\ndone. results saved to results/tipping_point_toxicchat.json")
    return output


if __name__ == "__main__":
    run_tipping_point()
