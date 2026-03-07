from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from gate.noise_estimator import NoiseEstimator, BatchSignals, print_calibration_curve

SAFE_BANDS = {"CLEAN", "WATCH"}
UNSAFE_BANDS = {"DANGER", "CRITICAL"}

BAND_COLORS = {
    "CLEAN": "\033[92m",
    "WATCH": "\033[93m",
    "DANGER": "\033[91m",
    "CRITICAL": "\033[91m",
}
RESET = "\033[0m"
BOLD = "\033[1m"


def _color(text: str, band: str) -> str:
    return f"{BAND_COLORS.get(band, '')}{text}{RESET}"


def _bar(value: float, width: int = 30, threshold: float = None) -> str:
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    if threshold:
        marker_pos = int(threshold * width)
        bar_list = list(bar)
        if 0 <= marker_pos < width:
            bar_list[marker_pos] = "│"
        bar = "".join(bar_list)
    return f"[{bar}]"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_report(
    batch_path: str,
    signals: BatchSignals,
    safe: bool,
    extra_stats: dict,
) -> None:

    band = signals.noise_band
    c = lambda t: _color(t, band)
    width = 60

    print()
    print("─" * width)
    print(f"  {BOLD}NoiseCliff Quality Gate{RESET}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("─" * width)

    print(f"\n  Batch:    {batch_path}")
    print(f"  Samples:  {signals.n_samples:,}")
    print(
        f"  Toxic:    {_fmt_pct(signals.toxic_rate)}  "
        f"(baseline 7.3%,  drift {_fmt_pct(signals.toxic_rate_drift)})"
    )

    print(
        f"\n  {'Estimated noise rate':<28} "
        f"{c(BOLD + _fmt_pct(signals.estimated_noise) + RESET)}"
    )
    print(
        f"  {'Estimated PR-AUC if trained':<28} "
        f"{c(BOLD + str(signals.estimated_prauc) + RESET)}  "
        f"(clean baseline 0.6281)"
    )
    print(f"  {'Collapse risk':<28} " f"{c(BOLD + band + RESET)}")

    print(f"\n  {'Safe to retrain':<28} " f"{'✓  YES' if safe else c('✗  NO')}")

    print(f"\n  {'─' * (width - 4)}")
    print(f"  Signal breakdown")
    print(f"  {'─' * (width - 4)}")

    signals_table = [
        (
            "Mean entropy",
            signals.mean_entropy,
            0.9785,
            "> 0.9785 = at cliff (20% noise)",
        ),
        (
            "Mean margin |p−0.5|",
            signals.mean_margin,
            0.0729,
            "< 0.0729 = at cliff (20% noise)",
        ),
        (
            "Near-threshold (40-60%)",
            signals.near_threshold,
            0.7180,
            "> 0.7180 = at cliff (20% noise)",
        ),
        ("Toxic rate drift", signals.toxic_rate_drift, 0.15, "> 0.15 = warning"),
    ]

    for label, value, threshold, note in signals_table:
        bar = _bar(min(value, 1.0), width=24, threshold=threshold)
        print(f"  {label:<28} {value:.4f}  {bar}  {note}")

    if extra_stats:
        print(f"\n  {'─' * (width - 4)}")
        print(f"  Batch statistics")
        print(f"  {'─' * (width - 4)}")
        for k, v in extra_stats.items():
            print(f"  {k:<28} {v}")

    print(f"\n  {'─' * (width - 4)}")
    print(f"  Recommendation")
    print(f"  {'─' * (width - 4)}")
    print(f"  {signals.recommended_action}")

    actions = _get_detailed_actions(signals)
    for i, action in enumerate(actions, 1):
        print(f"  {i}. {action}")

    print()
    print("─" * width)
    print()


def _get_detailed_actions(signals: BatchSignals) -> list[str]:
    """Generate specific actions based on which signals are elevated."""
    actions = []

    near_boundary_count = int(signals.near_threshold * signals.n_samples)

    if signals.noise_band in ("DANGER", "CRITICAL"):
        actions.append(
            f"Apply loss filtering before retraining "
            f"(recovers ~0.015 PR-AUC at this noise level based on cleaning study)"
        )

    if signals.near_threshold >= 0.25:
        actions.append(
            f"Send {near_boundary_count:,} near-boundary examples "
            f"({_fmt_pct(signals.near_threshold)} of batch) to adjudication"
        )

    if signals.toxic_rate_drift >= 0.05:
        actions.append(
            f"Toxic rate drifted {_fmt_pct(signals.toxic_rate_drift)} from baseline — "
            f"check if label source composition changed"
        )

    if signals.noise_band in ("DANGER", "CRITICAL"):
        actions.append(
            "Do NOT use confidence filtering — "
            "collapses toxic class prediction on imbalanced data (see cleaning study)"
        )

    if signals.noise_band == "CRITICAL":
        actions.append(
            "Audit individual label sources before proceeding. "
            "At this noise level, no cleaning strategy fully recovers PR-AUC."
        )

    if not actions:
        actions.append("No corrective actions required.")

    return actions


def load_batch(csv_path: str) -> tuple[list[str], list[int], dict]:
    df = pd.read_csv(csv_path)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\n"
            f"Expected columns: text, label\n"
            f"Found columns: {list(df.columns)}"
        )

    # drop rows with null text or label
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    after = len(df)
    if before != after:
        print(f"  Warning: dropped {before - after} rows with null values")

    invalid = df[~df["label"].isin([0, 1])]
    if len(invalid) > 0:
        raise ValueError(
            f"Labels must be 0 or 1. Found {len(invalid)} invalid values: "
            f"{df['label'].unique().tolist()}"
        )

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    # extra stats to surface in report
    extra_stats = {
        "Total rows": f"{len(df):,}",
        "Toxic (label=1)": f"{labels.count(1):,}  ({labels.count(1)/len(labels)*100:.1f}%)",
        "Non-toxic (label=0)": f"{labels.count(0):,}  ({labels.count(0)/len(labels)*100:.1f}%)",
        "Avg text length": f"{df['text'].str.len().mean():.0f} chars",
    }

    # optional: labeler_id column
    if "labeler_id" in df.columns:
        n_labelers = df["labeler_id"].nunique()
        extra_stats["Labelers"] = str(n_labelers)
        # [TODO] labeler disagreement rate -- requires multi-label per example
        # [TODO] per-labeler outlier score

    return texts, labels, extra_stats


def load_baseline(baseline: str) -> tuple[list[str], list[int]]:
    if baseline == "toxicchat":
        print(f"  Loading ToxicChat baseline...")
        splits = load_toxicchat()
        texts = splits["train"]["texts"]
        labels = splits["train"]["labels"]
        print(
            f"  Baseline: {len(texts):,} samples  "
            f"(toxic: {sum(labels):,} / {len(labels):,}  "
            f"= {sum(labels)/len(labels)*100:.1f}%)"
        )
        return texts, labels
    else:
        raise ValueError(
            f"Unknown baseline: '{baseline}'. "
            f"Currently supported: toxicchat\n"
            f"To add a custom baseline, pass a CSV path instead."
        )


def run_gate(
    batch_path: str,
    baseline: str = "toxicchat",
    save: bool = False,
    ci: bool = False,
    show_curve: bool = False,
) -> BatchSignals:

    if show_curve:
        print_calibration_curve()

    print(f"\n  Loading batch: {batch_path}")
    batch_texts, batch_labels, extra_stats = load_batch(batch_path)

    estimator = NoiseEstimator()
    signals = estimator.estimate(batch_texts, batch_labels)

    safe = signals.noise_band in SAFE_BANDS

    print_report(batch_path, signals, safe, extra_stats)

    if save:
        output = {
            "timestamp": datetime.now().isoformat(),
            "batch": batch_path,
            "baseline": baseline,
            "safe_to_retrain": safe,
            "signals": {
                "n_samples": signals.n_samples,
                "toxic_rate": signals.toxic_rate,
                "toxic_rate_drift": signals.toxic_rate_drift,
                "mean_entropy": signals.mean_entropy,
                "mean_margin": signals.mean_margin,
                "near_threshold": signals.near_threshold,
                "estimated_prauc": signals.estimated_prauc,
                "estimated_noise": signals.estimated_noise,
                "noise_band": signals.noise_band,
                "recommended_action": signals.recommended_action,
            },
        }
        out_path = (
            Path("results")
            / f"gate_{Path(batch_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Report saved → {out_path}\n")

    if ci and not safe:
        print(f"  CI mode: exiting with code 1 (unsafe to retrain)\n")
        sys.exit(1)

    return signals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NoiseCliff pre-training quality gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gate/check.py --batch new_labels.csv --baseline toxicchat
  python gate/check.py --batch new_labels.csv --baseline toxicchat --save
  python gate/check.py --batch new_labels.csv --baseline toxicchat --ci
  python gate/check.py --calibration-curve
        """,
    )
    parser.add_argument(
        "--batch", type=str, help="Path to new label batch CSV (text, label columns)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="toxicchat",
        help="Baseline dataset to calibrate against (default: toxicchat)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save full report to results/ as JSON"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with code 1 if unsafe to retrain (for CI pipelines)",
    )
    parser.add_argument(
        "--calibration-curve",
        action="store_true",
        help="Print calibration curve and exit",
    )

    args = parser.parse_args()

    if args.calibration_curve:
        print_calibration_curve()
        sys.exit(0)

    if not args.batch:
        parser.error("--batch is required unless --calibration-curve is set")

    if not Path(args.batch).exists():
        print(f"\n  Error: batch file not found: {args.batch}\n")
        sys.exit(1)

    run_gate(
        batch_path=args.batch,
        baseline=args.baseline,
        save=args.save,
        ci=args.ci,
    )


if __name__ == "__main__":
    main()
