from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import Pipeline
except ImportError as e:
    raise ImportError(
        "gate/noise_estimator.py requires scikit-learn. "
        "Run: pip install scikit-learn"
    ) from e

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent
_TIPPING_POINT_JSON = _REPO_ROOT / "results" / "tipping_point_toxicchat.json"
_BASELINE_TOXIC_RATE = 0.073


@dataclass
class BatchSignals:
    n_samples: int
    toxic_rate: float
    toxic_rate_drift: float
    mean_entropy: float
    mean_margin: float
    near_threshold: float
    estimated_prauc: float
    estimated_noise: float
    noise_band: str
    recommended_action: str


def _binary_entropy(p: float) -> float:
    p = max(min(p, 1 - 1e-9), 1e-9)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class _CalibrationCurve:

    def __init__(self, json_path: Path):
        if not json_path.exists():
            raise FileNotFoundError(
                f"Tipping point results not found at {json_path}.\n"
                f"Run: python experiments/find_tipping_point.py"
            )

        with open(json_path) as f:
            data = json.load(f)

        noise_levels = []
        prauc_means = []

        noise_levels.append(0.0)
        prauc_means.append(0.6281)

        for noise_str, result in data["results"].items():
            noise_levels.append(float(noise_str))
            prauc_means.append(result["prauc_mean"])

        self.noise_levels = np.array(noise_levels)
        self.prauc_means = np.array(prauc_means)

        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(-self.prauc_means, self.noise_levels)

    def prauc_to_noise(self, prauc: float) -> float:
        """Given an estimated PR-AUC, return estimated noise rate."""
        estimated = float(self._iso.predict([[-prauc]])[0])
        return round(float(np.clip(estimated, 0.0, 0.5)), 3)

    def noise_to_prauc(self, noise: float) -> float:
        """Given an estimated noise rate, return expected PR-AUC."""
        return round(float(np.interp(noise, self.noise_levels, self.prauc_means)), 4)

    @property
    def clean_prauc(self) -> float:
        return float(self.prauc_means[0])

    @property
    def cliff_noise(self) -> float:
        """The measured tipping point noise level."""
        return 0.20


def _batch_cross_val_proba(
    texts: list[str],
    labels: list[int],
    cv: int = 3,
) -> np.ndarray:

    labels_arr = np.array(labels)
    n_toxic = int(labels_arr.sum())
    n_cv = cv

    if n_toxic < cv * 2:
        if n_toxic >= 4:
            n_cv = 2
        else:
            warnings.warn(
                f"Batch has only {n_toxic} toxic examples — too few for cross-val probe. "
                f"Entropy/margin signals will be uninformative. "
                f"Toxic rate drift will carry full weight.",
                UserWarning,
                stacklevel=3,
            )
            return np.full(len(texts), 0.5)

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(
            pipe,
            texts,
            labels_arr,
            cv=n_cv,
            method="predict_proba",
        )

    return proba[:, 1]


class NoiseEstimator:

    _BANDS = [
        (0.00, 0.10, "CLEAN", "Safe to retrain."),
        (
            0.10,
            0.18,
            "WATCH",
            "Mild noise detected. Consider loss filtering before retraining.",
        ),
        (
            0.18,
            0.25,
            "DANGER",
            "Near or past the 20% cliff. Apply loss filtering. Send boundary examples to adjudication.",
        ),
        (
            0.25,
            1.00,
            "CRITICAL",
            "Severe noise. Do not retrain. Audit label sources before proceeding.",
        ),
    ]

    def __init__(self, tipping_point_json: Path = _TIPPING_POINT_JSON):
        self._curve = _CalibrationCurve(tipping_point_json)

    def estimate(
        self,
        batch_texts: list[str],
        batch_labels: list[int],
        cv: int = 3,
    ) -> BatchSignals:

        n = len(batch_texts)
        if n < 50:
            raise ValueError(
                f"Batch too small ({n} samples). Need at least 50 for reliable estimates."
            )

        print(f"  Running {cv}-fold cross-val probe on batch ({n} samples)...")
        p_toxic = _batch_cross_val_proba(batch_texts, batch_labels, cv=cv)

        entropies = np.array([_binary_entropy(float(p)) for p in p_toxic])
        margins = np.abs(p_toxic - 0.5)
        near_mask = (p_toxic >= 0.4) & (p_toxic <= 0.6)

        toxic_rate = float(np.mean(np.array(batch_labels)))
        toxic_rate_drift = abs(toxic_rate - _BASELINE_TOXIC_RATE)
        mean_entropy = float(np.mean(entropies))
        mean_margin = float(np.mean(margins))
        near_threshold = float(np.mean(near_mask))

        _noise_levels = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
        _entropy_curve = np.array(
            [0.8656, 0.9294, 0.9541, 0.9679, 0.9785, 0.9866, 0.9881, 0.9906, 0.9918]
        )
        _margin_curve = np.array(
            [0.1963, 0.1394, 0.1091, 0.0871, 0.0729, 0.0570, 0.0528, 0.0474, 0.0420]
        )
        _near_curve = np.array(
            [0.1320, 0.2780, 0.4480, 0.6220, 0.7180, 0.8580, 0.8880, 0.9260, 0.9480]
        )

        entropy_vote = float(np.interp(mean_entropy, _entropy_curve, _noise_levels))
        margin_vote = float(
            np.interp(mean_margin, _margin_curve[::-1], _noise_levels[::-1])
        )
        near_vote = float(np.interp(near_threshold, _near_curve, _noise_levels))

        drift_vote = float(np.clip(toxic_rate_drift / 0.15 * 0.10, 0, 0.40))

        signal_noise = (
            0.35 * entropy_vote
            + 0.35 * margin_vote
            + 0.20 * near_vote
            + 0.10 * drift_vote
        )
        signal_noise = float(np.clip(signal_noise, 0.0, 0.40))

        clean_prauc = self._curve.clean_prauc
        estimated_prauc = float(self._curve.noise_to_prauc(signal_noise))

        estimated_noise = round(signal_noise, 3)
        band, action = self._get_band(estimated_noise)

        return BatchSignals(
            n_samples=n,
            toxic_rate=round(toxic_rate, 4),
            toxic_rate_drift=round(toxic_rate_drift, 4),
            mean_entropy=round(mean_entropy, 4),
            mean_margin=round(mean_margin, 4),
            near_threshold=round(near_threshold, 4),
            estimated_prauc=round(estimated_prauc, 4),
            estimated_noise=estimated_noise,
            noise_band=band,
            recommended_action=action,
        )

    def _get_band(self, noise: float) -> tuple[str, str]:
        for lo, hi, band, action in self._BANDS:
            if lo <= noise < hi:
                return band, action
        return "CRITICAL", self._BANDS[-1][3]


def print_calibration_curve() -> None:

    curve = _CalibrationCurve(_TIPPING_POINT_JSON)
    print("\nCalibration curve (from tipping_point_toxicchat.json):")
    print(f"  {'Noise':>8}  {'PR-AUC':>8}")
    print(f"  {'-----':>8}  {'------':>8}")
    for noise, prauc in zip(curve.noise_levels, curve.prauc_means):
        print(f"  {noise:>8.0%}  {prauc:>8.4f}")
    print(f"\n  Cliff at: {curve.cliff_noise:.0%}")
    print(f"  Clean PR-AUC: {curve.clean_prauc:.4f}\n")


if __name__ == "__main__":
    print_calibration_curve()
