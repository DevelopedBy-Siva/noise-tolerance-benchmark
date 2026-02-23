import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from config import NOISE_LEVELS, BASE_DIR
from data.loader import load_sst2
from noise.injector import inject_label_noise
from training.trainer import train

SAVED_MODELS_DIR = BASE_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)

DEMO_SEED = 42


def save_all_models():
    print("loading data...")
    splits = load_sst2(seed=DEMO_SEED)
    train_texts = splits["train"]["texts"]
    train_labels = splits["train"]["labels"]

    for model_name in ["logreg", "distilbert"]:
        print(f"\ntraining and saving {model_name} at each noise level...")

        for noise_level in tqdm(NOISE_LEVELS, desc=model_name):
            noisy_texts, noisy_labels = inject_label_noise(
                train_texts, train_labels, noise_level, seed=DEMO_SEED
            )

            model = train(model_name, noisy_texts, noisy_labels, seed=DEMO_SEED)

            if model_name == "logreg":
                save_path = SAVED_MODELS_DIR / f"logreg_noise_{noise_level}.pkl"
                model.save(str(save_path))
            else:
                save_path = SAVED_MODELS_DIR / f"distilbert_noise_{noise_level}"
                model.save(str(save_path))

            print(f"  saved -> {save_path}")

    print("\nall models saved")


if __name__ == "__main__":
    save_all_models()
