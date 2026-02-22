from datasets import load_dataset
from sklearn.model_selection import train_test_split
from config import DATA_DIR, SEEDS


def load_sst2(seed=SEEDS[0]):
    dataset = load_dataset("glue", "sst2", cache_dir=str(DATA_DIR))

    train_texts = dataset["train"]["sentence"]
    train_labels = dataset["train"]["label"]

    val_texts = dataset["validation"]["sentence"]
    val_labels = dataset["validation"]["label"]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        random_state=seed,
        stratify=train_labels,
    )

    return {
        "train": {"texts": list(train_texts), "labels": list(train_labels)},
        "val": {"texts": list(val_texts), "labels": list(val_labels)},
        "test": {"texts": list(test_texts), "labels": list(test_labels)},
    }


def load_sst2_subset(n_samples, seed=SEEDS[0]):
    """
    caps the training set at n_samples.
    Used in the quantity vs quality experiment.
    """
    splits = load_sst2(seed=seed)

    texts = splits["train"]["texts"]
    labels = splits["train"]["labels"]

    if n_samples >= len(texts):
        return splits

    subset_texts, _, subset_labels, _ = train_test_split(
        texts,
        labels,
        train_size=n_samples,
        random_state=seed,
        stratify=labels,
    )

    splits["train"]["texts"] = subset_texts
    splits["train"]["labels"] = subset_labels

    return splits
