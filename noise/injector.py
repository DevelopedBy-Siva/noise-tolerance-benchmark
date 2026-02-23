import random
import numpy as np
from config import SEEDS


def inject_label_noise(texts, labels, noise_level, seed=SEEDS[0]):
    """Randomly flips labels. 0 becomes 1 and 1 becomes 0."""
    rng = random.Random(seed)
    noisy_labels = labels.copy()

    n_to_flip = int(len(noisy_labels) * noise_level)
    indices_to_flip = rng.sample(range(len(noisy_labels)), n_to_flip)

    for i in indices_to_flip:
        noisy_labels[i] = 1 - noisy_labels[i]

    return texts.copy(), noisy_labels


def inject_text_noise(texts, labels, noise_level, seed=SEEDS[0]):
    """
    Corrupts text at the word level. For each selected sample, randomly
    applies one of: word deletion, character swap, or word duplication.
    """
    rng = random.Random(seed)
    noisy_texts = texts.copy()

    n_to_corrupt = int(len(noisy_texts) * noise_level)
    indices_to_corrupt = rng.sample(range(len(noisy_texts)), n_to_corrupt)

    for i in indices_to_corrupt:
        noisy_texts[i] = _corrupt_text(noisy_texts[i], rng)

    return noisy_texts, labels.copy()


def inject_structural_noise(texts, labels, noise_level, seed=SEEDS[0]):
    """
    Adds two kinds of structural garbage to the dataset.
    Half the noise budget duplicates minority class samples heavily.
    The other half inserts meaningless very short strings.
    """
    rng = random.Random(seed)
    noisy_texts = texts.copy()
    noisy_labels = labels.copy()

    n_to_add = int(len(texts) * noise_level)
    half = n_to_add // 2

    minority_label = 0 if labels.count(0) < labels.count(1) else 1
    minority_indices = [i for i, l in enumerate(labels) if l == minority_label]

    for _ in range(half):
        i = rng.choice(minority_indices)
        noisy_texts.append(texts[i])
        noisy_labels.append(labels[i])

    junk_samples = ["ok", "yes", "no", ".", "good", "bad", "fine", "sure"]
    for _ in range(n_to_add - half):
        noisy_texts.append(rng.choice(junk_samples))
        noisy_labels.append(rng.randint(0, 1))

    combined = list(zip(noisy_texts, noisy_labels))
    rng.shuffle(combined)
    noisy_texts, noisy_labels = zip(*combined)

    return list(noisy_texts), list(noisy_labels)


def inject_label_noise_conditional(texts, labels, noise_level, seed=SEEDS[0]):
    """
    Class-conditional label flipping for imbalanced datasets like ToxicChat.
    Toxic labels (1) and non-toxic labels (0) are flipped independently,
    each at the specified noise_level rate. This preserves class prevalence
    better than uniform random flipping, which would disproportionately
    destroy the already-rare toxic signal.
    """
    rng = random.Random(seed)
    noisy_labels = list(labels)

    toxic_indices = [i for i, l in enumerate(labels) if l == 1]
    nontoxic_indices = [i for i, l in enumerate(labels) if l == 0]

    n_toxic_flip = int(len(toxic_indices) * noise_level)
    n_nontoxic_flip = int(len(nontoxic_indices) * noise_level)

    for i in rng.sample(toxic_indices, n_toxic_flip):
        noisy_labels[i] = 0

    for i in rng.sample(nontoxic_indices, n_nontoxic_flip):
        noisy_labels[i] = 1

    return list(texts), noisy_labels


def _corrupt_text(text, rng):
    words = text.split()
    if len(words) < 2:
        return text

    strategy = rng.choice(["delete", "swap_chars", "duplicate"])

    if strategy == "delete":
        idx = rng.randint(0, len(words) - 1)
        words.pop(idx)

    elif strategy == "swap_chars":
        idx = rng.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 2:
            pos = rng.randint(0, len(word) - 2)
            word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
            words[idx] = word

    elif strategy == "duplicate":
        idx = rng.randint(0, len(words) - 1)
        words.insert(idx, words[idx])

    return " ".join(words)
