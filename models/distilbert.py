import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
from config import DISTILBERT_CONFIG

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class SST2Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class DistilBertModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            DISTILBERT_CONFIG["model_name"]
        )
        self.model = None

    def _tokenize(self, texts):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=DISTILBERT_CONFIG["max_length"],
        )

    def fit(self, texts, labels):
        encodings = self._tokenize(list(texts))
        dataset = SST2Dataset(encodings, list(labels))
        loader = DataLoader(
            dataset,
            batch_size=DISTILBERT_CONFIG["batch_size"],
            shuffle=True,
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_CONFIG["model_name"],
            num_labels=2,
        ).to(self.device)

        optimizer = AdamW(
            self.model.parameters(),
            lr=DISTILBERT_CONFIG["learning_rate"],
        )

        total_steps = len(loader) * DISTILBERT_CONFIG["epochs"]
        warmup_steps = int(total_steps * DISTILBERT_CONFIG["warmup_ratio"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(DISTILBERT_CONFIG["epochs"]):
            loop = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
            for batch in loop:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_batch = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch,
                )

                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
                loop.set_postfix(loss=outputs.loss.item())

        return self

    def predict_proba(self, texts):
        encodings = self._tokenize(list(texts))
        dataset = SST2Dataset(encodings, [0] * len(texts))
        loader = DataLoader(
            dataset,
            batch_size=DISTILBERT_CONFIG["batch_size"],
            shuffle=False,
        )

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.extend(probs)

        return np.array(all_probs)

    def predict(self, texts):
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1).tolist()

    def get_loss_per_sample(self, texts, labels):
        """
        Per-sample log loss. Same contract as LogRegModel.
        Higher means the model is more uncertain or wrong about that sample.
        """
        proba = self.predict_proba(texts)
        eps = 1e-9
        losses = []
        for i, label in enumerate(labels):
            p = np.clip(proba[i][label], eps, 1 - eps)
            losses.append(-np.log(p))
        return losses

    def save(self, path):
        """
        Saves the model and tokenizer to a directory.
        Path should be something like saved_models/distilbert_noise_0.0/
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path):
        """
        Loads a saved model from disk and returns a ready-to-use DistilBertModel.
        """
        instance = cls.__new__(cls)
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        instance.model = DistilBertForSequenceClassification.from_pretrained(path).to(
            instance.device
        )
        instance.model.eval()
        return instance
