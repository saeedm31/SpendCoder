"""Fine-tune a pretrained transformer to classify purchase lines (Hugging Face Transformers, Trainer API).

What happens, in short:
1. A pretrained model (DistilRoBERTa, 82M parameters) already "knows" English from reading a lot of text.
2. We add a small classification head on top: one output per spend category, randomly initialised.
3. We train the whole network for a few passes (epochs) over our labelled lines, with a small learning rate,
   so the pretrained knowledge is adjusted to our task instead of being overwritten.
4. The loss is class-weighted, so mistakes on rare categories cost more (like the baseline's balanced weights).
5. After each epoch we score the validation year and keep the best checkpoint (by macro F1).

The notebook (notebooks/SpendCoder_finetuning.ipynb) walks through the same steps cell by cell.
Requires: torch, transformers, datasets (not needed for the baseline, tests or serving).
"""

import math

import numpy as np

from .metrics import scores
from .taxonomy import CATEGORIES, ID2LABEL, LABEL2ID

MODEL_NAME = "distilroberta-base"
MAX_LENGTH = 96          # 95% of purchase lines have at most about 92 tokens; longer ones are cut


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def class_weights(labels, n_classes=len(CATEGORIES), power=0.5):
    """Loss weight per category: rarer categories get more weight.

    Balanced weights (n / (k * count)) raised to `power` (0.5 = square root, softer than fully balanced, so training
    stays stable), scaled so the average weight over the training lines is 1.
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.maximum(np.bincount(labels, minlength=n_classes), 1)
    w = (len(labels) / (n_classes * counts)) ** power
    return w * len(labels) / w[labels].sum()


def weighted_trainer_class():
    """A Trainer whose loss is cross-entropy with class weights (imported lazily: needs torch + transformers)."""
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weight = None if self.class_weights is None else self.class_weights.to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def compute_metrics(eval_pred):
    """Called by the Trainer after each epoch on the validation set."""
    logits, labels = eval_pred
    return scores(labels, softmax(np.asarray(logits)))


def train(train_df, val_df, model_name=MODEL_NAME, output_dir="models/spendcoder", epochs=8, learning_rate=5e-5,
          batch_size=32, max_length=MAX_LENGTH, patience=2, weighted=True, seed=42):
    """Fine-tune and return the Trainer (its .model is the best checkpoint, already saved to output_dir)."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, TrainingArguments, set_seed)

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    def to_dataset(df):
        return Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True)).map(tokenize, batched=True)

    train_ds, val_ds = to_dataset(train_df), to_dataset(val_df)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CATEGORIES), id2label=ID2LABEL, label2id=LABEL2ID)

    steps = math.ceil(len(train_ds) / batch_size) * epochs
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=int(0.06 * steps),          # warm up the learning rate for the first 6% of steps
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),          # half precision on a GPU: faster, less memory
        logging_steps=50,
        report_to="none",
        seed=seed,
    )
    weights = class_weights(train_df["label"]) if weighted else None
    trainer = weighted_trainer_class()(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),   # pads each batch to its longest line
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


def predict_proba(model, tokenizer, texts, batch_size=128, max_length=MAX_LENGTH):
    """Category probabilities for a list of texts with a PyTorch model."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[i:i + batch_size]), truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(model.device)
            out.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CATEGORIES)))
