import os, random
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score
from src.config import settings

random.seed(42); np.random.seed(42)

def load_intent_datasets():
    data_files = {
        "train": str(settings.data_dir / "intent_train.csv"),
        "validation": str(settings.data_dir / "intent_val.csv")
    }
    ds = load_dataset("csv", data_files=data_files)
    # Build label list
    labels = sorted(set(ds["train"]["intent"]))
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    def map_fn(batch):
        batch["label"] = [label2id[l] for l in batch["intent"]]
        return batch
    ds = ds.map(map_fn)
    return ds, labels, label2id, id2label

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        "acc": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro"))
    }

def main():
    ds, labels, label2id, id2label = load_intent_datasets()
    tok = AutoTokenizer.from_pretrained(settings.intent_model_name)
    def tok_fn(batch):
        return tok(batch["text"], truncation=True, padding=False, max_length=128)
    ds = ds.map(tok_fn, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.intent_model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(settings.intent_hf_dir),
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_compare="f1_macro",
        logging_steps=50,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(settings.intent_hf_dir)
    tok.save_pretrained(settings.intent_hf_dir)
    with open(settings.intent_hf_dir / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

if __name__ == "__main__":
    os.makedirs(settings.intent_hf_dir, exist_ok=True)
    main()
