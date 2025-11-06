from pathlib import Path
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
from src.config import settings

def _assert_data():
    if not settings.intent_train_csv.exists():
        raise FileNotFoundError(f"Missing: {settings.intent_train_csv}")
    if not settings.intent_val_csv.exists():
        raise FileNotFoundError(f"Missing: {settings.intent_val_csv}")

def load_intent_datasets() -> tuple[DatasetDict, list[str], dict, dict]:
    files = {
        "train": str(settings.intent_train_csv),
        "validation": str(settings.intent_val_csv),
    }
    ds = load_dataset("csv", data_files=files)

    # kumpulkan label unik
    labels = sorted(list(set(ds["train"]["intent"]) | set(ds["validation"]["intent"])))
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    def mapper(ex):
        ex["label"] = label2id[ex["intent"]]
        return ex

    ds = ds.map(mapper)
    return ds, labels, label2id, id2label

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro", zero_division=0)
    return {"eval_acc": acc, "eval_f1_macro": f1m}

def main():
    _assert_data()
    tok = AutoTokenizer.from_pretrained(settings.intent_model_name)

    ds, labels, label2id, id2label = load_intent_datasets()

    def tok_map(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=128,
        )

    ds = ds.map(tok_map, batched=True)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ["input_ids","token_type_ids","attention_mask","label"]])
    ds.set_format(type="torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.intent_model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    collator = DataCollatorWithPadding(tokenizer=tok)

    training_args = TrainingArguments(
        output_dir=str(settings.intent_hf_dir),
        num_train_epochs=8,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        seed=42,
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # simpan tokenizer + labels
    settings.intent_hf_dir.mkdir(parents=True, exist_ok=True)
    (settings.intent_hf_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    trainer.save_model(settings.intent_hf_dir)
    tok.save_pretrained(settings.intent_hf_dir)
    print(f"Training selesai → {settings.intent_hf_dir}")

if __name__ == "__main__":
    main()
