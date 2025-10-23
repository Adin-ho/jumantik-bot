import sys
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
from src.config import settings

def _assert_data():
    missing = []
    if not settings.intent_train_csv.exists():
        missing.append(str(settings.intent_train_csv))
    if not settings.intent_val_csv.exists():
        missing.append(str(settings.intent_val_csv))
    if missing:
        raise FileNotFoundError("File dataset hilang:\n- " + "\n- ".join(missing))

def load_intent_datasets():
    data_files = {
        "train": str(settings.intent_train_csv),
        "validation": str(settings.intent_val_csv),
    }
    ds = load_dataset("csv", data_files=data_files)

    labels = sorted(set(ds["train"]["intent"]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    def map_fn(batch):
        return {"label": [label2id[x] for x in batch["intent"]]}

    # ⬇️ INI FIX PENTING
    ds = ds.map(map_fn, batched=True)

    return ds, labels, label2id, id2label

def compute_metrics(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"acc": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro"))}

def main():
    _assert_data()
    ds, labels, label2id, id2label = load_intent_datasets()

    tok = AutoTokenizer.from_pretrained(settings.intent_model_name)
    ds = ds.map(lambda b: tok(b["text"], truncation=True), batched=True)
    collator = DataCollatorWithPadding(tokenizer=tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.intent_model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=str(settings.intent_hf_dir),
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to=[],
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
    settings.intent_hf_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(settings.intent_hf_dir)
    tok.save_pretrained(settings.intent_hf_dir)
    (settings.intent_hf_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    print("Training selesai →", settings.intent_hf_dir)

if __name__ == "__main__":
    main()
