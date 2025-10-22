import os, random
import numpy as np
from typing import List, Tuple
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer)
from datasets import Dataset
from seqeval.metrics import f1_score, classification_report
from src.config import settings

random.seed(42); np.random.seed(42)

def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sents, tags = [], []
    cur_tokens, cur_tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                if cur_tokens:
                    sents.append(cur_tokens); tags.append(cur_tags)
                    cur_tokens, cur_tags = [], []
                continue
            tok, tag = line.split() if " " in line else (line, "O")
            cur_tokens.append(tok); cur_tags.append(tag)
    if cur_tokens:
        sents.append(cur_tokens); tags.append(cur_tags)
    return sents, tags

def build_dataset(conll_path: str):
    tokens, ner_tags = read_conll(conll_path)
    return Dataset.from_dict({"tokens": tokens, "ner_tags": ner_tags})

def main():
    train_ds = build_dataset(str(settings.data_dir / "ner_train.conll"))
    val_ds   = build_dataset(str(settings.data_dir / "ner_val.conll"))

    # label list
    labels = sorted({t for seq in train_ds["ner_tags"] for t in seq} | 
                    {t for seq in val_ds["ner_tags"] for t in seq})
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in enumerate(labels)}

    tok = AutoTokenizer.from_pretrained(settings.ner_model_name)

    def tok_fn(batch):
        tokenized = tok(batch["tokens"], is_split_into_words=True, truncation=True, padding=False, max_length=192)
        aligned_labels = []
        for i, labels_seq in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev = None
            label_ids = []
            for w_id in word_ids:
                if w_id is None:
                    label_ids.append(-100)
                else:
                    label = labels_seq[w_id]
                    # For sub-tokens, use I- tag if continuation
                    if prev == w_id:
                        if label.startswith('B-'): label = 'I-' + label[2:]
                    label_ids.append(label2id[label])
                    prev = w_id
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    train_tok = train_ds.map(tok_fn, batched=True)
    val_tok   = val_ds.map(tok_fn,   batched=True)

    collator = DataCollatorForTokenClassification(tokenizer=tok)

    model = AutoModelForTokenClassification.from_pretrained(
        settings.ner_model_name, num_labels=len(labels),
        id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(settings.ner_hf_dir),
        learning_rate=4e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=6,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=False
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels_ids = p.label_ids
        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(preds, labels_ids):
            tl, tp = [], []
            for p_i, l_i in zip(pred_seq, label_seq):
                if l_i == -100: continue
                tl.append(labels[l_i]); tp.append(labels[p_i])
            true_labels.append(tl); true_preds.append(tp)
        return {"f1": float(f1_score(true_labels, true_preds))}

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        tokenizer=tok, data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(settings.ner_hf_dir)
    tok.save_pretrained(settings.ner_hf_dir)
    with open(settings.ner_hf_dir / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

if __name__ == "__main__":
    os.makedirs(settings.ner_hf_dir, exist_ok=True)
    main()
