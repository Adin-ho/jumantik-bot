from pathlib import Path
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer)
from seqeval.metrics import f1_score
import numpy as np
from src.config import settings

# pakai backbone sama dengan intent:
MODEL_NAME = settings.intent_model_name
NER_HF_DIR = settings.models_dir / "ner_hf"

def read_conll(path: Path):
    sents, tags = [], []
    cur_toks, cur_tags = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line:
            if cur_toks:
                sents.append(cur_toks); tags.append(cur_tags)
                cur_toks, cur_tags = [], []
            continue
        parts = line.split()
        tok = parts[0]; tag = parts[1] if len(parts)>1 else "O"
        cur_toks.append(tok); cur_tags.append(tag)
    if cur_toks: sents.append(cur_toks); tags.append(cur_tags)
    return sents, tags

def build_ds(train_p: Path, val_p: Path):
    tr_x, tr_y = read_conll(train_p); va_x, va_y = read_conll(val_p)
    return Dataset.from_dict({"tokens":tr_x,"ner_tags":tr_y}), Dataset.from_dict({"tokens":va_x,"ner_tags":va_y})

def main():
    train_ds, val_ds = build_ds(Path("data/ner_train.conll"), Path("data/ner_val.conll"))
    labels = sorted({t for seq in train_ds["ner_tags"] for t in seq} | {t for seq in val_ds["ner_tags"] for t in seq})
    label2id = {l:i for i,l in enumerate(labels)}; id2label = {i:l for l,i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok_fn(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=192)
        all_labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)
            prev = None; lab_ids=[]
            for w in word_ids:
                if w is None: lab_ids.append(-100)
                else:
                    lab = tags[w]
                    if prev == w and lab.startswith("B-"): lab = "I-"+lab[2:]
                    lab_ids.append(label2id[lab]); prev=w
            all_labels.append(lab_ids)
        enc["labels"] = all_labels
        return enc

    tr_tok = train_ds.map(tok_fn, batched=True)
    va_tok = val_ds.map(tok_fn, batched=True)
    coll = DataCollatorForTokenClassification(tokenizer=tok)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        true_l, pred_l = [], []
        for pr, lb in zip(preds, p.label_ids):
            tl, pl = [], []
            for p_i, l_i in zip(pr, lb):
                if l_i == -100: continue
                tl.append(labels[l_i]); pl.append(labels[p_i])
            true_l.append(tl); pred_l.append(pl)
        return {"f1": float(f1_score(true_l, pred_l))}

    args = TrainingArguments(
        output_dir=str(NER_HF_DIR), per_device_train_batch_size=16,
        per_device_eval_batch_size=32, num_train_epochs=5, learning_rate=4e-5,
        evaluation_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=50, report_to=[]
    )

    trainer = Trainer(model=model, args=args, train_dataset=tr_tok, eval_dataset=va_tok,
                      tokenizer=tok, data_collator=coll, compute_metrics=metrics)
    trainer.train()
    NER_HF_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(NER_HF_DIR)
    tok.save_pretrained(NER_HF_DIR)
    (NER_HF_DIR/"labels.txt").write_text("\n".join(labels), encoding="utf-8")
    print("NER saved ->", NER_HF_DIR)

if __name__ == "__main__":
    main()
