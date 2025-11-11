import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.inference import IntentPredictor
from src.config import settings

def main():
    df = pd.read_csv(settings.intent_val_csv)
    clf = IntentPredictor()

    preds, confs = [], []
    for t in df["text"].tolist():
        out = clf.predict(t)
        preds.append(out["intent"])
        confs.append(out["confidence"])

    y_true = df["intent"].tolist()
    acc = accuracy_score(y_true, preds)
    f1m = f1_score(y_true, preds, average="macro", zero_division=0)

    print(f"Accuracy: {acc}")
    print(f"Macro-F1: {f1m}\n")
    print("Report:\n", classification_report(y_true, preds, zero_division=0))

    labels = sorted(list(set(y_true) | set(preds)))
    print("Labels:", labels)
    cm = confusion_matrix(y_true, preds, labels=labels)
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)

    runs = Path("runs"); runs.mkdir(exist_ok=True)
    pd.DataFrame({
        "text": df["text"],
        "gold": y_true,
        "pred": preds,
        "conf": confs
    }).to_csv(runs / "intent_eval.csv", index=False)
    print(f"Saved -> {runs / 'intent_eval.csv'}")

if __name__ == "__main__":
    main()
