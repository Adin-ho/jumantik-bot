# tools/evaluate_intent.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from src.inference import IntentPredictor

VAL = Path("data/intent_val.csv")  # kolom: text,intent

def main():
    assert VAL.exists(), f"File tidak ada: {VAL}"
    df = pd.read_csv(VAL)

    clf = IntentPredictor()
    y_true, y_pred, confs = [], [], []

    for _, r in df.iterrows():
        out = clf.predict(str(r["text"]))
        y_true.append(str(r["intent"]))
        y_pred.append(out["intent"])
        confs.append(out["confidence"])

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("Accuracy:", acc)
    print("Macro-F1:", f1m)
    print(
        "\nReport:\n",
        classification_report(y_true, y_pred, zero_division=0),
    )

    # Confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nLabels:", labels)
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)

    # Simpan hasil baris per baris
    out_dir = Path("runs"); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"text": df["text"], "gold": y_true, "pred": y_pred, "conf": confs}
    ).to_csv(out_dir / "intent_eval.csv", index=False)
    print("Saved ->", out_dir / "intent_eval.csv")

if __name__ == "__main__":
    main()
