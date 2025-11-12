# src/train_intent.py
from __future__ import annotations

import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score

from .config import (
    INTENT_TRAIN, INTENT_VAL,
    INTENT_VECT_PATH, INTENT_CLF_PATH,
)
from .simple_nlp import normalize_text


def assert_file(path: Path, label: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{label} tidak ditemukan di: {path}\n"
            "Pastikan file ada di `data/kb/` atau `data/` dengan header `text,intent`."
        )


def main():
    assert_file(INTENT_TRAIN, "INTENT_TRAIN")
    assert_file(INTENT_VAL, "INTENT_VAL")

    df_tr = pd.read_csv(INTENT_TRAIN, encoding="utf-8-sig").dropna()
    df_va = pd.read_csv(INTENT_VAL,   encoding="utf-8-sig").dropna()

    Xtr = df_tr["text"].astype(str).apply(normalize_text)
    ytr = df_tr["intent"].astype(str)

    Xva = df_va["text"].astype(str).apply(normalize_text)
    yva = df_va["intent"].astype(str)

    # TF-IDF + Logistic Regression (multinomial) — aman untuk dataset kecil
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=None,
        class_weight="balanced",   # bantu kelas minoritas
        multi_class="auto",        # otomatis multinomial dgn lbfgs
        solver="lbfgs",
    )

    pipe = make_pipeline(vec, clf)
    pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xva)

    print(classification_report(yva, yhat))
    print("Macro F1:", f1_score(yva, yhat, average="macro"))

    # simpan komponen pipeline agar kompatibel dgn loader di simple_nlp
    vec_fitted = pipe.named_steps["tfidfvectorizer"]
    clf_fitted = pipe.named_steps["logisticregression"]

    INTENT_VECT_PATH.write_bytes(pickle.dumps(vec_fitted))
    INTENT_CLF_PATH.write_bytes(pickle.dumps(clf_fitted))
    print(f"Saved -> {INTENT_VECT_PATH.name}, {INTENT_CLF_PATH.name}")


if __name__ == "__main__":
    main()
