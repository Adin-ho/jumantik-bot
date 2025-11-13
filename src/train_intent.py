from __future__ import annotations
import pickle
from collections import Counter
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from .config import (
    INTENT_TRAIN, INTENT_VAL,
    INTENT_VECT_PATH, INTENT_CLF_PATH, INTENT_LABELS_PATH,
)
from .simple_nlp import normalize_text

def _load_df():
    if not INTENT_TRAIN.exists():
        print("❌ File training intent tidak ditemukan.")
        print("Working dir  :", INTENT_TRAIN.parents[2])
        print("Data dir      :", INTENT_TRAIN.parent)
        print("Harus ada file berikut:")
        print(f"- {INTENT_TRAIN}")
        print("\nLetakkan CSV di salah satu lokasi ini:")
        print(f"  1) {INTENT_TRAIN.parent}  (direkomendasikan)")
        print("  2) set env JUMANTIK_DATA_DIR=path_kamu")
        raise SystemExit(1)
    df_tr = pd.read_csv(INTENT_TRAIN, encoding="utf-8-sig").dropna(subset=["text", "intent"])
    df_val = pd.read_csv(INTENT_VAL, encoding="utf-8-sig").dropna(subset=["text", "intent"]) if INTENT_VAL.exists() else None
    return df_tr, df_val

def main():
    df_tr, _ = _load_df()
    Xtr = df_tr["text"].astype(str).apply(normalize_text)
    ytr = df_tr["intent"].astype(str)

    # Tentukan CV aman (min jumlah sampel per kelas)
    counts = Counter(ytr)
    min_count = min(counts.values())
    if min_count >= 3:
        cv = 3
    elif min_count == 2:
        cv = 2
    else:
        cv = 0  # tak bisa calibrate

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    base = LinearSVC()

    if cv >= 2:
        clf = CalibratedClassifierCV(base, cv=cv)
        pipe = make_pipeline(vec, clf)
        pipe.fit(Xtr, ytr)
        vect = pipe.named_steps["tfidfvectorizer"]
        final_clf = pipe.named_steps["calibratedclassifiercv"]
    else:
        # fallback: tanpa calibrate (nanti proba dihitung manual di simple_nlp)
        pipe = make_pipeline(vec, base)
        pipe.fit(Xtr, ytr)
        vect = pipe.named_steps["tfidfvectorizer"]
        final_clf = pipe.named_steps["linearsvc"]

    INTENT_VECT_PATH.write_bytes(pickle.dumps(vect))
    INTENT_CLF_PATH.write_bytes(pickle.dumps(final_clf))
    INTENT_LABELS_PATH.write_bytes(pickle.dumps(getattr(final_clf, "classes_", [])))
    print("✅ Intent model saved:",
          INTENT_VECT_PATH.name, INTENT_CLF_PATH.name, INTENT_LABELS_PATH.name)

if __name__ == "__main__":
    main()
