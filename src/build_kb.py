from __future__ import annotations
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import KB_CSV, KB_VECT_PATH, KB_TEXTS_PATH
from .simple_nlp import normalize_text

def main():
    df = pd.read_csv(KB_CSV, encoding="utf-8-sig")
    assert {"title", "text"}.issubset(df.columns), "kb.csv wajib kolom: title,text"

    texts = (df["title"].astype(str) + " . " + df["text"].astype(str)).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    vec.fit([normalize_text(t) for t in texts])

    KB_VECT_PATH.write_bytes(pickle.dumps(vec))
    KB_TEXTS_PATH.write_bytes(pickle.dumps(texts))
    print(f"✅ KB built: {KB_VECT_PATH.name} {KB_TEXTS_PATH.name}")

if __name__ == "__main__":
    main()
