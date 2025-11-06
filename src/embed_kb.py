# src/embed_kb.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from src.config import settings

def read_kb_csv(path: Path):
    df = pd.read_csv(path)
    req = ["id","title","section","text","source"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ada di {path}")
    ids   = df["id"].astype(str).tolist()
    texts = (df["title"].fillna("")+" "+df["section"].fillna("")+" "+df["text"].fillna("")).astype(str).tolist()
    meta  = df[["id","title","section","text","source"]].to_dict(orient="records")
    return ids, texts, meta

def main():
    out_dir = Path(settings.kb_index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, texts, meta = read_kb_csv(settings.kb_csv)

    print("Memuat model embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # embed → float32, normalisasi, bebas NaN/Inf
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embs = embs.astype("float32")
    embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # gunakan Inner Product karena sudah dinormalisasi
    index.add(embs)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    # simpan ids sebagai object + allow_pickle saat load
    np.save(out_dir / "ids.npy", np.array(ids, dtype=object), allow_pickle=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    print(f"OK → index: {out_dir / 'faiss.index'}")
    print(f"OK → ids:   {out_dir / 'ids.npy'}")
    print(f"OK → meta:  {out_dir / 'meta.json'}")

if __name__ == "__main__":
    main()
