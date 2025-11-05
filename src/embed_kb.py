# src/embed_kb.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import settings

# -------- utils ----------
def _pick(df: pd.DataFrame, candidates: List[str], default: str = "") -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna("")
    return pd.Series([default] * len(df))

def read_kb_csv(path: Path) -> List[Dict]:
    """
    Baca KB dari CSV dengan berbagai kemungkinan header:
      - title, section, text, source
      - pertanyaan, jawaban, (kategori|section), (sumber|source)
      - question, answer, ...
    Return list of dict: {id,title,section,text,source}
    """
    df = pd.read_csv(path)
    # normalisasi nama kolom: lower, strip spasi
    df.columns = [c.strip().lower() for c in df.columns]

    # id opsional
    id_col = _pick(df, ["id", "no", "idx"], default="")
    title_col   = _pick(df, ["title", "pertanyaan", "question"])
    section_col = _pick(df, ["section", "kategori", "topic"])
    text_col    = _pick(df, ["text", "jawaban", "answer", "isi"])
    source_col  = _pick(df, ["source", "sumber", "ref"])

    rows = []
    for i in range(len(df)):
        rid = str(id_col.iloc[i]) if str(id_col.iloc[i]).strip() else str(i)
        rows.append({
            "id": rid,
            "title": str(title_col.iloc[i]).strip(),
            "section": str(section_col.iloc[i]).strip(),
            "text": str(text_col.iloc[i]).strip(),
            "source": str(source_col.iloc[i]).strip(),
        })
    # buang baris kosong total
    rows = [r for r in rows if any(r[k] for k in ("title","text","section"))]
    if not rows:
        raise ValueError(f"CSV {path} tidak memiliki konten yang valid.")
    return rows

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def main():
    kb_csv = settings.kb_csv
    out_dir = settings.kb_index_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    kb = read_kb_csv(kb_csv)

    # gabung teks untuk embedding (judul + section + isi)
    texts = [f"{r['title']} {r['section']} {r['text']}".strip() for r in kb]

    print(f"Memuat model embedding: {settings.kb_embed_model}")
    model = SentenceTransformer(settings.kb_embed_model)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    embs = embs.astype("float32")
    embs = _l2_normalize(embs)  # untuk cosine similarity dengan IndexFlatIP

    # FAISS index (cosine via inner product setelah normalisasi)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # simpan
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "ids.npy", np.array([r["id"] for r in kb], dtype=object))
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"OK → index: {out_dir / 'faiss.index'}")
    print(f"OK → ids:   {out_dir / 'ids.npy'}")
    print(f"OK → meta:  {out_dir / 'meta.json'}")

if __name__ == "__main__":
    main()
