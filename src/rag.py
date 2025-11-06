# src/rag.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json, math
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def _load_meta(p: Path) -> List[Dict[str, Any]]:
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else []
    except Exception:
        out = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    out.append(o)
            except Exception:
                pass
        return out

class RAGSearcher:
    def __init__(self, index_dir: str | Path = "data/kb/index"):
        d = Path(index_dir)
        self.index = faiss.read_index(str(d / "faiss.index"))
        self.ids   = np.load(d / "ids.npy", allow_pickle=True).tolist()
        self.ids   = [str(x) for x in self.ids]
        meta_list  = _load_meta(d / "meta.json")
        self.meta  = {str(m.get("id","")): m for m in meta_list}
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def _embed(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros((1, self.index.d), dtype="float32")
        v = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        return v

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        qv = self._embed(query)
        scores, idxs = self.index.search(qv, top_k)
        scores, idxs = scores[0], idxs[0]
        results = []
        for s, i in zip(scores, idxs):
            if i < 0 or i >= len(self.ids): 
                continue
            s = float(s)
            if not math.isfinite(s):
                s = -1e9
            did = self.ids[i]
            m   = self.meta.get(did, {})
            results.append({
                "id": did,
                "score": s,
                "title":   str(m.get("title","") or ""),
                "section": str(m.get("section","") or ""),
                "text":    str(m.get("text","") or ""),
                "source":  str(m.get("source","") or ""),
            })
        return results
