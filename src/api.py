# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
import numpy as np

from src.config import settings
from src.inference import IntentPredictor, NERTagger, compose_answer

# ==== APP DULU! ====
app = FastAPI(title="Jumantik Bot API")

# ---------- Models ----------
class Query(BaseModel):
    text: str

class Entity(BaseModel):
    type: str
    text: str

class RagQuery(BaseModel):
    text: str
    top_k: int = 3

# ---------- Global singletons ----------
intent_pred: Optional[IntentPredictor] = None
ner_tagger: Optional[NERTagger] = None

# RAG bits
kb_index = None
kb_ids = None
kb_meta = None
kb_model = None  # sentence-transformers model (lazy import)

# ---------- Startup ----------
@app.on_event("startup")
def _startup():
    global intent_pred, ner_tagger
    intent_pred = IntentPredictor()
    ner_tagger = NERTagger()
    _rag_load()

def _rag_load():
    """Load FAISS index & embedder bila tersedia (silent jika belum ada)."""
    global kb_index, kb_ids, kb_meta, kb_model
    try:
        import faiss  # import lokal supaya tidak wajib terpasang saat tidak pakai RAG
        from sentence_transformers import SentenceTransformer

        idx_dir = settings.kb_index_dir
        idx_path = idx_dir / "faiss.index"
        ids_path = idx_dir / "ids.npy"
        meta_path = idx_dir / "meta.json"

        if not (idx_path.exists() and ids_path.exists() and meta_path.exists()):
            print("[RAG] Index belum tersedia, lewati.")
            return

        kb_index = faiss.read_index(str(idx_path))
        kb_ids = np.load(ids_path, allow_pickle=True)
        kb_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kb_model = SentenceTransformer(settings.kb_embed_model)
        print("[RAG] Index & model siap.")
    except Exception as e:
        print(f"[RAG] Tidak aktif: {e}")

# ---------- Helpers ----------
def _rag_search(q: str, top_k: int = 3):
    if kb_index is None or kb_model is None:
        raise HTTPException(status_code=404, detail="RAG index belum tersedia. Jalankan: python -m src.embed_kb")
    qv = kb_model.encode([q], convert_to_numpy=True).astype("float32")
    D, I = kb_index.search(qv, top_k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        mid = str(kb_ids[idx])
        m = kb_meta[idx]
        hits.append({
            "id": mid,
            "score": float(score),
            "title": m.get("title"),
            "section": m.get("section"),
            "text": m.get("text"),
            "source": m.get("source")
        })
    return hits

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/intent")
def predict_intent(q: Query):
    out = intent_pred.predict(q.text)
    return out

@app.post("/ner", response_model=List[Entity])
def predict_ner(q: Query):
    ents = ner_tagger.tag(q.text)
    return [Entity(**e) for e in ents]

@app.post("/answer")
def answer(q: Query):
    intent = intent_pred.predict(q.text)
    ents = ner_tagger.tag(q.text)

    # normalisasi field sederhana
    fields = {"RT": None, "RW": None, "NAMA": None, "ALAMAT": None}
    for e in ents:
        t = e["type"].upper()
        if t in ("RT", "RW"):
            # ambil digit terakhir di teks (mis. "05" -> "5")
            import re
            m = re.search(r"\d+", e["text"])
            if m:
                fields[t] = m.group(0).lstrip("0") or "0"
        elif t in ("NAMA",):
            fields["NAMA"] = e["text"]
        elif t in ("ALAMAT",):
            # heuristik: simpan nama jalan/daerah yang terdeteksi
            fields["ALAMAT"] = e["text"] if e["text"] else fields["ALAMAT"]

    ans = compose_answer(intent["intent"], ents)
    return {"intent": intent, "entities": ents, "fields": fields, "answer": ans}

# ---- RAG optional ----
@app.post("/search")
def search(q: RagQuery):
    return {"query": q.text, "results": _rag_search(q.text, q.top_k)}

@app.post("/answer_rag")
def answer_rag(q: RagQuery):
    hits = _rag_search(q.text, q.top_k)
    if not hits:
        return {"answer": "Maaf, tidak ada referensi yang relevan ditemukan.", "results": []}
    context = " ".join(h["text"] for h in hits if h.get("text"))
    return {"answer": f"Ringkasan dari basis pengetahuan: {context}", "results": hits}
