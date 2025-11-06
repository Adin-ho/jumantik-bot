# src/api.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any
import math

from src.rag import RAGSearcher
from src.inference import IntentPredictor, NERTagger, compose_answer

app = FastAPI()

rag_searcher: Optional[RAGSearcher] = None
intent_pred: Optional[IntentPredictor] = None
ner_tagger: Optional[NERTagger] = None

@app.on_event("startup")
def _startup():
    global rag_searcher, intent_pred, ner_tagger
    try:
        rag_searcher = RAGSearcher("data/kb/index")
        print("[RAG] Index & model siap.")
    except Exception as e:
        rag_searcher = None
        print(f"[RAG] Gagal inisialisasi: {e}")
    try:
        intent_pred = IntentPredictor()
        ner_tagger  = NERTagger()
    except Exception as e:
        print(f"[NLP] Inisialisasi intent/ner gagal: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def search(q: Dict[str, Any]):
    global rag_searcher
    if rag_searcher is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index belum tersedia. Jalankan: python -m src.build_kb && python -m src.embed_kb",
        )
    text = (q or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' wajib diisi.")
    res = rag_searcher.search(text, top_k=3)
    # sanitize terakhir agar aman JSON
    for r in res:
        s = float(r.get("score", 0.0))
        if not math.isfinite(s):
            s = -1e9
        r["score"] = s
    return {"query": text, "results": res}
