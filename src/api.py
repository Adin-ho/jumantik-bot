# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, warnings

# Hindari spam warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

# ==== LAZY SINGLETONS ====
_INTENT = None
_NER = None
_RAG = None

def get_intent():
    global _INTENT
    if _INTENT is None:
        from src.inference import IntentPredictor
        _INTENT = IntentPredictor()
    return _INTENT

def get_ner():
    global _NER
    if _NER is None:
        from src.inference import NERTagger
        _NER = NERTagger()
    return _NER

def get_rag():
    global _RAG
    if _RAG is None:
        from src.rag import RAGSearcher
        _RAG = RAGSearcher()   # pastikan RAG tidak load apa-apa yang NaN
    return _RAG

class Query(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/intent")
def intent(q: Query):
    pred = get_intent().predict(q.text)
    return pred

@app.post("/ner")
def ner(q: Query):
    ents = get_ner().tag(q.text)
    return ents

@app.post("/answer")
def answer(q: Query):
    out = get_intent().predict(q.text)
    ents = get_ner().tag(q.text)
    from src.inference import compose_answer
    ans = compose_answer(out["intent"], ents)
    return {
        "intent": {"intent": out["intent"], "confidence": float(out["confidence"])},
        "entities": ents,
        "answer": ans,
    }

@app.post("/search")
def search(q: Query):
    try:
        res = get_rag().search(q.text, top_k=3)
        # Pastikan tidak ada NaN
        for r in res:
            if r.get("score") is None or (r["score"] != r["score"]):  # NaN check
                r["score"] = 0.0
        return {"query": q.text, "results": res}
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))

@app.post("/answer_rag")
def answer_rag(q: Query):
    res = get_rag().search(q.text, top_k=3)
    text_chunks = [r.get("text","") for r in res]
    if not text_chunks:
        return {"answer": "Maaf, belum ada referensi di KB."}
    # Ringkas sangat sederhana (placeholder)
    summary = text_chunks[0][:500]
    return {"answer": summary, "sources": res}
