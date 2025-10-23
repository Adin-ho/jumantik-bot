from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import IntentPredictor, NERTagger, compose_answer, extract_structured

app = FastAPI(title="Jumantik Chatbot API")

# init model sekali di startup
intent = IntentPredictor()
ner = NERTagger()

# RAG opsional (aman bila tidak ada)
try:
    from src.rag import Retriever, synthesize_answer
    try:
        retriever = Retriever()  # misal: load models/rag/index.pkl
    except Exception:
        retriever = None
except Exception:
    retriever = None

class QueryIn(BaseModel):
    text: str

class Entity(BaseModel):
    type: str
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/intent")
def predict_intent(q: QueryIn):
    return intent.predict(q.text)

@app.post("/ner")
def predict_ner(q: QueryIn) -> List[Entity]:
    ents = ner.tag(q.text)
    return [Entity(**e) for e in ents]

@app.post("/answer")
def answer(q: QueryIn):
    pi = intent.predict(q.text)
    ents = ner.tag(q.text)
    fields = extract_structured(q.text, ents)
    ans = compose_answer(pi["intent"], ents)
    return {"intent": pi, "entities": ents, "fields": fields, "answer": ans}

@app.post("/search")
def search(q: QueryIn):
    if retriever is None:
        raise HTTPException(status_code=404, detail="RAG index belum tersedia. Jalankan: python -m src.rag_build")
    return retriever.search(q.text, k=5)

INTENT_THRESHOLD = 0.35

@app.post("/answer")
def answer(q: QueryIn):
    pi = intent.predict(q.text)
    ents = ner.tag(q.text)
    fields = extract_structured(q.text, ents)
    if pi["confidence"] < INTENT_THRESHOLD:
        return {
            "intent": pi, "entities": ents, "fields": fields,
            "answer": "Saya belum yakin maksudnya. Apakah ingin tanya jadwal jumantik, lapor jentik, atau prosedur fogging?"
        }
    ans = compose_answer(pi["intent"], ents)
    return {"intent": pi, "entities": ents, "fields": fields, "answer": ans}
