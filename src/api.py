from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from src.inference import IntentPredictor, NERTagger, RAG, compose_answer

app = FastAPI(title="Jumantik Chatbot API", version="0.1.0")

intent = IntentPredictor()
ner = NERTagger()
rag = None
try:
    rag = RAG()
except Exception:
    rag = None

class QueryIn(BaseModel):
    text: str

class IntentOut(BaseModel):
    intent: str
    confidence: float

class Entity(BaseModel):
    type: str
    text: str

class SearchItem(BaseModel):
    title: str
    url: str | None = None
    section: str | None = None
    text: str
    score: float

class AnswerOut(BaseModel):
    intent: IntentOut
    entities: List[Entity] = []
    passages: List[SearchItem] = []
    answer: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/intent", response_model=IntentOut)
def predict_intent(q: QueryIn):
    pi = intent.predict(q.text)
    return IntentOut(**pi)

@app.post("/ner", response_model=List[Entity])
def predict_ner(q: QueryIn):
    ents = [Entity(**e) for e in ner.tag(q.text)]
    return ents

@app.post("/search", response_model=List[SearchItem])
def search(q: QueryIn):
    if not rag: return []
    res = [SearchItem(**r) for r in rag.search(q.text)]
    return res

@app.post("/answer", response_model=AnswerOut)
def answer(q: QueryIn):
    pi = intent.predict(q.text)
    ents = [e for e in ner.tag(q.text)]
    passages = rag.search(q.text, k=5) if rag else []
    ans = compose_answer(pi["intent"], ents, passages)
    return AnswerOut(intent=pi, entities=[Entity(**e) for e in ents],
                     passages=[SearchItem(**p) for p in passages],
                     answer=ans)
