from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from fastapi.responses import HTMLResponse

from .simple_nlp import infer_intent, infer_ner, answer_rule_or_rag

app = FastAPI(title="Jumantik-Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DemoReq(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/web/", response_class=HTMLResponse)
def web():
    html = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.post("/demo")
def demo_api(req: DemoReq):
    intent, conf, topk = infer_intent(req.text)
    ents = infer_ner(req.text)
    ans = answer_rule_or_rag(req.text)
    return {
        "text": req.text,
        "intent": {"label": intent, "confidence": conf, "topk": topk},
        "ner": ents,
        "answer": ans,
    }
