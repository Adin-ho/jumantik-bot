from __future__ import annotations

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path

from .simple_nlp import (
    infer_intent,
    infer_ner,
    answer_rule_or_rag,
)

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

app = FastAPI(title="Jumantik-Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextReq(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return {"ok": True, "msg": "ready"}

@app.post("/intent")
def intent_api(req: TextReq):
    return infer_intent(req.text)

@app.post("/ner")
def ner_api(req: TextReq):
    return infer_ner(req.text)

@app.post("/answer")
def answer_api(req: TextReq):
    return answer_rule_or_rag(req.text)

# Endpoint demo yang dipakai oleh web tester
@app.post("/demo")
def demo_api(req: TextReq):
    intent = infer_intent(req.text)
    ents = infer_ner(req.text)
    ans = answer_rule_or_rag(req.text)
    return {
        "intent": intent,
        "entities": ents,
        "answer": ans.get("answer"),
        "source": ans.get("source"),
        "results": ans.get("results", []),
    }

# ---------- Static web (simple tester) ----------
@app.get("/web/", response_class=HTMLResponse)
def web_index():
    index = WEB_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Web tester tidak ditemukan</h1>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))

@app.get("/favicon.ico")
def favicon():
    fav = WEB_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(str(fav))
    return JSONResponse({}, status_code=404)
