from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import WEB_INDEX_PATH
from .simple_nlp import intent_payload, entities_payload, answer_rule_or_rag

app = FastAPI(title="Jumantik Bot")

class DemoReq(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/web/")
def web_ui():
    return FileResponse(WEB_INDEX_PATH)

@app.post("/demo")
def demo_api(req: DemoReq):
    q = (req.text or "").strip()
    intent = intent_payload(q)
    ents = entities_payload(q)
    ans = answer_rule_or_rag(q)
    # struktur disesuaikan dengan index.html
    return JSONResponse({
        "text": q,
        "intent": intent,
        "entities": ents,
        "answer": {
            "mode": ans.get("mode", "-"),
            "text": ans.get("text") or ans.get("answer") or "-",
            "refs": [ans.get("source")] if ans.get("source") else [],
        },
    })
