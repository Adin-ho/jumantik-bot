# src/demo_router.py
from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from .inference import predict_intent, extract_entities, answer_rule_or_rag

router = APIRouter(prefix="", tags=["demo"])

class DemoReq(BaseModel):
    text: str

class DemoResp(BaseModel):
    text: str
    intent: Dict[str, Any]
    entities: List[Dict[str, str]]
    answer: Dict[str, Any]

@router.post("/demo", response_model=DemoResp)
def demo(req: DemoReq):
    intent = predict_intent(req.text)
    ents = extract_entities(req.text)
    ans = answer_rule_or_rag(req.text, intent.get("intent"), float(intent.get("confidence") or 0.0))
    return DemoResp(text=req.text, intent=intent, entities=ents, answer=ans)
