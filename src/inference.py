# src/inference.py
from __future__ import annotations
from typing import Dict, Any, List
import os
import requests

HARD_THRESHOLD = 0.70
SOFT_THRESHOLD = 0.50
API_BASE = os.environ.get("JUMANTIK_API_BASE", "http://127.0.0.1:8000")

def _http_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def predict_intent(text: str) -> Dict[str, Any]:
    out = _http_post("/intent", {"text": text})
    return {
        "intent": out.get("intent"),
        "confidence": float(out.get("confidence", 0.0)),
        "topk": out.get("topk", []),
    }

def extract_entities(text: str) -> List[Dict[str, str]]:
    res = _http_post("/ner", {"text": text})
    ents = res.get("entities", res if isinstance(res, list) else [])
    if not isinstance(ents, list):
        return []
    STOP_TOKENS = {"mau","ingin","lapor","tanya","mohon","minta"}
    out = []
    for e in ents:
        etype = str(e.get("type", "")).upper()
        etext = str(e.get("text", "")).strip()
        if etype == "ALAMAT":
            toks, keep = etext.split(), []
            for w in toks:
                if w.lower() in STOP_TOKENS:
                    break
                keep.append(w)
            etext = " ".join(keep).rstrip(",.;:")
        if etype and etext:
            out.append({"type": etype, "text": etext})
    return out

def _answer_with_rag(text: str) -> Dict[str, Any]:
    rag = _http_post("/answer_rag", {"text": text})
    return {
        "mode": rag.get("mode", "rag"),
        "answer": rag.get("answer") or rag.get("text") or "",
        "summary": rag.get("summary", ""),
        "refs": rag.get("refs", []),
    }

def answer_rule_or_rag(text: str, intent: str | None, confidence: float) -> Dict[str, Any]:
    intent = intent or "other"
    confidence = float(confidence or 0.0)

    rule_answer = None
    try:
        rule = _http_post("/answer", {"text": text})
        rule_answer = rule.get("answer") or rule.get("text")
    except Exception:
        rule_answer = None

    if confidence >= HARD_THRESHOLD and rule_answer:
        return {"mode": "rule", "answer": rule_answer, "refs": []}

    if confidence >= SOFT_THRESHOLD and rule_answer:
        rag = _answer_with_rag(text)
        merged = (rule_answer or "").strip()
        if rag.get("summary"):
            merged = (merged + "\n\nRingkasan: " + rag["summary"]).strip()
        return {"mode": "hybrid", "answer": merged or rag.get("answer",""), "refs": rag.get("refs", [])}

    return _answer_with_rag(text)
