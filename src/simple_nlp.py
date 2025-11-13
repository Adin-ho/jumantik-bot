from __future__ import annotations
import re
import pickle
from typing import List, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV

from .config import (
    INTENT_TRAIN, KB_CSV,
    INTENT_VECT_PATH, INTENT_CLF_PATH, INTENT_LABELS_PATH,
    KB_VECT_PATH, KB_TEXTS_PATH,
    INTENT_CONFIDENCE_MIN, RAG_MIN_SIM,
    FALLBACK_ANSWER, ANSWER_MODE,
)
from .utils_data import get_schedule_for_rw, advance_after_visit

# ---------- Utilities ----------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- NER (regex sederhana) ----------
NER_PATTERNS = {
    "RT": re.compile(r"\brt\s*[:\-]?\s*(\d{1,3})\b", re.I),
    "RW": re.compile(r"\brw\s*[:\-]?\s*(\d{1,3})\b", re.I),
    "NAMA": re.compile(r"\bsaya\s+([A-Za-z ]{2,40})\b", re.I),
    "ALAMAT": re.compile(r"\b(di|alamat)\s+([A-Za-z0-9 .\/\-]{3,80})", re.I),
}
def infer_ner(text: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for label, pat in NER_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        if label in ("RT", "RW"):
            out.append({"type": label, "text": m.group(1)})
        elif label == "ALAMAT":
            out.append({"type": label, "text": m.group(2).strip()})
        else:
            out.append({"type": label, "text": m.group(1).strip()})
    return out

# ---------- INTENT: lazy loader ----------
_VEC: TfidfVectorizer | None = None
_CLF: CalibratedClassifierCV | None = None
_LABELS: List[str] | None = None

def _load_intent() -> None:
    """Lazy-load komponen intent dari disk."""
    global _VEC, _CLF, _LABELS
    if (_VEC is not None) and (_CLF is not None) and (_LABELS is not None) and len(_LABELS) > 0:
        return
    if not (INTENT_VECT_PATH.exists() and INTENT_CLF_PATH.exists()):
        raise RuntimeError(
            "Model intent belum dilatih. Jalankan: python -m src.train_intent"
        )
    _VEC = pickle.loads(INTENT_VECT_PATH.read_bytes())
    _CLF = pickle.loads(INTENT_CLF_PATH.read_bytes())
    # LABELS diprioritaskan dari file; fallback ke classes_
    if INTENT_LABELS_PATH.exists():
        _LABELS = pickle.loads(INTENT_LABELS_PATH.read_bytes())
    else:
        _LABELS = list(getattr(_CLF, "classes_", []))

def infer_intent(text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    _load_intent()
    assert _VEC is not None and _CLF is not None and _LABELS is not None
    X = _VEC.transform([normalize_text(text)])
    # aman untuk calibrated/non-calibrated
    if hasattr(_CLF, "predict_proba"):
        proba = _CLF.predict_proba(X)[0]
    else:
        dec = _CLF.decision_function(X)
        if dec.ndim == 1: dec = dec.reshape(1, -1)
        dec = dec - dec.max(axis=1, keepdims=True)
        exps = np.exp(dec)
        proba = (exps / (exps.sum(axis=1, keepdims=True) + 1e-9))[0]
    pairs = sorted([(str(_LABELS[i]), float(proba[i])) for i in range(len(_LABELS))],
                   key=lambda x: x[1], reverse=True)
    top = pairs[0] if pairs else ("", 0.0)
    return top[0], top[1], pairs[:3]

# ---------- KB (TF-IDF) ----------
_KB_VEC: TfidfVectorizer | None = None
_KB_TEXTS: List[str] | None = None
_KB_ROWS: List[Dict[str, str]] | None = None

def _ensure_kb():
    global _KB_VEC, _KB_TEXTS, _KB_ROWS
    if (_KB_VEC is not None) and (_KB_TEXTS is not None) and (_KB_ROWS is not None):
        return
    # vektor + teks
    if KB_VECT_PATH.exists() and KB_TEXTS_PATH.exists():
        _KB_VEC = pickle.loads(KB_VECT_PATH.read_bytes())
        _KB_TEXTS = pickle.loads(KB_TEXTS_PATH.read_bytes())
    else:
        df = pd.read_csv(KB_CSV, encoding="utf-8-sig")
        if not {"title", "text"}.issubset(df.columns):
            raise RuntimeError("kb.csv wajib punya kolom: title,text")
        texts = (df["title"].astype(str) + " . " + df["text"].astype(str)).tolist()
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        vec.fit([normalize_text(t) for t in texts])
        KB_VECT_PATH.write_bytes(pickle.dumps(vec))
        KB_TEXTS_PATH.write_bytes(pickle.dumps(texts))
        _KB_VEC, _KB_TEXTS = vec, texts
    # metadata baris untuk snippet
    df2 = pd.read_csv(KB_CSV, encoding="utf-8-sig")
    _KB_ROWS = df2[["title", "text"]].to_dict(orient="records")

def search_kb(query: str, topk: int = 3) -> List[Dict[str, Any]]:
    _ensure_kb()
    assert _KB_VEC is not None and _KB_TEXTS is not None and _KB_ROWS is not None
    qv = _KB_VEC.transform([normalize_text(query)])
    mv = _KB_VEC.transform([normalize_text(t) for t in _KB_TEXTS])
    sims = (mv @ qv.T).toarray().ravel()  # cosine (tfidf sudah L2-norm)
    idx = sims.argsort()[::-1][:topk]
    out = []
    for i in idx:
        out.append({
            "score": float(sims[i]),
            "title": _KB_ROWS[i]["title"],
            "snippet": _KB_ROWS[i]["text"][:500],
            "text": _KB_ROWS[i]["text"],
        })
    return out

# ---------- RULE answers ----------
RULE_ANSWERS = {
    "prosedur_fogging": (
        "Fogging **bukan** pencegahan utama DBD. Fokus PSN 3M Plus mingguan. "
        "Fogging dilakukan **selektif** setelah ada kasus dan asesmen Puskesmas/Kelurahan."
    ),
    "jadwal_kunjungan": (
        "Jadwal Jumantik diatur per-RW setiap minggu. "
        "Saya cek jadwal terdekat berdasarkan RW yang kamu sebut."
    ),
    "lapor_jentik": (
        "Terima kasih laporannya. Mohon kirim format: **Nama**, **Alamat**, **RT/RW**, dan **lokasi jentik** "
        "(bak mandi, talang, vas, dsb). Petugas akan menindaklanjuti."
    ),
}

def _rule_answer(intent: str, ents_list: List[Dict[str, str]]) -> Dict[str, Any] | None:
    if intent not in RULE_ANSWERS:
        return None
    if intent == "jadwal_kunjungan":
        rw = next((e["text"] for e in ents_list if e["type"] == "RW"), None)
        if rw:
            rec = get_schedule_for_rw(rw)
            msg = (f"{RULE_ANSWERS[intent]}\n"
                   f"• RW {int(rw):02d} dijadwalkan pada **{rec['next_date']}**.\n"
                   f"Jika kunjungan hari ini sudah selesai, balas: "
                   f"**kunjungan RW {int(rw):02d} selesai** (jadwal akan bergeser otomatis).")
            return {"mode": "rule", "text": msg}
        return {"mode": "rule", "text": RULE_ANSWERS[intent] + " Sertakan RW ya (mis. RW 05)."}
    return {"mode": "rule", "text": RULE_ANSWERS[intent]}

def _maybe_complete_visit(text: str) -> Dict[str, Any] | None:
    m = re.search(r"kunjungan\s*rw\s*(\d{1,3})\s*selesai", text, re.I)
    if not m:
        return None
    rw = m.group(1)
    rec = advance_after_visit(rw)
    return {"mode": "rule",
            "text": f"Terima kasih. Jadwal RW {int(rw):02d} digeser otomatis ke **{rec['next_date']}**."}

# ---------- Orkestrasi ----------
def answer_rule_or_rag(q: str) -> Dict[str, Any]:
    # perintah khusus
    done = _maybe_complete_visit(q)
    if done:
        return done

    ents = infer_ner(q)
    intent, conf, topk = infer_intent(q)

    # 1) RULE jika confidence cukup
    if ANSWER_MODE in ("rule", "hybrid") and conf >= INTENT_CONFIDENCE_MIN:
        rule = _rule_answer(intent, ents)
        if rule:
            return rule

    # 2) RAG fallback
    if ANSWER_MODE in ("rag", "hybrid"):
        try:
            hits = search_kb(q, topk=3)
            if hits and hits[0]["score"] >= RAG_MIN_SIM:
                best = hits[0]
                return {"mode": "rag", "text": best["snippet"], "source": best["title"]}
        except Exception:
            pass

    return {"mode": "fallback", "text": FALLBACK_ANSWER}

# ---------- Helper untuk API ----------
def intent_payload(text: str) -> Dict[str, Any]:
    label, conf, topk = infer_intent(text)
    return {
        "intent": label,
        "confidence": float(conf),
        "topk": [{"intent": l, "confidence": float(p)} for (l, p) in topk],
    }

def entities_payload(text: str) -> List[Dict[str, str]]:
    return infer_ner(text)
