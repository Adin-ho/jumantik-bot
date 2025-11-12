from __future__ import annotations
import re
import pickle
from typing import List, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

from .config import (
    INTENT_TRAIN, INTENT_VAL, KB_CSV,
    INTENT_VECT_PATH, INTENT_CLF_PATH,
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

def infer_ner(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for label, pat in NER_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        if label in ("RT", "RW"):
            out[label] = m.group(1)
        elif label == "ALAMAT":
            out[label] = m.group(2).strip()
        else:
            out[label] = m.group(1).strip()
    return out

# ---------- INTENT: lazy load / train ----------
_VEC: TfidfVectorizer | None = None
_CLF: CalibratedClassifierCV | None = None
_LABELS: List[str] = []

def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig")

def train_intent_if_needed() -> None:
    global _VEC, _CLF, _LABELS
    if _VEC and _CLF and _LABELS:
        return
    # coba load dari disk
    if INTENT_VECT_PATH.exists() and INTENT_CLF_PATH.exists():
        _VEC = pickle.loads(INTENT_VECT_PATH.read_bytes())
        _CLF = pickle.loads(INTENT_CLF_PATH.read_bytes())
        _LABELS = getattr(_CLF, "classes_", []).tolist()
        return

    df = _read_csv(INTENT_TRAIN)
    # kolom wajib: text,intent
    df = df.dropna(subset=["text", "intent"])
    X = df["text"].astype(str).apply(normalize_text)
    y = df["intent"].astype(str)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    base = LinearSVC()
    clf = CalibratedClassifierCV(base, cv=5)  # supaya ada predict_proba

    pipe = make_pipeline(vec, clf)
    pipe.fit(X, y)

    # simpan terpisah vectorizer & clf
    _VEC = pipe.named_steps["tfidfvectorizer"]
    _CLF = pipe.named_steps["calibratedclassifiercv"]
    _LABELS = _CLF.classes_.tolist()

    INTENT_VECT_PATH.write_bytes(pickle.dumps(_VEC))
    INTENT_CLF_PATH.write_bytes(pickle.dumps(_CLF))

def infer_intent(text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    train_intent_if_needed()
    assert _VEC and _CLF
    X = _VEC.transform([normalize_text(text)])
    proba = _CLF.predict_proba(X)[0]
    labels = _CLF.classes_
    pairs = sorted([(labels[i], float(proba[i])) for i in range(len(labels))],
                   key=lambda x: x[1], reverse=True)
    top = pairs[0]
    return top[0], top[1], pairs[:3]

# ---------- KB search (TF-IDF local) ----------
_KB_VEC: TfidfVectorizer | None = None
_KB_TEXTS: List[str] | None = None
_KB_ROWS: List[Dict[str, str]] | None = None

def _ensure_kb():
    global _KB_VEC, _KB_TEXTS, _KB_ROWS
    if _KB_VEC and _KB_TEXTS and _KB_ROWS:
        return

    if KB_VECT_PATH.exists() and KB_TEXTS_PATH.exists():
        _KB_VEC = pickle.loads(KB_VECT_PATH.read_bytes())
        _KB_TEXTS = pickle.loads(KB_TEXTS_PATH.read_bytes())
    else:
        df = _read_csv(KB_CSV)
        if not {"title", "text"}.issubset(df.columns):
            raise RuntimeError("kb.csv wajib punya kolom: title,text")
        texts = (df["title"].astype(str) + " . " + df["text"].astype(str)).tolist()
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        mat = vec.fit_transform([normalize_text(t) for t in texts])
        KB_VECT_PATH.write_bytes(pickle.dumps(vec))
        KB_TEXTS_PATH.write_bytes(pickle.dumps(texts))
        _KB_VEC, _KB_TEXTS = vec, texts

    # rows untuk metadata title/text
    df = _read_csv(KB_CSV)
    _KB_ROWS = df[["title", "text"]].to_dict(orient="records")

def search_kb(query: str, topk: int = 3) -> List[Dict[str, Any]]:
    _ensure_kb()
    assert _KB_VEC and _KB_TEXTS and _KB_ROWS
    qv = _KB_VEC.transform([normalize_text(query)])  # type: ignore
    mv = _KB_VEC.transform([normalize_text(t) for t in _KB_TEXTS])
    # cosine (tfidf normalized, dot == cosine)
    sims = (mv @ qv.T).toarray().ravel()
    idx = sims.argsort()[::-1][:topk]
    out = []
    for i in idx:
        sc = float(sims[i])
        row = _KB_ROWS[i]
        out.append({
            "score": sc,
            "title": row["title"],
            "snippet": row["text"][:500],
            "text": row["text"],
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

def _rule_answer(intent: str, text: str, ents: Dict[str, str]) -> Dict[str, Any] | None:
    if intent not in RULE_ANSWERS:
        return None
    if intent == "jadwal_kunjungan":
        rw = ents.get("RW")
        if rw:
            rec = get_schedule_for_rw(rw)
            msg = (f"{RULE_ANSWERS[intent]} \n"
                   f"• RW {int(rw):02d} dijadwalkan pada **{rec['next_date']}**. "
                   f"Jika kunjungan hari ini sudah selesai, balas dengan: "
                   f"**kunjungan RW {int(rw):02d} selesai** untuk menggeser jadwal otomatis.")
            return {"mode": "rule", "text": msg}
        # tanpa RW, berikan instruksi
        return {"mode": "rule", "text": RULE_ANSWERS[intent] + " Sertakan RW ya (mis. RW 05)."}
    return {"mode": "rule", "text": RULE_ANSWERS[intent]}

def _maybe_complete_visit(text: str) -> Dict[str, Any] | None:
    m = re.search(r"kunjungan\s*rw\s*(\d{1,3})\s*selesai", text, re.I)
    if not m:
        return None
    rw = m.group(1)
    rec = advance_after_visit(rw)
    return {
        "mode": "rule",
        "text": f"Terima kasih. Jadwal RW {int(rw):02d} digeser otomatis ke **{rec['next_date']}**."
    }
def _predict_proba_safe(clf, X):
    """Kembalikan probabilitas (n_samples, n_classes) sebisanya."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    if hasattr(clf, "decision_function"):
        dec = clf.decision_function(X)
        if dec.ndim == 1:
            dec = dec.reshape(-1, 1)
        # softmax
        dec = dec - dec.max(axis=1, keepdims=True)
        exps = np.exp(dec)
        return exps / (exps.sum(axis=1, keepdims=True) + 1e-9)
    # fallback: rata
    n = getattr(clf, "classes_", [])
    k = len(n) if len(n) > 0 else 1
    return np.ones((len(X), k)) / k

def answer_rule_or_rag(q: str) -> Dict[str, Any]:
    # perintah khusus 'kunjungan RW XX selesai'
    done = _maybe_complete_visit(q)
    if done:
        return done

    ents = infer_ner(q)
    intent, conf, _ = infer_intent(q)

    if ANSWER_MODE in ("rule", "hybrid") and conf >= INTENT_CONFIDENCE_MIN:
        ans = _rule_answer(intent, q, ents)
        if ans:
            return ans

    if ANSWER_MODE in ("rag", "hybrid"):
        try:
            hits = search_kb(q, topk=3)
            if hits and hits[0]["score"] >= RAG_MIN_SIM:
                best = hits[0]
                return {"mode": "rag", "text": best["snippet"], "source": best["title"]}
        except Exception:
            pass

    return {"mode": "fallback", "text": FALLBACK_ANSWER}
