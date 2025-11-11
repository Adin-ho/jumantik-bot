from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ===================== Helper path & normalizer =====================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def normalize_text(s: str) -> str:
    """Normalisasi ringan: lowercase + spasi rapih."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    # seragamkan penulisan alamat umum
    s = s.replace("jalan", "jl").replace("jln", "jl").replace("jl.", "jl")
    s = re.sub(r"\s+", " ", s)
    return s


# ===================== INTENT (scikit-learn) =====================

_INTENT_LABELS: List[str] = []
_INTENT_PIPE: Pipeline | None = None


def _load_intent_df() -> pd.DataFrame:
    train_csv = DATA / "intent_train.csv"
    val_csv   = DATA / "intent_val.csv"
    if not train_csv.exists():
        raise RuntimeError(f"File tidak ditemukan: {train_csv}")

    def _read(p: Path) -> pd.DataFrame:
        return pd.read_csv(p, encoding="utf-8-sig")

    df = _read(train_csv)
    if val_csv.exists():
        try:
            df2 = _read(val_csv)
            df = pd.concat([df, df2], ignore_index=True)
        except Exception:
            pass

    # deteksi kolom (robust terhadap variasi nama kolom)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("text", list(df.columns)[0])
    intent_col = cols.get("intent", list(df.columns)[1] if len(df.columns) > 1 else list(df.columns)[0])

    out = pd.DataFrame({
        "text": df[text_col].map(normalize_text),
        "intent": df[intent_col].map(lambda x: str(x).strip().lower().replace(" ", "_")),
    })
    out = out[(out["text"] != "") & (out["intent"] != "")].drop_duplicates()
    return out


def train_intent_if_needed() -> None:
    """Latih sekali saat dipakai pertama kali. Aman untuk dataset kecil."""
    global _INTENT_PIPE, _INTENT_LABELS
    if _INTENT_PIPE is not None:
        return

    df = _load_intent_df()
    labels = sorted(df["intent"].unique().tolist())
    _INTENT_LABELS = labels

    # Jika hanya ada 1 kelas, buat dummy yang selalu proba=1 untuk kelas tsb
    if len(labels) == 1:
        class DummyClf:
            classes_ = np.array(labels)
            def fit(self, X, y=None): return self
            def predict_proba(self, X): return np.tile(np.array([[1.0]]), (len(X), 1))
        class Stacker:
            def fit(self, X, y=None): return self
            def transform(self, X): return sp.csr_matrix((len(X), 1))
        _INTENT_PIPE = Pipeline([("stack", Stacker()), ("clf", DummyClf())])
        _INTENT_PIPE.fit(df["text"].tolist(), df["intent"].tolist())
        return

    # Gabungan fitur word bi-gram + char 3–5
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)

    class Stacker:
        def fit(self, X, y=None):
            self.w = word.fit(X, y)
            self.c = char.fit(X, y)
            return self
        def transform(self, X):
            return sp.hstack([self.w.transform(X), self.c.transform(X)], format="csr")

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",   # stabil untuk data kecil
        multi_class="auto",
    )

    _INTENT_PIPE = Pipeline([
        ("stack", Stacker()),
        ("clf", clf),
    ])

    _INTENT_PIPE.fit(df["text"].tolist(), df["intent"].tolist())


def infer_intent(text: str) -> Dict[str, Any]:
    """Kembalikan {intent, confidence, topk} dengan probabilitas finite (tidak NaN)."""
    train_intent_if_needed()
    x = [normalize_text(text)]
    proba = _INTENT_PIPE.predict_proba(x)[0]  # type: ignore
    classes = list(getattr(_INTENT_PIPE, "classes_", [])) or _INTENT_LABELS
    pairs = sorted(zip(classes, proba), key=lambda z: float(z[1]), reverse=True)
    topk = [{"intent": str(i), "confidence": float(s)} for i, s in pairs[:5]]
    best = pairs[0]
    return {"intent": str(best[0]), "confidence": float(best[1]), "topk": topk}


# ===================== NER sederhana (regex) =====================

_NER_PATTERNS = {
    "RT": re.compile(r"\brt\s*0*([0-9]{1,3})\b", flags=re.I),
    "RW": re.compile(r"\brw\s*0*([0-9]{1,3})\b", flags=re.I),
}

def _extract_name(text: str) -> str | None:
    t = normalize_text(text)
    # pola "nama saya <...>" atau "saya <...>"
    m = re.search(r"\bnama\s+saya\s+([a-zA-Z\.\-']{2,}(?:\s+[a-zA-Z\.\-']{2,}){0,2})", t)
    if not m:
        m = re.search(r"\bsaya\s+([a-zA-Z\.\-']{2,}(?:\s+[a-zA-Z\.\-']{2,}){0,2})", t)
    if m:
        name = m.group(1).strip().title()
        bad = {"mau", "ingin", "lapor", "bertanya"}
        if name.split()[0].lower() not in bad:
            return name
    return None

def _extract_address(text: str) -> str | None:
    # ambil frasa setelah "di" atau "alamat" sampai sebelum kata kerja umum atau tanda baca
    m = re.search(r"(?:alamat\s*[:\-]?\s*|di\s+)([^,.]+?)(?=\s+mau|\s+ingin|\s+lapor|,|\.|$)", text, flags=re.I)
    if m:
        return m.group(1).strip()
    return None

def infer_ner(text: str) -> List[Dict[str, str]]:
    """
    Return list entitas: [{type,text}, ...]  dengan type ∈ {RT,RW,NAMA,ALAMAT}
    """
    ents: List[Dict[str, str]] = []

    for k, pat in _NER_PATTERNS.items():
        for g in pat.findall(text):
            ents.append({"type": k, "text": str(int(g))})  # hapus leading zero

    nm = _extract_name(text)
    if nm:
        ents.append({"type": "NAMA", "text": nm})

    al = _extract_address(text)
    if al:
        ents.append({"type": "ALAMAT", "text": al})

    return ents


# ===================== KB sederhana (RAG mini) =====================

def _load_kb() -> pd.DataFrame:
    kb_csv = DATA / "kb.csv"
    if not kb_csv.exists():
        # fallback KB minimal bila file belum ada
        rows = [
            {"title": "PSN 3M Plus", "text": "PSN 3M Plus adalah langkah utama pencegahan DBD: Menguras, Menutup, Mendaur ulang + upaya tambahan."},
            {"title": "Fogging", "text": "Fogging dilakukan bila ada kasus DBD dan ditemukan jentik. PSN tetap prioritas."},
            {"title": "Peran Jumantik", "text": "Jumantik memeriksa jentik secara rutin dan edukasi PSN kepada warga."},
        ]
        return pd.DataFrame(rows)
    return pd.read_csv(kb_csv, encoding="utf-8-sig")

_KB_DF: pd.DataFrame | None = None
_KB_VEC: TfidfVectorizer | None = None
_KB_MAT = None

def _ensure_kb():
    global _KB_DF, _KB_VEC, _KB_MAT
    if _KB_DF is not None:
        return
    _KB_DF = _load_kb()
    _KB_DF = _KB_DF.fillna("")
    texts = (_KB_DF["title"].astype(str) + " . " + _KB_DF["text"].astype(str)).tolist()
    _KB_VEC = TfidfVectorizer(ngram_range=(1,2), min_df=1).fit(texts)
    _KB_MAT = _KB_VEC.transform(texts)

def search_kb(query: str, topk: int = 5):
    _ensure_kb()
    qv = _KB_VEC.transform([normalize_text(query)])  # type: ignore
    scores = (qv @ _KB_MAT.T).toarray().ravel()      # type: ignore
    idx = np.argsort(-scores)[:topk]
    out = []
    for i in idx:
        out.append({
            "score": float(scores[int(i)]),
            "title": str(_KB_DF.iloc[int(i)]["title"]),   # type: ignore
            "text":  str(_KB_DF.iloc[int(i)]["text"]),    # type: ignore
        })
    return out

def answer_rule_or_rag(text: str) -> Dict[str, Any]:
    q = normalize_text(text)
    # aturan singkat
    if "fogging" in q:
        ans = ("Fogging dilakukan bila ada **kasus DBD terkonfirmasi** dan **ditemukan jentik** di sekitar lokasi. "
               "PSN 3M Plus tetap menjadi langkah utama pencegahan. Silakan koordinasi dengan RT/RW atau puskesmas.")
        return {"answer": ans, "source": "rule"}
    # default: RAG mini
    hits = search_kb(q, topk=3)
    context = " ".join([h["text"] for h in hits])
    answer = context if context else "Maaf, saya belum menemukan jawaban di basis pengetahuan."
    return {"answer": answer, "source": "kb", "results": hits}
