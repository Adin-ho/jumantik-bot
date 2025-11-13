from __future__ import annotations
from pathlib import Path
import os

# --- path dasar
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("JUMANTIK_DATA_DIR", ROOT_DIR / "data" / "kb")).resolve()
MODELS_DIR = (ROOT_DIR / "models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- file data
INTENT_TRAIN = DATA_DIR / "intent_train.csv"
INTENT_VAL   = DATA_DIR / "intent_val.csv"
KB_CSV       = DATA_DIR / "kb.csv"

# --- file model intent
INTENT_VECT_PATH   = MODELS_DIR / "intent_vectorizer.pkl"
INTENT_CLF_PATH    = MODELS_DIR / "intent_clf.pkl"
INTENT_LABELS_PATH = MODELS_DIR / "intent_labels.pkl"

# --- file vektor KB
KB_VECT_PATH  = MODELS_DIR / "kb_tfidf.pkl"
KB_TEXTS_PATH = MODELS_DIR / "kb_texts.pkl"

# --- parameter
INTENT_CONFIDENCE_MIN = float(os.getenv("INTENT_CONF_MIN", "0.55"))  # 55%
RAG_MIN_SIM           = float(os.getenv("RAG_MIN_SIM", "0.15"))
ANSWER_MODE           = os.getenv("ANSWER_MODE", "hybrid")  # rule | rag | hybrid
FALLBACK_ANSWER       = "Maaf, saya belum paham. Bisakah diperjelas?"

# --- web
WEB_INDEX_PATH = (ROOT_DIR / "web" / "index.html").resolve()
