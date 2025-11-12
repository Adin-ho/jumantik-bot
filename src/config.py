from pathlib import Path

# ====== ROOT ======
ROOT = Path(__file__).resolve().parents[1]

# ====== Helper pilih path yang ada ======
def pick_first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    # kalau semua belum ada, kembalikan kandidat pertama (supaya error-nya jelas)
    return candidates[0]

# ====== DIRS ======
DATA_DIR = ROOT / "data"
DATA_KB_DIR = DATA_DIR / "kb"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ====== DATASETS (cari di dua lokasi: /data/kb dan /data) ======
INTENT_TRAIN = pick_first_existing(DATA_KB_DIR / "intent_train.csv", DATA_DIR / "intent_train.csv")
INTENT_VAL   = pick_first_existing(DATA_KB_DIR / "intent_val.csv",   DATA_DIR / "intent_val.csv")
KB_CSV       = pick_first_existing(DATA_KB_DIR / "kb.csv",           DATA_DIR / "kb.csv")

SCHEDULE_JSON = DATA_KB_DIR / "schedule.json"  # letak jadwal

# ====== ANSWER MODE ======
ANSWER_MODE = "hybrid"   # "rule" | "rag" | "hybrid"
INTENT_CONFIDENCE_MIN = 0.38
RAG_MIN_SIM = 0.18

# ====== FALLBACK ======
FALLBACK_ANSWER = (
    "Maaf, aku belum nemu jawaban yang pas. "
    "Coba tuliskan pertanyaan lebih spesifik (mis. sertakan RT/RW/Alamat)."
)

# ====== SAVE FILES ======
INTENT_VECT_PATH = MODELS_DIR / "intent_tfidf.pkl"
INTENT_CLF_PATH  = MODELS_DIR / "intent_clf.pkl"
KB_VECT_PATH     = MODELS_DIR / "kb_tfidf.pkl"
KB_TEXTS_PATH    = MODELS_DIR / "kb_texts.pkl"
