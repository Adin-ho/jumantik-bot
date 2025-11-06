from pathlib import Path
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseModel):
    # ===== Paths umum
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"

    # ===== Intent dataset (CSV)
    intent_train_csv: Path = data_dir / "intent_train.csv"
    intent_val_csv: Path = data_dir / "intent_val.csv"

    # ===== Folder model Hugging Face + ONNX untuk Intent
    intent_hf_dir: Path = models_dir / "intent_hf"
    intent_onnx_dir: Path = models_dir / "intent_onnx"

    # ===== Folder model Hugging Face + ONNX untuk NER (opsional, jika pakai)
    ner_hf_dir: Path = models_dir / "ner_hf"
    ner_onnx_dir: Path = models_dir / "ner_onnx"

    # ===== Knowledge Base & RAG
    kb_raw_csv: Path = data_dir / "kb" / "raw" / "data_chatbot_jumantik.csv"  # sumber mentah (opsional)
    kb_csv: Path = data_dir / "kb.csv"                                        # hasil build_kb.py
    kb_index_dir: Path = data_dir / "kb" / "index"
    kb_faiss: Path = kb_index_dir / "faiss.index"
    kb_ids: Path = kb_index_dir / "ids.npy"
    kb_meta: Path = kb_index_dir / "meta.json"
    embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ===== Model backbone untuk fine-tuning intent
    intent_model_name: str = "indolem/indobert-base-uncased"

    # ===== ONNX Runtime
    onnx_providers: list[str] = ["CPUExecutionProvider"]

settings = Settings()
