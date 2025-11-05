# src/config.py
from pydantic import BaseModel
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../jumatik-bot

class Settings(BaseModel):
    # Folder utama model
    models_dir: Path = ROOT / "models"

    # Intent
    intent_hf_dir: Path = models_dir / "intent_hf"
    intent_onnx_dir: Path = models_dir / "intent_onnx"

    # NER
    ner_hf_dir: Path = models_dir / "ner_hf"
    ner_onnx_dir: Path = models_dir / "ner_onnx"

    # Knowledge Base (RAG)
    kb_raw_dir: Path = ROOT / "data" / "kb" / "raw"
    kb_clean_csv: Path = ROOT / "data" / "kb.csv"            # hasil build_kb.py
    kb_index_dir: Path = ROOT / "data" / "kb" / "index"       # faiss.index, ids.npy, meta.json
    kb_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ONNXRuntime
    onnx_providers: tuple[str, ...] = ("CPUExecutionProvider",)

    # Training default (boleh diabaikan kalau tidak dipakai)
    intent_model_name: str = "indolem/indobert-base-uncased"
    max_len_intent: int = 128
    max_len_ner: int = 192

settings = Settings()
