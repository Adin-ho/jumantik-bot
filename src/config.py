from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    intent_model_name: str = "indobenchmark/indobert-lite-base-p2"
    ner_model_name: str = "indobenchmark/indobert-lite-base-p2"
    embed_model_name: str = "LazarusNLP/all-indo-e5-small-v4"

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    intent_hf_dir: Path = models_dir / "intent_hf"
    ner_hf_dir: Path = models_dir / "ner_hf"
    intent_onnx_dir: Path = models_dir / "intent_onnx"
    ner_onnx_dir: Path = models_dir / "ner_onnx"

    kb_csv: Path = data_dir / "kb.csv"
    kb_index_dir: Path = models_dir / "kb_faiss"

settings = Settings()
