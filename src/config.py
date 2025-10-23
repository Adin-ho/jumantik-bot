from pathlib import Path
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseModel):
    root_dir: Path = ROOT
    models_dir: Path = ROOT / "models"

    # INTENT
    intent_hf_dir: Path = models_dir / "intent_hf"
    intent_onnx_dir: Path = models_dir / "intent_onnx"

    # NER
    ner_hf_dir: Path = models_dir / "ner_hf"
    ner_onnx_dir: Path = models_dir / "ner_onnx"

    # Data (opsional untuk evaluasi)
    data_dir: Path = ROOT / "data"
    intent_val_csv: Path = data_dir / "intent_val.csv"

settings = Settings()
