# src/export_onnx_ner.py
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.exporters.onnx import export
from optimum.exporters.tasks import TasksManager

MODEL_DIR = Path("models/ner_hf")
OUT_DIR   = Path("models/ner_onnx")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
    task = "token-classification"

    _, _ = TasksManager.determine_framework(MODEL_DIR)
    onnx_cfg = TasksManager.get_exporter_config_constructor(
        exporter="onnx", model=model, task=task
    )(model.config)

    export(
        preprocessor=tok,
        model=model,
        config=onnx_cfg,
        opset=14,
        output=OUT_DIR / "model.onnx",
    )
    tok.save_pretrained(OUT_DIR)
    print("NER ONNX saved to:", OUT_DIR / "model.onnx")

if __name__ == "__main__":
    main()
