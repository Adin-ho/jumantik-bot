import os
from src.config import settings
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForTokenClassification
from transformers import AutoTokenizer
from optimum.pipelines import ORTOptimizeModel
from optimum.optimization import OptimizationConfig

def export_intent():
    model = ORTModelForSequenceClassification.from_pretrained(
        str(settings.intent_hf_dir), from_transformers=True
    )
    tok = AutoTokenizer.from_pretrained(str(settings.intent_hf_dir))
    os.makedirs(settings.intent_onnx_dir, exist_ok=True)
    model.save_pretrained(settings.intent_onnx_dir)
    tok.save_pretrained(settings.intent_onnx_dir)

    optimizer = ORTOptimizeModel.from_pretrained(
        settings.intent_onnx_dir, feature="sequence-classification"
    )
    optimizer.optimize(OptimizationConfig(optimization_level=2))
    optimizer.save_pretrained(settings.intent_onnx_dir)

def export_ner():
    model = ORTModelForTokenClassification.from_pretrained(
        str(settings.ner_hf_dir), from_transformers=True
    )
    tok = AutoTokenizer.from_pretrained(str(settings.ner_hf_dir))
    os.makedirs(settings.ner_onnx_dir, exist_ok=True)
    model.save_pretrained(settings.ner_onnx_dir)
    tok.save_pretrained(settings.ner_onnx_dir)

    optimizer = ORTOptimizeModel.from_pretrained(
        settings.ner_onnx_dir, feature="token-classification"
    )
    optimizer.optimize(OptimizationConfig(optimization_level=2))
    optimizer.save_pretrained(settings.ner_onnx_dir)

if __name__ == "__main__":
    export_intent()
    export_ner()
    print("Export & optimization done. You can run INT8 quantization via: ")
    print("  optimum-cli onnxruntime quantize --model models/intent_onnx --per-channel True --weight-only")
    print("  optimum-cli onnxruntime quantize --model models/ner_onnx --per-channel True --weight-only")
