from onnxruntime.quantization import quantize_dynamic, QuantType

pairs = [
    ("models/intent_onnx/model.onnx", "models/intent_onnx/model-int8.onnx"),
    ("models/ner_onnx/model.onnx",    "models/ner_onnx/model-int8.onnx"),
]

for src, dst in pairs:
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)  # tanpa optimize_model
    print("Saved:", dst)
