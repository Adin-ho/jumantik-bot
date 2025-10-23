from pathlib import Path
import time, json
import onnx
import onnxruntime as ort
from src.inference import IntentPredictor, NERTagger

def check_onnx_signature(path: Path):
    m = onnx.load(str(path))
    inps = {i.name: i.type.tensor_type.elem_type for i in m.graph.input}
    # 7 = INT64
    return [n for n, t in inps.items() if t != 7]

def shapes_match(sess: ort.InferenceSession, num_labels: int):
    out = sess.get_outputs()[0]
    return (out.shape[-1] == num_labels)

def main():
    ok = True
    ip = IntentPredictor()
    bad = check_onnx_signature(Path(ip.sess._model_path))
    print("[intent] bad-dtype-inputs:", bad)
    ok &= (len(bad) == 0)
    t0=time.time(); r=ip.predict("Tes jadwal jumantik RW 05"); t1=time.time()
    print("[intent]", r, "latency_ms:", round((t1-t0)*1000,1))
    ok &= "intent" in r

    nt = NERTagger()
    ner_onnx = Path(nt.sess._model_path)
    cfg = json.loads((ner_onnx.parent/"config.json").read_text(encoding="utf-8"))
    num_labels = len(cfg.get("id2label", {})) or len(nt.labels)
    ok &= shapes_match(nt.sess, num_labels)
    t0=time.time(); ents=nt.tag("Saya Adi di Jl. Mawar RT 05 RW 03 lapor jentik"); t1=time.time()
    print("[ner]", ents, "latency_ms:", round((t1-t0)*1000,1))

    print("\nSELFTEST:", "PASS ✅" if ok else "CHECK ❗")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
