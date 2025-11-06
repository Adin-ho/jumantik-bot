from __future__ import annotations
from pathlib import Path
import numpy as np
import json
import onnxruntime as ort
from transformers import AutoTokenizer
from src.config import settings

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def _ensure_int64(enc: dict) -> dict:
    fixed = {}
    for k, v in enc.items():
        arr = v
        if arr.dtype == np.int32:
            arr = arr.astype(np.int64)
        fixed[k] = arr
    return fixed

# ================== INTENT ==================
class IntentPredictor:
    def __init__(self):
        onnx_int8 = settings.intent_onnx_dir / "model-int8.onnx"
        onnx_fp32 = settings.intent_onnx_dir / "model.onnx"
        onnx_path = onnx_int8 if onnx_int8.exists() else onnx_fp32
        if not onnx_path.exists():
            raise FileNotFoundError(f"Intent ONNX tidak ditemukan: {onnx_path}")

        self.sess = ort.InferenceSession(str(onnx_path), providers=settings.onnx_providers)

        tok_dir = settings.intent_onnx_dir if (settings.intent_onnx_dir / "tokenizer.json").exists() else settings.intent_hf_dir
        self.tok = AutoTokenizer.from_pretrained(tok_dir)

        labels_path = settings.intent_hf_dir / "labels.txt"
        if not labels_path.exists():
            raise FileNotFoundError("labels.txt tidak ditemukan. Jalankan training intent dulu.")
        self.labels = labels_path.read_text(encoding="utf-8").splitlines()

    def predict(self, text: str) -> dict:
        enc = self.tok(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
        enc = _ensure_int64(enc)
        logits = self.sess.run(None, enc)[0][0]
        prob = _softmax(logits)
        i = int(prob.argmax())
        return {"intent": self.labels[i], "confidence": float(prob[i])}

# ================== NER (opsional) ==================
class NERTagger:
    def __init__(self):
        onnx_int8 = settings.ner_onnx_dir / "model-int8.onnx"
        onnx_fp32 = settings.ner_onnx_dir / "model.onnx"
        onnx_path = onnx_int8 if onnx_int8.exists() else onnx_fp32
        if not onnx_path.exists():
            raise FileNotFoundError(f"NER ONNX tidak ditemukan: {onnx_path}")

        self.sess = ort.InferenceSession(str(onnx_path), providers=settings.onnx_providers)
        tok_dir = settings.ner_onnx_dir if (settings.ner_onnx_dir / "tokenizer.json").exists() else settings.ner_hf_dir
        self.tok = AutoTokenizer.from_pretrained(tok_dir)

        labels_path = settings.ner_hf_dir / "labels.txt"
        if not labels_path.exists():
            raise FileNotFoundError("NER labels.txt tidak ditemukan.")
        self.labels = labels_path.read_text(encoding="utf-8").splitlines()

    def tag(self, text: str) -> list[dict]:
        enc = self.tok(text, return_tensors="np", truncation=True, padding="max_length", max_length=192)
        enc = _ensure_int64(enc)
        logits = self.sess.run(None, enc)[0]  # [1, seq, L]
        ids = enc["input_ids"][0]
        toks = self.tok.convert_ids_to_tokens(ids)
        preds = logits.argmax(-1)[0]

        entities = []
        cur = None

        for tok, lid in zip(toks, preds):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                if cur:
                    entities.append(cur)
                cur = None
                continue
            lab = self.labels[int(lid)]
            if lab == "O":
                if cur:
                    entities.append(cur)
                cur = None
                continue

            prefix, ent_type = lab.split("-", 1)
            piece = tok.replace("##", "")
            if prefix == "B" or (cur and cur["type"] != ent_type):
                if cur:
                    entities.append(cur)
                cur = {"type": ent_type, "text": piece}
            else:
                # I- type
                if cur is None:
                    cur = {"type": ent_type, "text": piece}
                else:
                    cur["text"] += piece if tok.startswith("##") else f" {piece}"

        if cur:
            entities.append(cur)
        # post-fix RT/RW numeric normalization
        for e in entities:
            if e["type"] in {"RT", "RW", "NAMA"}:
                e["text"] = "".join(ch for ch in e["text"] if ch.isdigit()) or e["text"]
        return entities

def compose_answer(intent: str, entities: list[dict]) -> str:
    if intent == "jadwal_kunjungan":
        rw = next((e["text"] for e in entities if e["type"] == "RW"), None)
        return "Jadwal Jumantik diinformasikan oleh kelurahan/puskesmas setempat." + (f" Lokasi RW {rw}." if rw else "")
    if intent == "lapor_jentik":
        addr = []
        for key in ("ALAMAT", "RT", "RW"):
            v = next((e["text"] for e in entities if e["type"] == key), None)
            if v: addr.append(f"{key} {v}" if key in ("RT","RW") else v)
        loc = (", ".join(addr)) if addr else None
        return "Untuk pelaporan jentik, mohon kirim nama, alamat lengkap (RT/RW), detail lokasi & foto bila ada." + (f" Lokasi: {loc}." if loc else "")
    if intent == "prosedur_fogging":
        return "Fogging hanya membunuh nyamuk dewasa; tetap lakukan PSN 3M Plus. Hubungi kelurahan/puskesmas untuk jadwal."
    if intent == "biaya_surat":
        return "Informasi biaya surat mengikuti ketentuan kelurahan setempat. Silakan hubungi loket pelayanan."
    if intent == "syarat_surat":
        return "Syarat surat umumnya: KTP/KK & formulir pengantar RT/RW. Detail menyesuaikan jenis surat."
    return "Pertanyaan diterima. Mohon detail tambahan agar kami bisa bantu."
