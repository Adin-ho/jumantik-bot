from __future__ import annotations
from pathlib import Path
import json
import re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from src.config import settings

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

# =====================
# Intent (ONNX)
# =====================
class IntentPredictor:
    def __init__(self):
        onnx_dir = settings.intent_onnx_dir
        onnx_int8 = onnx_dir / "model-int8.onnx"
        onnx_fp32 = onnx_dir / "model.onnx"
        onnx_path = onnx_int8 if onnx_int8.exists() else onnx_fp32
        if not onnx_path.exists():
            raise FileNotFoundError(f"Intent ONNX tidak ditemukan: {onnx_path}")

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        # tokenizer: utamakan yang di folder ONNX (hasil export), fallback ke HF
        tok_dir = onnx_dir if (onnx_dir / "tokenizer.json").exists() else settings.intent_hf_dir
        self.tok = AutoTokenizer.from_pretrained(tok_dir)

        # label
        labels_path = settings.intent_hf_dir / "labels.txt"
        if not labels_path.exists():
            raise FileNotFoundError("labels.txt untuk intent tidak ditemukan. Jalankan training intent dulu.")
        self.labels = labels_path.read_text(encoding="utf-8").splitlines()

    def predict(self, text: str) -> dict:
        enc = self.tok(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
        # ONNX BERT minta int64
        inputs = {k: (v.astype(np.int64) if v.dtype != np.int64 else v) for k, v in enc.items()}
        logits = self.sess.run(None, inputs)[0][0]  # [1, num_labels] → [num_labels]
        p = softmax(logits)
        i = int(p.argmax())
        return {"intent": self.labels[i], "confidence": float(p[i])}

# =====================
# NER (ONNX)
# =====================
class NERTagger:
    def __init__(self):
        ner_dir = settings.ner_onnx_dir
        onnx_int8 = ner_dir / "model-int8.onnx"
        onnx_fp32 = ner_dir / "model.onnx"
        onnx_path = onnx_int8 if onnx_int8.exists() else onnx_fp32
        if not onnx_path.exists():
            raise FileNotFoundError("NER ONNX tidak ditemukan. Export dulu NER ke ONNX.")

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        # Tokenizer: coba ONNX dir → fallback HF
        try:
            self.tok = AutoTokenizer.from_pretrained(ner_dir)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(settings.ner_hf_dir)

        # Label sinkron dari config.json (id2label). Fallback ke labels.txt jika perlu.
        self.labels = self._load_labels_from_config(ner_dir) or \
                      self._load_labels_from_txt(settings.ner_hf_dir / "labels.txt")
        if not self.labels:
            raise FileNotFoundError("Label NER tidak ditemukan. Pastikan config.json (id2label) atau labels.txt tersedia.")

    @staticmethod
    def _load_labels_from_config(folder: Path) -> list[str] | None:
        cfg_path = folder / "config.json"
        if not cfg_path.exists():
            return None
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            id2label = cfg.get("id2label") or {}
            if not isinstance(id2label, dict) or not id2label:
                return None
            # urutkan index 0..N-1
            return [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(len(id2label))]
        except Exception:
            return None

    @staticmethod
    def _load_labels_from_txt(path: Path) -> list[str] | None:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").splitlines()

    def tag(self, text: str) -> list[dict]:
        enc = self.tok(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=192,
        )
        # ONNX minta int64
        inputs = {k: (v.astype(np.int64) if v.dtype != np.int64 else v) for k, v in enc.items()}
        logits = self.sess.run(None, inputs)[0]  # [1, seq, num_labels]

        ids = enc["input_ids"][0].tolist()
        toks = self.tok.convert_ids_to_tokens(ids)
        pred = logits.argmax(-1)[0].tolist()

        ents, cur = [], None
        for tok, lid in zip(toks, pred):
            # skip token khusus
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                if cur and cur["text"]:
                    ents.append(cur)
                cur = None
                continue

            lab = self.labels[int(lid)]
            if lab == "O":
                if cur and cur["text"]:
                    ents.append(cur)
                cur = None
                continue

            # robust bila label tanpa '-'
            if "-" in lab:
                pre, typ = lab.split("-", 1)
            else:
                pre, typ = "I", lab

            piece = tok.replace("##", "")

            # mulai entitas baru jika B, atau belum ada cur, atau tipe berganti
            if pre == "B" or cur is None or cur["type"] != typ:
                if cur and cur["text"]:
                    ents.append(cur)
                cur = {"type": typ, "text": piece}
            else:
                # lanjutan wordpiece
                cur["text"] += piece if tok.startswith("##") else f" {piece}"

        if cur and cur["text"]:
            ents.append(cur)

        return ents

# =====================
# Post-processor: struktur data operasional
# =====================
def extract_structured(text: str, ner_entities: list[dict]) -> dict:
    out = {"RT": None, "RW": None, "NAMA": None, "ALAMAT": None}

    # --- RT/RW via regex lebih tepercaya ---
    m_rt = re.search(r"\bRT\s*0?(\d{1,3})\b", text, flags=re.I)
    m_rw = re.search(r"\bRW\s*0?(\d{1,3})\b", text, flags=re.I)
    if m_rt:
        out["RT"] = m_rt.group(1)
    if m_rw:
        out["RW"] = m_rw.group(1)

    # --- Bersihkan entitas NER yang jelas salah ---
    cleaned = []
    for e in ner_entities or []:
        t = e["type"].upper()
        val = (e["text"] or "").strip().lower()
        # buang token tunggal 'rt'/'rw' tanpa angka
        if t in {"RT", "RW"} and val in {"rt", "rw"}:
            continue
        # nama tidak boleh murni angka (misal "05")
        if t == "NAMA" and re.fullmatch(r"\d+[,.]?", val):
            continue
        cleaned.append({"type": t, "text": e["text"].strip()})
    ner_entities = cleaned

    # --- NAMA: dari NER jika ada & bukan angka; else fallback pattern ---
    name_ner = next((e["text"] for e in ner_entities if e["type"] == "NAMA"), None)
    if name_ner and not re.fullmatch(r"\d+[,.]?", name_ner):
        out["NAMA"] = name_ner
    else:
        m_name = re.search(r"\b(?:saya|nama saya)\s+([A-ZÀ-ÖØ-Ý][\w'.-]+)", text, flags=re.I)
        if m_name:
            out["NAMA"] = m_name.group(1)

    # --- ALAMAT: ambil setelah Jl./Jalan, prioritas yang paling panjang ---
    m_addr = re.search(r"\b(?:Jl\.?|Jalan)\s+(.+?)(?=\bRT\b|\bRW\b|$)", text, flags=re.I)
    addr = m_addr.group(1).strip(" ,.") if m_addr else None
    addr_ner = next((e["text"] for e in ner_entities if e["type"] == "ALAMAT"), None)
    if addr_ner and (not addr or len(addr_ner) > len(addr)):
        addr = addr_ner
    out["ALAMAT"] = addr

    return out


# =====================
# Compose simple answer
# =====================
def compose_answer(intent: str, entities: list[dict]) -> str:
    fields = extract_structured(" ".join([e["text"] for e in entities]) if entities else "", entities)

    if intent == "jadwal_kunjungan":
        rw = fields.get("RW")
        return (
            "Jadwal Jumantik diinformasikan oleh kelurahan/puskesmas setempat."
            + (f" Lokasi: RW {rw}." if rw else "")
        )
    if intent == "lapor_jentik":
        addr = ", ".join([f for f in [fields.get("ALAMAT"), fields.get("RT"), fields.get("RW")] if f])
        return (
            "Untuk pelaporan jentik, kirim nama, alamat lengkap (RT/RW), detail lokasi & foto bila ada."
            + (f" Lokasi: {addr}." if addr else "")
        )
    if intent == "prosedur_fogging":
        return ("Fogging hanya membunuh nyamuk dewasa dan bukan solusi utama. "
                "Tetap lakukan PSN 3M Plus. Hubungi kelurahan/puskesmas untuk jadwal.")
    # default
    return "Pertanyaan diterima. Mohon detail tambahan agar kami bisa bantu."
