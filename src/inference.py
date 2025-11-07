# src/inference.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ====== Utils ======

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Stabil, anti-NaN
    x = np.nan_to_num(x, copy=False)  # gantikan NaN/inf
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    s = e.sum(axis=axis, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return e / s

def _read_id2label(model_dir: Path) -> Dict[int, str]:
    cfg = model_dir / "config.json"
    if cfg.exists():
        with cfg.open("r", encoding="utf-8") as f:
            data = json.load(f)
        id2label = data.get("id2label") or {}
        # kunci bisa string; ubah ke int
        return {int(k): v for k, v in id2label.items()}
    # fallback
    return {0: "lapor_jentik", 1: "jadwal_kunjungan", 2: "prosedur_fogging"}

# ====== Intent ======

class IntentPredictor:
    def __init__(
        self,
        onnx_path: str | Path = "models/intent_onnx/model.onnx",
        hf_dir: str | Path = "models/intent_hf",
        tokenizer_name: str | Path | None = None,
        max_len: int = 128,
    ):
        self.onnx_path = Path(onnx_path)
        self.hf_dir = Path(hf_dir)
        if tokenizer_name is None:
            tokenizer_name = self.hf_dir  # pakai tokenizer hasil training
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name), local_files_only=True)
        self.max_len = max_len

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.id2label = _read_id2label(self.hf_dir)

    def _tokenize(self, text: str) -> Dict[str, np.ndarray]:
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors=None,
        )
        # Pastikan shape [1, L] dan dtype INT64 (sesuai ONNX)
        out: Dict[str, np.ndarray] = {}
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            if k in enc:
                arr = np.array(enc[k], dtype=np.int64)  # <<< FIX KRUSIAL
                if arr.ndim == 1:
                    arr = np.expand_dims(arr, 0)
                out[k] = arr
        # Filter: hanya kirim yang dibutuhkan model
        out = {k: v for k, v in out.items() if k in self.input_names}
        # Jika model minta key yang belum ada (mis. token_type_ids), sediakan nol
        for needed in self.input_names:
            if needed not in out:
                # buat zeros dengan panjang sama dgn input_ids
                if "input_ids" in out:
                    shape = out["input_ids"].shape
                else:
                    # fallback aman
                    shape = (1, self.max_len)
                out[needed] = np.zeros(shape, dtype=np.int64)  # >>> INT64
        return out

    def predict_topk(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        ort_inputs = self._tokenize(text)
        logits = self.sess.run(None, ort_inputs)[0]  # [1, C]
        probs = _softmax(logits, axis=-1)[0]         # [C]
        probs = np.nan_to_num(probs, copy=False)
        idxs = np.argsort(-probs)[:k]
        results: List[Tuple[str, float]] = []
        for i in idxs:
            label = self.id2label.get(int(i), str(i))
            conf = float(probs[i])
            # jaga2 conf out of range
            if not np.isfinite(conf):
                conf = 0.0
            results.append((label, conf))
        return results

    def predict(self, text: str) -> Dict[str, Any]:
        topk = self.predict_topk(text, k=3)
        best_label, best_conf = topk[0]
        return {
            "intent": best_label,
            "confidence": float(best_conf),
            "topk": [{"intent": l, "confidence": float(c)} for l, c in topk],
        }

# ====== NER Sederhana (regex rule-based) ======

import re

class NERTagger:
    RE_RT = re.compile(r"\bRT\s*0*([0-9]{1,3})\b", flags=re.IGNORECASE)
    RE_RW = re.compile(r"\bRW\s*0*([0-9]{1,3})\b", flags=re.IGNORECASE)
    RE_NAMA = re.compile(r"\b(saya|nama\s*saya)\s+([A-Z][a-zA-Z]+)", flags=re.IGNORECASE)
    RE_ALAMAT = re.compile(r"(Jl\.?|Jalan)\s+[^,]+", flags=re.IGNORECASE)

    def tag(self, text: str) -> List[Dict[str, str]]:
        ents: List[Dict[str, str]] = []
        m = self.RE_RT.search(text)
        if m:
            ents.append({"type": "RT", "text": m.group(1)})
        m = self.RE_RW.search(text)
        if m:
            ents.append({"type": "RW", "text": m.group(1)})
        m = self.RE_NAMA.search(text)
        if m:
            ents.append({"type": "NAMA", "text": m.group(2)})
        m = self.RE_ALAMAT.search(text)
        if m:
            ents.append({"type": "ALAMAT", "text": m.group(0)})
        return ents

# ====== Compose Answer (rule-based) ======

def _get_ent(ents: List[Dict[str, str]], t: str) -> str | None:
    for e in ents:
        if e.get("type") == t and e.get("text"):
            return e["text"]
    return None

def compose_answer(intent: str, ents: List[Dict[str, str]]) -> str:
    intent = (intent or "").lower()
    rt = _get_ent(ents, "RT")
    rw = _get_ent(ents, "RW")
    nama = _get_ent(ents, "NAMA")
    alamat = _get_ent(ents, "ALAMAT")

    if intent == "lapor_jentik":
        who = f"{nama}" if nama else "Bapak/Ibu"
        lokasi = []
        if alamat: lokasi.append(alamat)
        if rt: lokasi.append(f"RT {rt}")
        if rw: lokasi.append(f"RW {rw}")
        lokasi_txt = ", ".join(lokasi) if lokasi else "lokasi Anda"
        return (f"Terima kasih {who}. Laporan jentik Anda kami catat untuk {lokasi_txt}. "
                "Petugas Jumantik akan menindaklanjuti sesuai jadwal kelurahan. "
                "Mohon tunggu informasi selanjutnya ya.")

    if intent == "jadwal_kunjungan":
        target = []
        if rt: target.append(f"RT {rt}")
        if rw: target.append(f"RW {rw}")
        target_txt = " ".join(target) if target else "wilayah Anda"
        return (f"Jadwal kunjungan Jumantik untuk {target_txt} akan diumumkan oleh pihak kelurahan/RT. "
                "Silakan pantau grup RT/RW atau papan pengumuman setempat.")

    if intent == "prosedur_fogging":
        return ("Fogging dilakukan jika ditemukan kasus DBD dan ada jentik di sekitar. "
                "PSN 3M Plus tetap menjadi langkah utama pencegahan. "
                "Silakan koordinasi dengan RT/RW atau Puskesmas untuk permohonan resmi.")

    # default
    return ("Baik. Mohon jelaskan kebutuhan Anda (lapor jentik, tanya jadwal kunjungan, atau prosedur fogging) "
            "agar kami bisa bantu dengan tepat.")
