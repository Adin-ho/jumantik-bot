import os, pickle
import numpy as np
import onnxruntime as ort
from typing import List, Dict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from src.config import settings

class IntentPredictor:
    def __init__(self):
        self.sess = ort.InferenceSession(str(settings.intent_onnx_dir / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(settings.intent_onnx_dir)
        labels_path = settings.intent_hf_dir / "labels.txt"
        self.labels = (labels_path.read_text(encoding="utf-8")).splitlines()

    def predict(self, text: str) -> Dict:
        enc = self.tok(text, return_tensors="np", truncation=True, padding="max_length", max_length=128)
        out = self.sess.run(None, {k: v for k,v in enc.items()})
        logits = out[0]
        probs = softmax(logits[0])
        idx = int(probs.argmax())
        return {"intent": self.labels[idx], "confidence": float(probs[idx])}

class NERTagger:
    def __init__(self):
        self.sess = ort.InferenceSession(str(settings.ner_onnx_dir / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(settings.ner_onnx_dir)
        labels_path = settings.ner_hf_dir / "labels.txt"
        self.labels = (labels_path.read_text(encoding="utf-8")).splitlines()

    def tag(self, text: str) -> List[Dict]:
        enc = self.tok(text, return_tensors="np", truncation=True, padding="max_length", max_length=192)
        out = self.sess.run(None, {k: v for k,v in enc.items()})
        logits = out[0]  # [1, seq, num_labels]
        ids = enc["input_ids"][0]
        tokens = self.tok.convert_ids_to_tokens(ids)
        preds = logits.argmax(-1)[0]
        entities = []
        cur = None
        for tok, lid in zip(tokens, preds):
            if tok in ["[CLS]","[SEP]","[PAD]"]: 
                if cur and cur["text"]: 
                    entities.append(cur); cur=None
                continue
            label = self.labels[int(lid)]
            if label == "O":
                if cur and cur["text"]:
                    entities.append(cur); cur=None
                continue
            prefix, ent = label.split("-", 1)
            word = tok.replace("##", "")
            if prefix == "B":
                if cur and cur["text"]:
                    entities.append(cur)
                cur = {"type": ent, "text": word}
            else:  # I-
                if cur and cur["type"] == ent:
                    cur["text"] += word if tok.startswith("##") else f" {word}"
                else:
                    cur = {"type": ent, "text": word}
        if cur and cur["text"]:
            entities.append(cur)
        return entities

class RAG:
    def __init__(self):
        self.model = SentenceTransformer(settings.embed_model_name)
        self.index = faiss.read_index(str(settings.kb_index_dir / "kb.index"))
        with open(settings.kb_index_dir / "kb_meta.pkl","rb") as f:
            self.meta = pickle.load(f)

    def search(self, query: str, k=5) -> List[Dict]:
        q = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(q.astype(np.float32), k)
        out = []
        for sc, idx in zip(scores[0], ids[0]):
            if idx == -1: continue
            r = dict(self.meta[idx])
            r["score"] = float(sc)
            out.append(r)
        return out

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

# Simple answer composer (template-ish)
def compose_answer(intent: str, entities: List[Dict], passages: List[Dict]) -> str:
    if intent == "jadwal_kunjungan":
        hint = next((p for p in passages if "Jadwal" in p["title"]), None)
        area = next((e["text"] for e in entities if e["type"] in ["RT","RW","ALAMAT"]), None)
        base = hint["text"] if hint else "Jadwal akan diinformasikan oleh pihak kelurahan/puskesmas."
        extra = f" Lokasi yang Anda sebut: {area}." if area else ""
        return f"{base}{extra}"
    if intent == "syarat_surat":
        hint = next((p for p in passages if "Persyaratan" in p["title"] or "Syarat" in p["section"]), None)
        return hint["text"] if hint else "Silakan siapkan KTP, KK, surat pengantar RT/RW, dan dokumen pendukung."
    if intent == "lapor_jentik":
        nama = next((e["text"] for e in entities if e["type"]=="NAMA"), None)
        addr = next((e["text"] for e in entities if e["type"] in ["ALAMAT","RT","RW"]), None)
        return (f"Baik{', ' + nama if nama else ''}. Untuk pelaporan jentik, mohon kirimkan detail lokasi"
                f"{' ('+addr+')' if addr else ''}, jumlah perkiraan jentik, dan foto jika ada. "
                "Tim akan menjadwalkan verifikasi lapangan.")
    # default / fallback
    if passages:
        return passages[0]["text"][:600] + ("..." if len(passages[0]["text"])>600 else "")
    return "Pertanyaan Anda sudah kami terima. Mohon detail tambahan agar kami bisa membantu."

# Quick demo
if __name__ == "__main__":
    intent = IntentPredictor()
    ner = NERTagger()
    rag = RAG()

    q = "Pak, jumantik RW 05 kapan datang? Saya Adi di Jl. Melati RT 05 RW 03."
    pi = intent.predict(q)
    ents = ner.tag(q)
    docs = rag.search(q, k=3)
    ans = compose_answer(pi["intent"], ents, docs)
    print(pi, ents, "\n---\n", ans)
