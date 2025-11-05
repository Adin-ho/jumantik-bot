# tools/build_intent_from_kb.py
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb" / "raw" / "data_chatbot_jumantik.csv"
OUT_TRAIN = ROOT / "data" / "intent_train.csv"
OUT_VAL = ROOT / "data" / "intent_val.csv"

# Target label final yang kita pakai
TARGET_LABELS = [
    "jadwal_kunjungan",
    "lapor_jentik",
    "prosedur_fogging",
    "biaya_surat",
    "syarat_surat",
]

# Aturan ringan untuk memetakan pertanyaan KB -> intent
RULES = [
    ("jadwal_kunjungan", r"\b(jadwal|kapan.*jumantik|kunjungan)\b"),
    ("lapor_jentik", r"\b(lapor|melapor|jentik|ada.*jentik)\b"),
    ("prosedur_fogging", r"\b(fogging|pengasapan|asap)\b"),
    ("biaya_surat", r"\b(biaya|bayar|tarif|harga).*(surat|pengantar)\b"),
    ("syarat_surat", r"\b(syarat|dokumen|berkas).*(surat|pengantar)\b"),
]

def label_by_rules(text: str) -> str | None:
    t = text.lower()
    for lbl, pat in RULES:
        if re.search(pat, t):
            return lbl
    return None

def main():
    df = pd.read_csv(KB)  # kolom: id, pertanyaan, jawaban, sumber
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    rows = []
    for _, r in df.iterrows():
        q = str(r.get("pertanyaan", "")).strip()
        if not q:
            continue
        lab = label_by_rules(q)
        if lab:
            rows.append({"text": q, "intent": lab})

    data = pd.DataFrame(rows)
    # drop duplikat pertanyaan agar clean
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Jaga agar hanya label target
    data = data[data["intent"].isin(TARGET_LABELS)].reset_index(drop=True)

    # (opsional) minimal 20 contoh/label → bisa mulai dari 8–10 dulu
    counts = data["intent"].value_counts()
    print("Label counts:\n", counts, "\n")

    if data.empty or counts.min() < 5:
        print("⚠️  Data terlalu sedikit / tidak seimbang. Tambah contoh atau longgarkan RULES.")
    
    X_train, X_val = train_test_split(
        data, test_size=0.2, stratify=data["intent"], random_state=42
        if data['intent'].nunique() > 1 and counts.min() >= 2 else None
    )

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUT_TRAIN, index=False)
    X_val.to_csv(OUT_VAL, index=False)
    print("Saved:", OUT_TRAIN)
    print("Saved:", OUT_VAL)

if __name__ == "__main__":
    main()
