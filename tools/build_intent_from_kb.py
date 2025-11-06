import random
import pandas as pd
from collections import Counter
from src.config import settings

# mapping judul kb -> intent (contoh—silakan sesuaikan kebutuhanmu)
RULES = [
    ("lapor", "lapor_jentik"),
    ("jadwal", "jadwal_kunjungan"),
    ("fogging", "prosedur_fogging"),
    ("biaya", "biaya_surat"),
    ("syarat", "syarat_surat"),
]

MIN_PER_CLASS = 8
VAL_RATIO = 0.2
RND = random.Random(42)

def infer_intent_from_title(title: str) -> str | None:
    t = title.lower()
    for kw, intent in RULES:
        if kw in t:
            return intent
    return None

def main():
    df = pd.read_csv(settings.kb_csv)
    rows = []
    for _, r in df.iterrows():
        title = str(r["title"])
        intent = infer_intent_from_title(title)
        if intent:
            rows.append({"text": title, "intent": intent})
    raw = pd.DataFrame(rows)
    print("Raw label counts:\n", raw["intent"].value_counts())

    # balance via simple augmentation (duplikasi ringan)
    by_label = {lbl: raw[raw["intent"] == lbl]["text"].tolist() for lbl in set(raw["intent"])}
    # pastikan semua label penting ada walau kosong
    for must in ["lapor_jentik", "jadwal_kunjungan", "prosedur_fogging", "biaya_surat", "syarat_surat"]:
        by_label.setdefault(must, [])

    balanced = []
    for lbl, texts in by_label.items():
        if len(texts) == 0:
            continue
        pool = texts.copy()
        while len(pool) < MIN_PER_CLASS:
            pool.append(RND.choice(texts))
        balanced += [{"text": t, "intent": lbl} for t in pool[:MIN_PER_CLASS]]

    out = pd.DataFrame(balanced)
    print("\nBalanced label counts:\n", out["intent"].value_counts())

    # split train/val stratified sederhana
    train_rows, val_rows = [], []
    for lbl, grp in out.groupby("intent"):
        n = len(grp)
        k = max(1, int(VAL_RATIO * n))
        idx = list(grp.index)
        RND.shuffle(idx)
        val_idx = set(idx[:k])
        for i, row in grp.iterrows():
            (val_rows if i in val_idx else train_rows).append(row)

    train = pd.DataFrame(train_rows)
    val = pd.DataFrame(val_rows)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(settings.intent_train_csv, index=False, encoding="utf-8")
    val.to_csv(settings.intent_val_csv, index=False, encoding="utf-8")
    print(f"\nSaved: {settings.intent_train_csv}")
    print(f"Saved: {settings.intent_val_csv}")

if __name__ == "__main__":
    main()
