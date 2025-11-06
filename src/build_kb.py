# src/build_kb.py
import pandas as pd
from src.config import settings

CANDIDATES = {
    "id": ["id", "ID", "no", "nomor"],
    "title": ["title", "judul", "pertanyaan", "question", "tanya"],
    "text": ["text", "jawaban", "answer", "isi", "konten", "content", "penjelasan"],
    "source": ["source", "sumber", "referensi"],
}

def pick_col(df, keys, required=True, default=""):
    for k in keys:
        if k in df.columns: 
            return k
        # kasus CSV punya spasi/kapital aneh
        low = [c for c in df.columns if c.strip().lower() == k.lower()]
        if low:
            return low[0]
    if required:
        raise ValueError(f"Tidak menemukan kolom kandidat: {keys} di file KB mentah.")
    return None

def main():
    inp = settings.kb_raw_csv
    if not inp.exists():
        raise FileNotFoundError(f"Tidak ada file: {inp}")

    df = pd.read_csv(inp)

    col_id = pick_col(df, CANDIDATES["id"], required=False)
    col_title = pick_col(df, CANDIDATES["title"], required=True)
    col_text = pick_col(df, CANDIDATES["text"], required=True)
    col_source = pick_col(df, CANDIDATES["source"], required=False)

    out = pd.DataFrame({
        "id": df[col_id] if col_id else range(1, len(df) + 1),
        "title": df[col_title].fillna("").astype(str),
        "section": "",
        "text": df[col_text].fillna("").astype(str),
        "source": df[col_source].fillna("").astype(str) if col_source else "",
    })

    # bersihkan \n, kutip ganda yang suka bikin FAISS/JSON rewel
    for c in ["title", "text", "source"]:
        out[c] = out[c].str.replace("\r", " ", regex=False).str.replace("\n", " ", regex=False).str.strip()

    out.to_csv(settings.kb_csv, index=False, encoding="utf-8")
    print(f"KB saved to {settings.kb_csv}")

if __name__ == "__main__":
    main()
