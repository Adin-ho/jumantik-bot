import os, pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.utils_data import read_kb_csv

def main():
    kb = read_kb_csv(settings.kb_csv)
    texts = [f"{r['title']} {r.get('section','')} {r['text']}" for r in kb]
    model = SentenceTransformer(settings.embed_model_name)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(emb, dtype=np.float32))

    os.makedirs(settings.kb_index_dir, exist_ok=True)
    faiss.write_index(index, str(settings.kb_index_dir / "kb.index"))
    with open(settings.kb_index_dir / "kb_meta.pkl", "wb") as f:
        pickle.dump(kb, f)
    print("KB index built:", len(kb), "docs")

if __name__ == "__main__":
    main()
