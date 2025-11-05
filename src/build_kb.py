# src/build_kb.py
from pathlib import Path
import pandas as pd
import json

INP = Path("data/kb/raw/data_chatbot_jumantik.csv")  # file kamu
OUT = Path("data/kb.csv")

def main():
    df = pd.read_csv(INP)  # kolom: id,pertanyaan,jawaban,sumber
    df = df.rename(columns={
        "pertanyaan": "question",
        "jawaban": "answer",
        "sumber": "source"
    })
    # Simpan ringkas untuk embed
    df[["id","question","answer","source"]].to_csv(OUT, index=False)
    print("KB saved to", OUT)

if __name__ == "__main__":
    main()
