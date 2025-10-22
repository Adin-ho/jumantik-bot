# Skeleton: baca semua .txt di data/raw jadi data/kb.csv
from pathlib import Path
import csv

RAW = Path("data/raw")
OUT = Path("data/kb.csv")

def main():
    rows = []
    for p in RAW.glob("**/*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        rows.append({"title": p.stem, "url":"", "section":"", "text":text})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title","url","section","text"])
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
