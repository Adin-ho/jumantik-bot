import csv
from typing import List, Dict

def read_kb_csv(path) -> List[Dict]:
    rows = []
    with open(path, newline='', encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows
