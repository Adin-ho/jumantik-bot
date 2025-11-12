from __future__ import annotations
from pathlib import Path
import json
from datetime import date, datetime, timedelta
from typing import Dict, Any

from .config import SCHEDULE_JSON

def _today() -> date:
    return datetime.now().date()

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# -------- Schedule helpers (per-RW) --------
def load_schedule() -> Dict[str, Any]:
    return _load_json(SCHEDULE_JSON)

def save_schedule(data: Dict[str, Any]) -> None:
    _save_json(SCHEDULE_JSON, data)

def _next_week(d: date) -> date:
    return d + timedelta(days=7)

def get_schedule_for_rw(rw: str) -> Dict[str, Any]:
    """Return {'rw': '05', 'next_date': 'YYYY-MM-DD'} and auto-roll if passed."""
    db = load_schedule()
    key = str(int(rw))  # normalisasi '05' -> '5'
    rec = db.get(key)

    if not rec:
        # Default: set ke Sabtu terdekat
        base = _today()
        # cari hari Sabtu (weekday=5)
        days_ahead = (5 - base.weekday()) % 7
        nxt = base + timedelta(days=days_ahead)
        rec = {"rw": key, "next_date": nxt.isoformat()}
        db[key] = rec
        save_schedule(db)
        return rec

    # rollover kalau tanggal sudah lewat
    nxt = datetime.fromisoformat(rec["next_date"]).date()
    while nxt < _today():
        nxt = _next_week(nxt)
    if nxt.isoformat() != rec["next_date"]:
        rec["next_date"] = nxt.isoformat()
        db[key] = rec
        save_schedule(db)
    return rec

def advance_after_visit(rw: str) -> Dict[str, Any]:
    """Panggil ini ketika kunjungan RW sudah dilakukan; jadwal geser seminggu."""
    db = load_schedule()
    key = str(int(rw))
    rec = db.get(key) or {"rw": key, "next_date": _today().isoformat()}
    nxt = datetime.fromisoformat(rec["next_date"]).date()
    rec["next_date"] = _next_week(max(nxt, _today())).isoformat()
    db[key] = rec
    save_schedule(db)
    return rec
