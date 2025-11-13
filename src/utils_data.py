from __future__ import annotations
from pathlib import Path
import json
from datetime import date, timedelta
from typing import Dict, Any
from .config import DATA_DIR

_SCH_FILE = DATA_DIR / "schedule.json"

def _load_sched() -> Dict[str, Any]:
    if _SCH_FILE.exists():
        try:
            return json.loads(_SCH_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    # seed default: semua RW dijadwalkan hari Sabtu minggu ini
    base = date.today()
    # cari Sabtu terdekat (atau hari ini kalau Sabtu)
    dow = base.weekday()  # Mon=0..Sun=6
    delta = (5 - dow) % 7
    next_sat = base + timedelta(days=delta)
    data = {"rw": {f"{i:02d}": {"next_date": str(next_sat)} for i in range(1, 51)}}
    _SCH_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

def _save_sched(obj: Dict[str, Any]) -> None:
    _SCH_FILE.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def get_schedule_for_rw(rw: str) -> Dict[str, Any]:
    rw = f"{int(rw):02d}"
    sched = _load_sched()
    return sched["rw"].get(rw, {"next_date": str(date.today())})

def advance_after_visit(rw: str) -> Dict[str, Any]:
    rw = f"{int(rw):02d}"
    sched = _load_sched()
    cur = sched["rw"].get(rw, {"next_date": str(date.today())})
    d = date.fromisoformat(cur["next_date"])
    # geser 7 hari ke depan
    new_d = d + timedelta(days=7)
    sched["rw"][rw] = {"next_date": str(new_d)}
    _save_sched(sched)
    return sched["rw"][rw]
