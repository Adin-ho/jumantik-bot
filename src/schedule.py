# src/schedule.py
from datetime import datetime, timedelta
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter(prefix="/schedule", tags=["schedule"])

# stub in-memory (ganti ke DB)
SCHEDULE = {}  # key: "RW-05", val: {"next_visit": "2025-11-12", "history":[...]}

class ReportDone(BaseModel):
    rw: str  # "05"
    rt: str | None = None
    note: str | None = None

@router.post("/done")
def report_done(p: ReportDone):
    key = f"RW-{p.rw}"
    today = datetime.now().date()
    next_visit = today + timedelta(days=14)  # aturanmu sendiri
    rec = SCHEDULE.get(key, {"history":[]})
    rec["history"].append({"date": str(today), "note": p.note})
    rec["next_visit"] = str(next_visit)
    SCHEDULE[key] = rec
    return {"ok": True, "next_visit": rec["next_visit"], "history": rec["history"]}

@router.get("/next")
def get_next(rw: str):
    key = f"RW-{rw}"
    return SCHEDULE.get(key, {"next_visit": None, "history": []})
