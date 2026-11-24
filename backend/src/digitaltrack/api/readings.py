"""Manually entered body-composition readings (e.g. copied from a Tanita printout).

Stored exactly as typed in one local JSON file; nothing is computed or inferred from the scan, and no standard
ranges are applied (a category is only stored if the person chose one). A later Tanita file import will write the
same record shape with source "tanita_import"."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import date as _date, datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

SEGMENTS = ("left_arm", "right_arm", "trunk", "left_leg", "right_leg")
Category = Literal["under", "standard", "upper", "over"]


class Segment(BaseModel):
    fat_pct: float = Field(ge=1, le=70)
    category: Category | None = None


class Reading(BaseModel):
    date: str
    weight_kg: float = Field(gt=20, lt=400)
    body_fat_pct: float = Field(ge=1, le=70)
    muscle_kg: float = Field(gt=5, lt=200)
    segments: dict[str, Segment] | None = None
    note: str = Field(default="", max_length=200)

    @field_validator("date")
    @classmethod
    def _date(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("date must be YYYY-MM-DD")
        _date.fromisoformat(v)
        return v

    @field_validator("segments")
    @classmethod
    def _segments(cls, v):
        if v is not None and set(v) != set(SEGMENTS):
            raise ValueError(f"segments must be exactly {list(SEGMENTS)} or omitted")
        return v


class Profile(BaseModel):
    sex: Literal["Male", "Female"] | None = None
    height_cm: float | None = Field(default=None, gt=50, lt=260)


class Target(BaseModel):
    weight_kg: float = Field(gt=20, lt=400)
    body_fat_pct: float = Field(ge=1, le=70)
    muscle_kg: float = Field(gt=5, lt=200)


class ReadingStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.path.is_file():
            return {"profile": {}, "target": None, "readings": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, doc: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, self.path)

    def get(self) -> dict:
        with self._lock:
            doc = self._load()
        doc["readings"].sort(key=lambda r: r["date"])
        return doc

    def add(self, reading: Reading) -> dict:
        with self._lock:
            doc = self._load()
            rec = reading.model_dump()
            rec.update(id=f"r_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}", source="manual")
            doc["readings"].append(rec)
            self._save(doc)
        return rec

    def delete(self, rid: str) -> bool:
        with self._lock:
            doc = self._load()
            keep = [r for r in doc["readings"] if r["id"] != rid]
            if len(keep) == len(doc["readings"]):
                return False
            doc["readings"] = keep
            self._save(doc)
        return True

    def set_field(self, key: str, value) -> None:
        with self._lock:
            doc = self._load()
            doc[key] = value
            self._save(doc)


def make_readings_router(store: ReadingStore) -> APIRouter:
    r = APIRouter(prefix="/api/readings")

    @r.get("")
    def get_all():
        return store.get()

    @r.post("", status_code=201)
    def add(reading: Reading):
        return store.add(reading)

    @r.delete("/target")
    def del_target():
        store.set_field("target", None)
        return {"target": None}

    @r.delete("/{rid}")
    def delete(rid: str):
        if not store.delete(rid):
            raise HTTPException(404, "no such reading")
        return {"deleted": rid}

    @r.put("/profile")
    def put_profile(p: Profile):
        store.set_field("profile", p.model_dump())
        return p

    @r.put("/target")
    def put_target(t: Target):
        store.set_field("target", t.model_dump())
        return t

    return r
