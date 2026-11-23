"""JSON-schema access. The schemas in <repo>/schemas are the single source of truth.

Current schemas live in schemas/*.schema.json. Frozen copies of older dataset schema versions live in
schemas/v<version>/ so that sessions written by older adapters validate against exactly the contract they were
written under.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .. import DATASET_SCHEMA_VERSION, LEGACY_DATASET_SCHEMA_VERSIONS

SCHEMA_DIR = Path(
    os.environ.get("DIGITALTRACK_SCHEMA_DIR", Path(__file__).resolve().parents[4] / "schemas")
)

DATASET_SCHEMA_NAMES = ("session", "frame_metadata", "camera_calibration", "turntable_calibration")
OUTPUT_SCHEMA_NAMES = ("measurements", "quality_report", "processing_log")
SCHEMA_NAMES = DATASET_SCHEMA_NAMES + OUTPUT_SCHEMA_NAMES


def _schema_path(name: str, version: str | None) -> Path:
    if name not in SCHEMA_NAMES:
        raise KeyError(f"unknown schema {name!r}; known: {SCHEMA_NAMES}")
    if version is None or version == DATASET_SCHEMA_VERSION or name in OUTPUT_SCHEMA_NAMES:
        return SCHEMA_DIR / f"{name}.schema.json"
    if version in LEGACY_DATASET_SCHEMA_VERSIONS:
        return SCHEMA_DIR / f"v{version}" / f"{name}.schema.json"
    raise KeyError(f"unsupported dataset schema version {version!r}")


@lru_cache(maxsize=None)
def load_schema(name: str, version: str | None = None) -> dict[str, Any]:
    """Load a schema. `version` selects a frozen dataset schema version (None = current)."""
    return json.loads(_schema_path(name, version).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def get_validator(name: str, version: str | None = None) -> Draft202012Validator:
    return Draft202012Validator(load_schema(name, version))


def schema_errors(instance: Any, name: str, version: str | None = None) -> list[str]:
    """Return human-readable schema violations (empty list == valid)."""
    errs = sorted(get_validator(name, version).iter_errors(instance), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errs]
