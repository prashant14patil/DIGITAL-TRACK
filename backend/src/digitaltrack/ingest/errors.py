"""Explicit, machine-readable ingestion errors. Ingestion never guesses: it stops with a code and a reason."""

from __future__ import annotations

from typing import Any


class IngestError(Exception):
    """A capture could not be ingested.

    `code` is stable and machine-readable (e.g. ``source_not_found``, ``invalid_metadata``,
    ``checksum_mismatch``, ``tool_missing``, ``no_video_stream``, ``frame_count_mismatch``).
    `problems` lists every individual issue when several were found at once.
    """

    def __init__(self, code: str, message: str, *, path: str | None = None,
                 problems: list[str] | None = None) -> None:
        self.code = code
        self.message = message
        self.path = path
        self.problems = list(problems or [])
        super().__init__(str(self))

    def __str__(self) -> str:
        s = f"{self.code}: {self.message}"
        if self.path:
            s += f" [{self.path}]"
        if self.problems:
            s += " (" + "; ".join(self.problems) + ")"
        return s

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "path": self.path, "problems": self.problems}
