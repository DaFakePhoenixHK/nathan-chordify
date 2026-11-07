"""Ingest errors carry a stable code plus a user-facing message and hint (DESIGN §5.1 step 4)."""
from __future__ import annotations


class IngestError(RuntimeError):
    """codes: bad_source, not_a_file, drm, too_long, too_large, rate_limited, backoff, forbidden,
    unavailable, network, decode_failed, download_failed."""

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict:
        return {"code": self.code, "msg": self.message, "hint": self.hint}

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}" + (f" Hint: {self.hint}" if self.hint else "")
