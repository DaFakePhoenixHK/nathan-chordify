"""Output contracts (DESIGN §5.13): analysis.json and display.json as pydantic models.

`python -m nathan_chordify.schema` regenerates shared/analysis.schema.json and shared/display.schema.json.
Serialize with `model.model_dump(mode="json", by_alias=True)` so `global` / `from` keep their JSON names.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class _M(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# ---- source / audio ---------------------------------------------------------------------------
class SourceInfo(_M):
    key: str
    type: Literal["youtube", "url", "upload"]
    url: str | None = None
    title: str
    artist: str | None = None
    duration: float


class AudioInfo(_M):
    play: str
    stems: list[str] = Field(default_factory=list)


# ---- key / tempo / meter / grid ---------------------------------------------------------------
class KeyGlobal(_M):
    tonic: str
    mode: Literal["major", "minor"]
    conf: float


class KeySegment(_M):
    start: float
    end: float
    tonic: str
    mode: Literal["major", "minor"]


class KeyChangeOut(_M):
    t: float
    bar: int
    from_: str = Field(alias="from")          # "B:major"
    to: str
    kind: Literal["step_up", "modulation", "relative", "parallel"]


class KeyInfo(_M):
    global_: KeyGlobal = Field(alias="global")
    segments: list[KeySegment]
    changes: list[KeyChangeOut] = Field(default_factory=list)


class SubdivisionOut(_M):
    from_bar: int
    cells_per_beat: Literal[2, 3, 4, 6]


class TimeRegion(_M):
    start: float
    end: float


class TempoInfo(_M):
    bpm: float
    subdivision: list[SubdivisionOut]
    free_time: list[TimeRegion] = Field(default_factory=list)


class MeterOut(_M):
    from_bar: int
    num: int
    den: int


class BeatOut(_M):
    i: int
    t: float
    bar: int                                   # −1 for pickup / free-time beats
    beat: int                                  # 1-based; 0 in free time
    conf: float


class BarOut(_M):
    i: int
    start: float
    end: float
    beats: int
    section: str | None = None
    free: bool = False


class SectionOut(_M):
    label: str
    group: str | None = None
    start: float
    end: float
    source: Literal["user", "auto"]


# ---- chords -----------------------------------------------------------------------------------
class Labels(_M):
    simple: str
    standard: str
    full: str


class Degree(_M):
    root: int | None                          # semitones above the local tonic; None for N
    bass: int | None
    nashville: str
    roman: str


class Alternative(_M):
    harte: str
    conf: float


class Voicing(_M):
    lh: list[int] = Field(default_factory=list)
    rh: list[int] = Field(default_factory=list)


class ChordOut(_M):
    id: str
    start: float
    end: float
    start_beat: float | None                   # None inside free time
    end_beat: float | None
    bar: int
    beat_in_bar: float | None
    dur_beats: float | None
    harte: str
    root: str | None                           # spelled, transpose 0, default accidentals; None for N
    quality: str
    bass: str | None
    label: Labels
    degree: Degree
    layer: Literal["main", "passing"]
    role: Literal["chord", "bass_event"] = "chord"
    min_detail: float | None = None            # None for main chords
    conf: float
    salience: float | None = None
    equivalents: list[str] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    voicing: Voicing = Field(default_factory=Voicing)
    sources: dict[str, float] = Field(default_factory=dict)
    edited: bool = False
    suggestion: bool = False


class BassNoteOut(_M):
    start: float
    end: float
    note: str
    midi: int
    conf: float
    role: Literal["chord_bass", "passing_tone"]
    min_detail: float | None = None


# ---- bookkeeping ------------------------------------------------------------------------------
class EditsInfo(_M):
    rev: int = 0
    n_ops: int = 0


class GtInfo(_M):
    status: Literal["none", "draft", "reviewed"] = "none"
    tapped_beats: bool = False
    chart: str | None = None


class Stats(_M):
    n_main: int
    n_passing: int
    runtime_s: dict[str, float] = Field(default_factory=dict)


class Analysis(_M):
    schema_version: str = SCHEMA_VERSION
    engine_version: str
    params_hash: str
    calibration: str | None = None
    mode: Literal["full", "fast"]
    created_at: str                            # ISO 8601 UTC
    source: SourceInfo
    audio: AudioInfo
    display: str
    key: KeyInfo
    tempo: TempoInfo
    meter: list[MeterOut]
    beats: list[BeatOut]
    bars: list[BarOut]
    sections: list[SectionOut] = Field(default_factory=list)
    chords: list[ChordOut]
    bass_line: list[BassNoteOut] = Field(default_factory=list)
    lyrics: str | None = None
    edits: EditsInfo = Field(default_factory=EditsInfo)
    gt: GtInfo = Field(default_factory=GtInfo)
    stats: Stats


# ---- display.json -----------------------------------------------------------------------------
class DisplayVariant(_M):
    key_names: list[str]                       # one per key segment
    chords: dict[str, list[str]]               # chord id → [simple, standard, full]
    bass_line: list[str]


class Display(_M):
    variants: dict[str, DisplayVariant]        # "<t>:<auto|sharp|flat>", t = 0..11
    voicings: dict[str, dict[str, Voicing]]    # "<t>" → chord id → voicing


def write_json_schemas() -> list[str]:
    from .config import SHARED_DIR
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, model in (("analysis", Analysis), ("display", Display)):
        p = SHARED_DIR / f"{name}.schema.json"
        p.write_text(json.dumps(model.model_json_schema(by_alias=True), indent=2), "utf-8")
        out.append(str(p))
    return out


if __name__ == "__main__":
    for p in write_json_schemas():
        print("wrote", p)
