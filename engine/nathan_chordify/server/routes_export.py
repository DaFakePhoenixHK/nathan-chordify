"""Exports (DESIGN §11, WP10): `lab`, `json` and the printable chord `sheet`; MIDI, MusicXML, iReal Pro in M7.
Every export uses the edited analysis (§5.13).

    GET /api/tracks/{key}/export/{fmt}?detail=0.8&transpose=&play_in=&level=&notation=
    GET /api/tracks/{key}/export/sheet?format=txt|chordpro|json|pdf&transpose=0&level=simple|standard|full
                                       &slash=1&lyrics=1&acc=auto&sheet=<result id>
        txt      the bar chart (+ chords over the words when a sheet with lyrics exists), "<title> (<key>) chords.txt"
        chordpro the same as a ChordPro file ([C]word), "<title> (<key>) chords.cho"
        json     every part separately (the web's print view)
        pdf      501 until a PDF library is in the venv: use Print → "Save as PDF" (export/sheet.py)
"""
from __future__ import annotations

import json
from urllib.parse import quote

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from ..export import lab_lines
from ..export import sheet as SHEET
from . import edits
from .common import ApiError, check_key, ctx, not_implemented

router = APIRouter()

LATER = {"midi-daw", "midi-fixed", "musicxml", "chordpro", "ireal", "text"}


def attachment(filename: str, fallback: str) -> dict:
    """Content-Disposition that keeps a Chinese title (RFC 5987) with an ASCII fallback."""
    return {"Content-Disposition": f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"}


def _filename_stem(title: str, key: str) -> str:
    stem = "".join("_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in title).strip(" .")
    return stem[:100] or key


SHEET_FORMATS = ("txt", "chordpro", "json", "pdf")


@router.get("/api/tracks/{key}/export/{fmt}")
def export(key: str, fmt: str, detail: float = Query(0.8, ge=0.0, le=1.0), transpose: int = 0,
           play_in: str | None = None, level: str | None = None, notation: str | None = None,
           out_format: str = Query("txt", alias="format"), slash: bool = True, lyrics: bool = True,
           acc: str = "auto", sheet: str | None = None, c=Depends(ctx)) -> Response:
    check_key(key)
    if fmt in LATER:
        raise not_implemented(f"The {fmt} export", "M7")
    if fmt not in ("lab", "json", "sheet"):
        raise ApiError(404, "bad_format", f"Unknown export format: {fmt}")
    p = c.cache_dir / key / "analysis.json"
    if not p.exists():
        raise ApiError(404, "not_found", "No analysis for this song.")
    a = edits.edited(p.parent).model.model_copy(deep=True)        # every export uses the edited analysis
    row = c.library.get(key)
    if row and row.get("renamed"):
        a.source.title, a.source.artist = row["title"], row["artist"]
    stem = _filename_stem(c.track_title(key) or a.source.title, key)
    if fmt == "sheet":
        return export_sheet(a, p.parent, key, stem, out_format, transpose, level or "full", slash, lyrics, acc, sheet)
    if fmt == "lab":
        if transpose % 12 or play_in:
            raise not_implemented("Transposed .lab export", "M7")   # .lab is Harte at concert pitch
        text = "\n".join(lab_lines(a, detail)) + "\n"
        return Response(text, media_type="text/plain; charset=utf-8", headers=attachment(f"{stem}.lab", f"{key}.lab"))
    return Response(a.model_dump_json(by_alias=True, indent=1), media_type="application/json",
                    headers=attachment(f"{stem}.json", f"{key}.json"))


def export_sheet(a, song_dir, key: str, stem: str, fmt: str, transpose: int, level: str, slash: bool,
                 lyrics: bool, acc: str, sheet_id: str | None) -> Response:
    """The printable chord sheet (export/sheet.py): bar chart + chords over the words."""
    if fmt not in SHEET_FORMATS:
        raise ApiError(400, "bad_format", f"format must be one of {', '.join(SHEET_FORMATS)}.")
    if level not in SHEET.LEVELS:
        raise ApiError(400, "bad_level", "level must be simple, standard or full.")
    if acc not in ("auto", "sharp", "flat"):
        raise ApiError(400, "bad_acc", "acc must be auto, sharp or flat.")
    if fmt == "pdf":
        raise ApiError(501, "no_pdf", "PDF files need a PDF library that isn't installed.",
                       "Use Print and choose \"Save as PDF\" in the print window instead.")
    ex = SHEET.build(a, song_dir, transpose=transpose, level=level, slash=slash, lyrics=lyrics, pref=acc, sheet_id=sheet_id)
    name = f"{stem} ({ex.key}) chords"
    if fmt == "json":
        return Response(json.dumps(ex.to_json(), ensure_ascii=False), media_type="application/json",
                        headers={"Cache-Control": "no-cache"})
    if fmt == "chordpro":
        if not ex.lyrics_chordpro:
            raise ApiError(404, "no_lyrics", "No lyrics were found for this song, so there is nothing to put the chords over.",
                           "Find or paste the song's sheet with its words first, or download the bar chart instead.")
        return Response(ex.to_chordpro(), media_type="text/plain; charset=utf-8",
                        headers=attachment(f"{name}.cho", f"{key}-chords.cho"))
    return Response(ex.to_text(), media_type="text/plain; charset=utf-8",
                    headers=attachment(f"{name}.txt", f"{key}-chords.txt"))
