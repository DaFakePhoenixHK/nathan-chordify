"""Routes of later milestones (DESIGN §11), answered with 501 until they are built.

M4 (edits.py, gt.py, /api/theory/parse) is built.
TODO(M7): routes_search.py, routes_lyrics.py (+ LRCLIB proxy), routes_setlists.py.
TODO(M8): WS /api/live (server/live.py, WP15).
"""
from __future__ import annotations

from fastapi import APIRouter

from .common import not_implemented

router = APIRouter()


@router.get("/api/search")
def search():
    raise not_implemented("YouTube search", "M7")


@router.api_route("/api/tracks/{key}/lyrics", methods=["GET", "PUT"])
def lyrics(key: str):
    raise not_implemented("Lyrics", "M7")


@router.get("/api/lyrics/lrclib")
def lrclib():
    raise not_implemented("Lyrics search (LRCLIB)", "M7")


@router.api_route("/api/setlists", methods=["GET", "POST"])
@router.api_route("/api/setlists/{setlist_id}", methods=["GET", "PUT", "DELETE"])
def setlists(setlist_id: str | None = None):
    raise not_implemented("Setlists", "M7")
