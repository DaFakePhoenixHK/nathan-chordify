"""server/routes_sheets: search (fake fetcher, no network), add, apply refine / sheet, undo, errors, settings."""
from __future__ import annotations

import json
import time
import urllib.parse

import pytest

from nathan_chordify import config, schema
from nathan_chordify.sheets import parsers as P
from nathan_chordify.sheets import align as A
from nathan_chordify.theory import harte as H

from test_server import client, env, put_cached_song  # noqa: F401  (env is a fixture)

KEY = "yt-SHEETTEST01"
VID = KEY[3:]
LA = "啦啦啦啦"


def fixture_bar_chords(a: schema.Analysis, transpose: int = 0) -> list[str]:
    """One chord symbol per bar of the fixture (the main chord covering most of the bar)."""
    mains = [c for c in a.chords if c.layer == "main"]
    out = []
    for bar in a.bars:
        best, ov = "N", 0.0
        for c in mains:
            o = min(c.end, bar.end) - max(c.start, bar.start)
            if o > ov:
                best, ov = c.harte, o
        out.append(H.harte_to_symbol(A.transpose_harte(best, transpose)) if best != "N" else "N.C.")
    return out


def chordpro_from_bars(bars: list[str]) -> str:
    """CantonHymn-style lyrics: two bars per line, a section label every 8 bars."""
    lines = []
    for i in range(0, len(bars), 2):
        if i % 16 == 0:
            lines.append(f"Verse {i // 16 + 1}" if i < 32 else "Chorus")
        lines.append(" ".join(f"[{b}]{LA}" for b in bars[i:i + 2]))
    return "\n".join(lines)


def bar_text(bars: list[str]) -> str:
    rows = ["| " + " | ".join(bars[i:i + 4]) + " |" for i in range(0, len(bars), 4)]
    return "[Verse 1]\n" + "\n".join(rows) + "\n"


class FakeFetcher:
    """Canned CantonHymn answers: one search hit (our test song, written in G) + its detail JSON."""

    def __init__(self, lyrics: str, offline: bool = False, extra_songs: list | None = None):
        self.lyrics, self.offline = lyrics, offline
        self.requests = 0
        self.events: list[tuple[str, str]] = []
        self.extra_songs = extra_songs or []

    def get(self, url: str, max_age_s=None):
        self.requests += 1
        if self.offline:
            self.events.append((url, "offline"))
            return None, "offline"
        if "advanced-search-songs.php" in url:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            assert q["filter1_target"] == ["song_name"] and q["filter1_val"] == ["回家"]
            songs = [{"songId": 1, "slug": "test-song", "parentSlug": None, "title": "回家", "person": "測試樂團",
                      "featuredVideoLink": "", "hasChordPattern": True}] + self.extra_songs
            body = json.dumps({"success": True, "data": {"total": len(songs), "page_index": 1, "per_page": 12, "songs": songs}})
            self.events.append((url, "200"))
            return body, "200"
        if "song-detail.php" in url and "slug=test-song" in url:
            body = json.dumps({"success": True, "data": {
                "currentSong": {"title": "回家", "slug": "test-song", "songkey": {"label": "G"},
                                "songyear_of_publication": [{"label": "2007"}], "owner": [{"label": "測試樂團"}],
                                "songlanguage": [{"label": "粵語"}], "lyrics": self.lyrics},
                "currentSongVideosInfo": [{"type": "video_1_youtube_id", "label": "歌詞影片1", "youtubeId": VID}],
                "parentSongVideosInfo": [], "otherRelatedSongs": []}}, ensure_ascii=False)
            self.events.append((url, "200"))
            return body, "200"
        self.events.append((url, "404"))
        return None, "404"

    def get_json(self, url, max_age_s=None):
        text, status = self.get(url, max_age_s)
        return (json.loads(text) if text else None), status

    def cached_date(self, url):
        return "2026-11-03"

    def forget(self, url):
        pass


def wait_done(c, timeout: float = 90.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = c.get(f"/api/tracks/{KEY}/sheets").json()
        if st["status"] != "searching":
            return st
        time.sleep(0.2)
    raise AssertionError("sheet search did not finish")


def test_search_add_apply_undo_round_trip(env):
    a = put_cached_song(KEY)
    bars_b = fixture_bar_chords(a)                         # the fixture's chords, one per bar (B major)
    fake = FakeFetcher(chordpro_from_bars(fixture_bar_chords(a, -4)))   # the site has it written in G
    with client(sheets_fetcher=fake) as c:
        st = c.get(f"/api/tracks/{KEY}/sheets").json()
        assert st["status"] == "idle" and st["results"] == [] and st["applied"] is None
        assert st["query"] == {"title": "", "artist": "", "terms": []}
        r = c.post(f"/api/tracks/{KEY}/sheets/search", json={})
        assert r.status_code == 200 and r.json()["status"] == "searching"
        st = wait_done(c)
        assert st["status"] == "done" and st["error"] is None, st
        assert st["query"]["title"] == "回家" and st["query"]["terms"][0] == "回家"
        assert fake.requests == 2                           # one search, one song page (CantonHymn only: found)
        assert len(st["results"]) == 1
        res = st["results"][0]
        assert res["source"] == "cantonhymn" and res["id"].startswith("cantonhymn:")
        assert res["title"] == "回家" and res["artist"] == "測試樂團" and res["year"] == 2007
        assert res["key_written"] == "G" and res["key_recording"] == "B" and res["transpose"] == 4
        assert res["verdict"] == "good" and res["video_match"] is True and res["also"] == []
        assert res["match"]["score"] > 0.5 and 0 <= res["match"]["title"] <= 1
        assert res["chart_text"].startswith("[Verse 1]") and "啦" not in res["chart_text"]
        assert res["lyrics_text"] and "[Verse 1]" in res["lyrics_text"] and "啦" in res["lyrics_text"]
        assert all(len(x["labels"]) == 3 for x in res["aligned"])
        # the chart is written in B now, and the aligned chords are the fixture's own, spelled like the display
        disp = c.get(f"/api/tracks/{KEY}/display").json()["variants"]["0:auto"]["chords"]
        hits = 0
        for x in res["aligned"]:
            mid = 0.5 * (x["t0"] + x["t1"])
            ch = next((ch for ch in a.chords if ch.layer == "main" and ch.start <= mid < ch.end), None)
            if ch is not None and ch.harte != "N" and H.parse_harte(ch.harte, strict=False) == H.parse_harte(x["harte"], strict=False):
                assert x["labels"] == disp[ch.id]
                hits += 1
        assert hits >= 0.6 * len(res["aligned"])
        # files: chords-only chart + lyrics, under the song's own cache folder only
        sheets_dir = config.CACHE_DIR / KEY / "sheets"
        files = sorted(p.name for p in sheets_dir.iterdir())
        assert "state.json" in files and any(f.endswith(".chart.txt") for f in files) and any(f.endswith(".lyrics.txt") for f in files)
        assert "啦" not in next(p for p in sheets_dir.glob("*.chart.txt")).read_text("utf-8")
        # a second search without force returns the cached state (no new requests)
        n = fake.requests
        assert c.post(f"/api/tracks/{KEY}/sheets/search", json={}).json()["status"] == "done"
        assert fake.requests == n

        # paste a bar-text chart (the first 24 bars, in B): added as "pasted"
        r = c.post(f"/api/tracks/{KEY}/sheets/add", json={"text": bar_text(bars_b[:24])})
        assert r.status_code == 200, r.text
        st = r.json()
        pasted = next(x for x in st["results"] if x["source"] == "pasted")
        assert pasted["id"].startswith("pasted:") and pasted["url"] is None and pasted["transpose"] == 0
        assert pasted["verdict"] in ("good", "maybe") and pasted["n_chords"] >= 4

        # apply "refine" (no decode state in the test cache → the chord-level refine)
        rid = res["id"]
        r = c.post(f"/api/tracks/{KEY}/sheets/{rid}/apply", json={"mode": "refine"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert set(out) == {"analysis", "display", "rev", "applied", "summary"}
        assert out["applied"]["id"] == rid and out["applied"]["mode"] == "refine" and out["applied"]["group"].startswith("sheet-")
        s = out["summary"]
        assert {"n_changed", "n_kept", "flagged"} <= set(s) and s["method"] == "analysis"
        assert all({"t0", "t1", "detected", "sheet"} <= set(f) for f in s["flagged"])
        a2 = schema.Analysis.model_validate(out["analysis"])
        assert a2.edits.rev == out["rev"]
        st = c.get(f"/api/tracks/{KEY}/sheets").json()
        assert st["applied"] and st["applied"]["group"] == out["applied"]["group"]

        # the EditBar's own undo of the group → applied is null again
        c.post(f"/api/tracks/{KEY}/edits", json={"ops": [{"id": "undo1", "kind": "revert", "target": {"group": out["applied"]["group"]}}]})
        assert c.get(f"/api/tracks/{KEY}/sheets").json()["applied"] is None

        # apply "sheet" (as written): the aligned chords become the main layer, one group
        r = c.post(f"/api/tracks/{KEY}/sheets/{rid}/apply", json={"mode": "sheet"})
        assert r.status_code == 200, r.text
        out2 = r.json()
        a3 = schema.Analysis.model_validate(out2["analysis"])
        edited = [ch for ch in a3.chords if ch.edited and ch.layer == "main"]
        assert len(edited) >= 0.8 * len([x for x in res["aligned"] if x["harte"] != "N"])
        assert out2["summary"]["n_changed"] == len(edited) and [sct.label for sct in a3.sections][:1] == ["Verse 1"]
        # re-applying another sheet reverts the previous sheet group first
        r = c.post(f"/api/tracks/{KEY}/sheets/{pasted['id']}/apply", json={"mode": "sheet"})
        assert r.status_code == 200
        a4 = schema.Analysis.model_validate(r.json()["analysis"])
        end24 = a.bars[24].start
        assert all(ch.start < end24 + 0.5 for ch in a4.chords if ch.edited and ch.layer == "main")   # only the pasted bars
        # DELETE → everything restored
        r = c.delete(f"/api/tracks/{KEY}/sheets/applied")
        assert r.status_code == 200 and r.json()["applied"] is None
        a5 = schema.Analysis.model_validate(r.json()["analysis"])
        assert not any(ch.edited for ch in a5.chords) and a5.sections == []
        assert c.get(f"/api/tracks/{KEY}/sheets").json()["applied"] is None


def test_errors_and_offline(env):
    put_cached_song(KEY)
    fake = FakeFetcher("", offline=True)
    with client(sheets_fetcher=fake) as c:
        r = c.get("/api/tracks/yt-NOTANALYSED/sheets")
        assert r.status_code == 409 and r.json()["code"] == "not_analysed"
        r = c.post(f"/api/tracks/{KEY}/sheets/add", json={"text": "la la la"})
        assert r.status_code == 400 and r.json()["code"] == "bad_chart"
        r = c.post(f"/api/tracks/{KEY}/sheets/add", json={})
        assert r.status_code == 400 and r.json()["code"] == "bad_chart"
        r = c.post(f"/api/tracks/{KEY}/sheets/add", json={"url": "https://example.com/some/page"})
        assert r.status_code == 404 and r.json()["code"] == "no_sheet"
        r = c.post(f"/api/tracks/{KEY}/sheets/add", json={"url": "https://cantonhymn.net/song/x"})
        assert r.status_code == 502 and r.json()["code"] == "sheets_unreachable"
        r = c.post(f"/api/tracks/{KEY}/sheets/nope:1234/apply", json={"mode": "refine"})
        assert r.status_code == 404 and r.json()["code"] == "no_sheet"
        r = c.post(f"/api/tracks/{KEY}/sheets/search", json={})
        assert r.json()["status"] == "searching"
        st = wait_done(c)
        assert st["status"] == "error" and st["error"]["code"] == "sheets_unreachable"
        assert "internet" in st["error"]["msg"].lower() or "reached" in st["error"]["msg"].lower()
        assert c.delete(f"/api/tracks/{KEY}/sheets/applied").status_code == 200   # nothing applied: still fine


def test_settings_get_patch_and_persist(env):
    with client() as c:
        assert c.get("/api/settings").json() == {"sheets_auto_lookup": True}
        r = c.patch("/api/settings", json={"sheets_auto_lookup": False})
        assert r.status_code == 200 and r.json() == {"sheets_auto_lookup": False}
        assert c.get("/api/settings").json() == {"sheets_auto_lookup": False}
        assert c.patch("/api/settings", json={"nope": 1}).status_code == 400
        assert c.patch("/api/settings", json={"sheets_auto_lookup": "yes"}).status_code == 400
        saved = json.loads((config.DATA_DIR / "settings.json").read_text("utf-8"))
        assert saved == {"sheets_auto_lookup": False}
    with client() as c:                                       # a restart keeps the saved value
        assert c.get("/api/settings").json() == {"sheets_auto_lookup": False}


def test_auto_lookup_is_off_in_stub_mode_and_reads_the_setting(env):
    from nathan_chordify.server import routes_sheets as RS
    put_cached_song(KEY)
    with client() as c:
        ctx = c.app.state.ctx
        RS.maybe_auto_search(ctx, KEY, "full")               # stub server: never searches
        assert c.get(f"/api/tracks/{KEY}/sheets").json()["status"] == "idle"
        assert RS.user_settings(ctx)["sheets_auto_lookup"] is True
