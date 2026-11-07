"""export/sheet + GET /api/tracks/{key}/export/sheet: the bar chart and the chords over the lyrics."""
from __future__ import annotations

from pathlib import Path

from nathan_chordify import config, schema
from nathan_chordify.export import sheet as SH
from nathan_chordify.server import stub
from nathan_chordify.sheets import lookup as L
from nathan_chordify.theory import chart as CH
from nathan_chordify.theory import harte as H

from test_server import client, env, put_cached_song  # noqa: F401  (env is a fixture)

KEY = "yt-SHEETEXPORT1"


def fixture() -> schema.Analysis:
    fa, _ = stub.fixture_paths()
    return schema.Analysis.model_validate_json(fa.read_text("utf-8"))


# ---- unit: placing our chords over the words --------------------------------------------------------

def test_place_chords_on_words_and_syllables():
    text = "[Verse 1]\n[C]Twinkle [G]twinkle [Am]little [F]star\n\n[C]月亮[G]代表[Am]我的心\n"
    lines = SH.parse_lyrics(text)
    assert [ln.kind for ln in lines] == ["label", "text", "blank", "text"]
    assert lines[1].text == "Twinkle twinkle little star" and [m[0] for m in lines[1].markers] == [0, 8, 16, 23]
    aligned = [(0, 2, "C:maj"), (2, 4, "G:maj"), (4, 6, "A:min"), (6, 8, "F:maj"),
               (8, 10, "C:maj"), (10, 12, "G:maj"), (12, 14, "A:min")]
    ours = [(0, 2, "C"), (2, 4, "G"), (4, 8, "Dm"), (8, 10, "C"), (10, 12, "G"), (12, 13, "Am"), (13, 14, "F")]
    pl = SH.place_chords(lines, aligned, ours)
    assert pl.placed and pl.share == 1.0
    out = SH.render_chordpro(pl)
    assert out.splitlines() == ["[Verse 1]", "[C]Twinkle [G]twinkle [Dm]little star", "", "[C]月亮[G]代表[Am]我[F]的心"]
    two = SH.two_line(out)
    assert two.splitlines()[1:3] == ["C       G       Dm", "Twinkle twinkle little star"]
    # CJK letters take two columns, so the chord line stays over the right syllable
    assert two.splitlines()[-2:] == ["C   G   Am F", "月亮代表我的心"]


def test_place_chords_falls_back_when_the_sheet_does_not_line_up():
    lines = SH.parse_lyrics("[C]la [G]la [Am]la [F]la\n")
    aligned = [(0, 2, "E:maj"), (2, 4, "B:maj"), (4, 6, "G#:min"), (6, 8, "F#:maj")]
    pl = SH.place_chords(lines, aligned, [(0, 4, "E"), (4, 8, "B")])
    assert not pl.placed
    assert SH.render_chordpro(pl) == "[C]la [G]la [Am]la [F]la\n"


def test_place_chords_can_time_the_markers_from_our_own_chords():
    """No sheet alignment: the markers are lined up with the detected chord sequence instead (families:
    the sheet's Fm matches our Fm7; our extra passing chords are skipped cheaply)."""
    lines = SH.parse_lyrics("[C]Twinkle [G]twinkle [Fm]little [F]star\n")
    events = [(0, 2, "C:maj"), (1, 2, "G:maj/3"), (2, 4, "G:maj"), (4, 6, "F:min7"), (6, 8, "F:maj")]
    ours = [(0, 1, "C"), (1, 2, "G/B"), (2, 4, "G"), (4, 6, "Fm7"), (6, 8, "F")]
    pl = SH.place_chords(lines, [], ours, events=events)
    assert pl.placed and pl.share == 1.0
    assert SH.render_chordpro(pl) == "[C][G/B]Twinkle [G]twinkle [Fm7]little [F]star\n"
    # a long stretch between two lines becomes a chord-only line (an interlude the sheet leaves out)
    lines2 = SH.parse_lyrics("[C]one [G]two\n[C]three [G]four\n")
    al = [(0, 1, "C:maj"), (1, 2, "G:maj"), (12, 13, "C:maj"), (13, 14, "G:maj")]
    ours2 = [(0, 1, "C"), (1, 2, "G"), (3, 6, "Am"), (6, 9, "F"), (9, 12, "Dm"), (12, 13, "C"), (13, 14, "G")]
    out = SH.render_chordpro(SH.place_chords(lines2, al, ours2))
    assert out == "[C]one [G]two\n[Am] [F] [Dm]\n[C]three [G]four\n"


def test_marker_times_interpolate_unmatched_markers():
    lines = SH.parse_lyrics("[C]a [X7]b [G]c\n")          # "X7" is not a chord: no time of its own
    times, share = SH.marker_times(lines, [(0, 1, "C:maj"), (4, 5, "G:maj")])
    assert times[0] == 0 and times[2] == 4 and 0 < times[1] < 4 and share == 1.0


# ---- the bar chart ---------------------------------------------------------------------------------

def test_bar_chart_round_trips_and_follows_level_slash_transpose():
    a = fixture()
    ex = SH.build(a, None, lyrics=False)
    assert ex.key == "B" and ex.bpm == 65 and ex.meter == "4/4" and ex.level == "full"
    chart = CH.parse(ex.chart_text)
    assert len(chart.bars) == len(a.bars) == 61
    assert ex.chart_text.splitlines()[0].startswith("| N . . G#m/C# |")
    assert "/" in ex.chart_text                                   # slash chords at Full
    plain = SH.build(a, None, level="simple", slash=False, lyrics=False, transpose=-1, pref="flat")
    assert plain.key == "Bb" and plain.key_original == "B" and plain.transpose == -1
    assert "/" not in plain.chart_text and "7" not in plain.chart_text
    assert "(original B, -1)" in "\n".join(plain.header_lines())
    text = ex.to_text()
    assert text.startswith("# 詩歌-回家 [歌詞] — Kelly:DD\n# Key: B\n") and '"." = the same chord continues' in text
    assert ex.lyrics_text is None and ex.to_json()["chordpro"] is None


def test_labels_match_display_json():
    a = fixture()
    _, fd = stub.fixture_paths()
    disp = schema.Display.model_validate_json(fd.read_text("utf-8"))
    chords = SH.main_chords(a)
    for t, lv, i in ((0, "full", 2), (3, "standard", 1), (7, "simple", 0)):
        labels = SH.label_main(a, chords, transpose=t, level=lv, slash=True)
        want = [disp.variants[f"{t}:auto"].chords[c.id][i] for c in chords]
        # slash=True shows every bass (display.json uses the per-level default), so compare without the bass
        assert [x.split("/")[0] for x in labels] == [x.split("/")[0] for x in want]
    assert SH.label_main(a, chords, level="full", slash=None if False else True) == SH.label_main(a, chords)


# ---- the route --------------------------------------------------------------------------------------

def _lyrics_for(a: schema.Analysis, transpose_harte: int = 0, per_line: int = 2) -> tuple[str, list[dict]]:
    """A CantonHymn-like lyrics text whose chords are the fixture's main chords, plus the aligned events."""
    mains = [c for c in SH.main_chords(a) if c.harte != "N"]
    lines, aligned, row = [], [], []
    for i, c in enumerate(mains):
        if i % 16 == 0:
            if row:
                lines.append(" ".join(row))
                row = []
            lines.append("")
            lines.append(f"[{'Verse ' + str(i // 16 + 1) if i < 32 else 'Chorus'}]")
        sym = H.harte_to_symbol(c.harte)
        row.append(f"[{sym}]啦啦")
        aligned.append({"t0": c.start, "t1": c.end, "harte": c.harte, "labels": [sym, sym, sym]})
        if len(row) == per_line:
            lines.append(" ".join(row))
            row = []
    if row:
        lines.append(" ".join(row))
    return "\n".join(lines).strip() + "\n", aligned


def _put_sheet(d: Path, rid: str, lyrics: str, aligned: list[dict], applied: bool, source: str = "cantonhymn") -> None:
    st = L.load_state(d)
    st["status"] = "done"
    st["results"] = (st.get("results") or []) + [{
        "id": rid, "source": source, "url": None, "title": "回家", "artist": None, "year": 2007, "key_written": "B",
        "key_recording": "B", "transpose": 0, "meter": "4/4", "n_chords": len(aligned),
        "match": {"title": 1, "align": 0.9, "score": 0.9}, "verdict": "good", "video_match": False, "also": [],
        "chart_text": "| B . . . |", "aligned": aligned, "sections": [], "warnings": [], "lyrics_text": lyrics}]
    if applied:
        st["applied"] = {"id": rid, "group": f"sheet-{rid}-x", "mode": "refine", "ts": "2026-11-04T00:00:00Z"}
    L.save_state(d, st)


def test_export_sheet_route_txt_json_pdf_and_no_lyrics(env):
    a = put_cached_song(KEY)
    with client() as c:
        r = c.get(f"/api/tracks/{KEY}/export/sheet")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
        assert "chords.txt" in r.headers["content-disposition"] and f"{KEY}-chords.txt" in r.headers["content-disposition"]
        body = r.text
        assert body.startswith("# ") and "| N . . G#m/C# |" in body
        assert len(CH.parse("\n".join(ln for ln in body.splitlines() if not ln.startswith("#"))).bars) == len(a.bars)
        r = c.get(f"/api/tracks/{KEY}/export/sheet?format=txt&transpose=2&level=standard&slash=0&acc=sharp")
        assert r.status_code == 200 and "# Key: C#" in r.text and "/" not in r.text.split("\n", 12)[-1]
        j = c.get(f"/api/tracks/{KEY}/export/sheet?format=json&transpose=-1").json()
        assert j["key"] == "Bb" and j["key_original"] == "B" and j["lyrics_text"] is None and j["chordpro"] is None
        assert j["text"].startswith("# ") and j["chart_text"].startswith("| ")
        r = c.get(f"/api/tracks/{KEY}/export/sheet?format=chordpro")
        assert r.status_code == 404 and r.json()["code"] == "no_lyrics"
        r = c.get(f"/api/tracks/{KEY}/export/sheet?format=pdf")
        assert r.status_code == 501 and "Save as PDF" in r.json()["hint"]
        assert c.get(f"/api/tracks/{KEY}/export/sheet?format=docx").status_code == 400
        assert c.get(f"/api/tracks/{KEY}/export/sheet?level=huge").status_code == 400
        assert c.get("/api/tracks/yt-NOSUCHSONG00/export/sheet").status_code == 404


def test_export_sheet_places_our_chords_over_the_fetched_lyrics(env):
    a = put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    lyrics, aligned = _lyrics_for(a)
    # a first result without lyrics, then the matching one (applied): the applied one is used
    _put_sheet(d, "ch-0", "", [], applied=False)
    _put_sheet(d, "ch-1", lyrics, aligned, applied=True)
    with client() as c:
        j = c.get(f"/api/tracks/{KEY}/export/sheet?format=json").json()
        assert j["lyrics_ours"] is True and j["lyrics_sheet_id"] == "ch-1" and j["lyrics_source"] == "cantonhymn"
        cp = j["lyrics_chordpro"]
        assert cp.splitlines()[0] == "[Verse 1]" and "[Chorus]" in cp
        # the words are the sheet's, the chords are ours (same number of lines, our full labels inline)
        assert len(cp.splitlines()) == len(lyrics.splitlines())
        ours = [x.split("/")[0] for x in SH.label_main(a, [c for c in SH.main_chords(a) if c.harte != "N"])]
        inline = [m for m in __import__("re").findall(r"\[([^\[\]]+)\]", cp) if m not in ("Verse 1", "Verse 2", "Chorus")]
        assert 0.8 * len(set(ours)) <= len(set(x.split("/")[0] for x in inline))
        assert "G#m/C#" in inline and "D#m7" in inline
        assert j["lyrics_text"].splitlines()[0] == "[Verse 1]" and "啦啦" in j["lyrics_text"]
        # the two-line form: a chord line above every lyric line
        two = j["lyrics_text"].splitlines()
        i = two.index(next(ln for ln in two if "啦" in ln))
        assert two[i - 1].strip() and "啦" not in two[i - 1]
        r = c.get(f"/api/tracks/{KEY}/export/sheet?format=chordpro&transpose=1&level=simple&slash=0")
        assert r.status_code == 200 and "chords.cho" in r.headers["content-disposition"]
        body = r.text
        assert body.startswith("{title: ") and "{key: C}" in body and "{comment: Verse 1}" in body
        assert "{start_of_tab}" in body and "| " in body and "/" not in body.split("{start_of_tab}")[1]
        assert "[Verse 1]" not in body and "[C#" not in body       # simple + no slash, moved to C
        txt = c.get(f"/api/tracks/{KEY}/export/sheet?format=txt").text
        assert "# Chords over the words" in txt and "啦啦" in txt
        # pick another sheet explicitly / turn the lyrics off
        assert c.get(f"/api/tracks/{KEY}/export/sheet?format=json&sheet=ch-0").json()["lyrics_sheet_id"] == "ch-1"
        assert c.get(f"/api/tracks/{KEY}/export/sheet?format=json&lyrics=0").json()["lyrics_text"] is None


def test_export_sheet_keeps_the_sheets_chords_when_alignment_is_poor(env):
    a = put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    lyrics, aligned = _lyrics_for(a)
    # the lyrics of ANOTHER song: chords that never match ours
    other = "\n".join(f"[Dm]啦啦 [Gm]啦啦" for _ in range(20)) + "\n"
    _put_sheet(d, "sw-1", other, aligned, applied=False, source="simplyworship")
    with client() as c:
        j = c.get(f"/api/tracks/{KEY}/export/sheet?format=json&transpose=2").json()
        assert j["lyrics_ours"] is False and any("sheet's own" in n for n in j["notes"])
        assert j["lyrics_chordpro"].splitlines()[0] == "[Em]啦啦 [Am]啦啦"      # the sheet's chords, moved up 2
        assert "# The sheet's own chords over the words" in j["text"]
    # the lyrics file on disk is used when the state has no text
    (d / "sheets" / "sw-1.lyrics.txt").write_text(lyrics, "utf-8")
    st = L.load_state(d)
    st["results"][0]["lyrics_text"] = None
    L.save_state(d, st)
    src = SH.lyrics_source(d)
    assert src is not None and src["id"] == "sw-1" and "啦啦" in src["lyrics_text"]
