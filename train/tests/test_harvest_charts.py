"""train/harvest_charts.py: source parsers, chord clean-up, bar inference, and every harvested chart.

All inputs below are synthetic ("la la" lyrics); no network is used."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nathan_chordify.theory import chart as CH  # noqa: E402
from train import harvest_charts as HC  # noqa: E402


# ---------------------------------------------------------------------------------------------
# chord symbols
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tok, sym", [
    ("G", "G"), ("Dsus", "Dsus4"), ("C2", "Csus2"), ("D2/F#", "Dadd9/F#"), ("A2/E", "Asus2/E"),
    ("Em7", "Em7"), ("(Em7)", "Em7"), ("B♭m7", "Bbm7"), ("C/B♭", "C/Bb"), ("F#m7b5", "F#ø7"),
    ("Cmaj7", "Cmaj7"), ("N.C.", "N"), ("Ｃ", "C"), ("Gsus4", "Gsus4"), ("Am7/G", "Am7/G"),
    ("Dsus2/4", "Dsus4"), ("Am(maj7)", "Am(maj7)"), ("F#m7(11)", "F#m11"),
])
def test_norm_chord_accepts(tok, sym):
    out, _ = HC.norm_chord(tok)
    assert out == sym
    CH.parse(f"| {out} |")                    # the repo's parser accepts it


@pytest.mark.parametrize("tok, sym", [("Cmaj13", "Cmaj9"), ("Cadd4", "C"), ("C4", "Csus4"), ("Dsus2sus4", "Dsus4")])
def test_norm_chord_simplifies_with_a_note(tok, sym):
    out, note = HC.norm_chord(tok)
    assert out == sym and note and "->" in note


@pytest.mark.parametrize("tok", ["", "la", "Chorus", "Bridge", "Amen", "Every", "x2", "//", "hello world"])
def test_norm_chord_rejects(tok):
    assert HC.norm_chord(tok)[0] is None


# ---------------------------------------------------------------------------------------------
# labels and syllables
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, label", [
    ("Verse 1", "Verse 1"), ("Verse1:", "Verse 1"), ("Chorus:", "Chorus"), ("[Bridge]", "Bridge"),
    ("Pre-Chorus 2", "Pre-Chorus 2"), ("Chorus x2", "Chorus"), ("副歌", "Chorus"), ("前奏：", "Intro"),
    ("間奏", "Interlude"), ("Ending", "Outro"), ("Tag", "Tag"),
])
def test_match_label(text, label):
    assert HC.match_label(text) == label


@pytest.mark.parametrize("text", ["la la la", "Where would I go", "我們", "Verse of the day is long"])
def test_match_label_ignores_lyrics(text):
    assert HC.match_label(text) is None


def test_syllables():
    assert HC.syllables("一二三四") == 4
    assert HC.syllables("beautiful") == 3
    assert HC.syllables("love") == 1
    assert HC.syllables("一二，三四") == 5            # a pause counts one
    assert HC.syllables(" // ") == 0


# ---------------------------------------------------------------------------------------------
# lines -> bars
# ---------------------------------------------------------------------------------------------

def _bars(chordpro: str, m: int = 4) -> str:
    lines = [HC.chordpro_line(x) for x in chordpro.strip().splitlines()]
    return HC.to_bar_text(HC.sectionize(lines, __import__("collections").Counter()), m)


def test_chordpro_line_segments():
    segs = HC.chordpro_line("la[F]la la[Am]la")
    assert [(s.chord, s.text) for s in segs] == [(None, "la"), ("F", "la la"), ("Am", "la")]


def test_one_chord_per_short_line_is_one_bar():
    text = _bars("""Verse 1
la la la [G]la
la la la la [D/F#]la la
la la la [G]la
la la la la [D/F#]la la""")
    assert text.splitlines() == ["[Verse 1]", "| G | D/F# | G | D/F# |"]


def test_close_chords_share_a_bar():
    text = _bars("""Chorus
la [G]la la [A]la la la la [Bm]la
[G]la [A]la la la la [Dsus4]la [D]""")
    assert text.splitlines() == ["[Chorus]", "| G . A . | Bm | G . A . | Dsus4 . D . |"]


def test_chordless_lines_extend_the_previous_chord():
    with_extra = _bars("[C]la la la la la la\nla la la la la la\n[G]la la la la la la")
    without = _bars("[C]la la la la la la\n[G]la la la la la la")
    assert CH.parse(with_extra).sections[0].bars and len(re.findall(r"\|", with_extra)) > len(re.findall(r"\|", without))


def test_instrumental_rhythm_marks():
    assert HC.instrumental_items([HC.Tok("chord", "C"), HC.Tok("text", " // "), HC.Tok("chord", "G"),
                                  HC.Tok("text", " // ")], 4) == [("C", 2), ("G", 2)]
    toks = [HC.Tok("text", "| "), HC.Tok("chord", "F"), HC.Tok("text", " | "), HC.Tok("chord", "C"),
            HC.Tok("text", " "), HC.Tok("chord", "G"), HC.Tok("text", " |")]
    assert HC.instrumental_items(toks, 4) == [("F", 4), ("C", 2), ("G", 2)]
    toks = [HC.Tok("chord", "G"), HC.Tok("text", " / / / "), HC.Tok("chord", "C"), HC.Tok("text", " / / /")]
    assert HC.instrumental_items(toks, 4) == [("G", 4), ("C", 4)]      # chord + 3 slashes = a bar
    assert HC.instrumental_items([HC.Tok("chord", "F"), HC.Tok("text", "  "), HC.Tok("chord", "C")], 4) == [("F", 4), ("C", 4)]
    line = HC.chordpro_line("| [Bm] / / [A/C#] | [D] / / / | [A] / / / |")
    assert HC.instrumental_items(HC.expand(line, HC.Counter()), 4) == [("Bm", 3), ("A/C#", 1), ("D", 4), ("A", 4)]


def test_meter_from_bar_lines_and_slashes():
    four = HC.Raw([HC.chordpro_line("| [Bm] / / [A] | [D] / / / | [A] / / / | [G] / / / |")])
    three = HC.Raw([HC.chordpro_line("| [G] / / | [C] / / | [D] / / | [G] / / |")])
    assert HC.slash_meter(four) == "4/4" and HC.slash_meter(three) == "3/4"
    assert HC.slash_meter(HC.Raw([HC.chordpro_line("[C] // [G] // [D] // [Em] //")])) is None


def test_intro_line_from_slashes():
    text = _bars("Intro\n[C // G // D/F# // Em //]")
    assert text.splitlines() == ["[Intro]", "| C . G . | D/F# . Em . |"]


def test_three_four():
    text = _bars("[D]la la la la la [G]la\n[A]la la la [D]la", m=3)
    CH.parse(text, 3)
    assert all(len(b.split()) in (1, 3) for b in re.findall(r"\| ([^|]+) (?=\|)", text))


# ---------------------------------------------------------------------------------------------
# source parsers (synthetic pages)
# ---------------------------------------------------------------------------------------------

WT_PAGE = """<html><div class="chord-pro-disp" style="">
<div class="chord-pro-line"><div class="chord-pro-segment"><div class="chord-pro-note">&nbsp;</div>
<div class="chord-pro-lyric">Verse 1</div></div></div>
<div class="chord-pro-line"><div class="chord-pro-segment"><div class="chord-pro-note">&nbsp;</div>
<div class="chord-pro-lyric">la la </div></div><div class="chord-pro-segment">
<div class="chord-pro-note">G&nbsp;</div><div class="chord-pro-lyric">la la la</div></div>
<div class="chord-pro-segment"><div class="chord-pro-note">C2&nbsp;</div><div class="chord-pro-lyric">la</div></div></div>
<div class="chord-pro-line"><div class="chord-pro-segment"><div class="chord-pro-note">D/F#&nbsp;</div>
<div class="chord-pro-lyric">la la</div></div><div class="chord-pro-segment"><div class="chord-pro-note">Em&nbsp;</div>
<div class="chord-pro-lyric">la</div></div></div>
</div>
<div class="detail"><span>BPM:</span><p>72</p></div>
<div class="detail"><span>Original Key:</span><p><a href="#">G</a></p></div></html>"""


def test_parse_worshiptogether():
    raw = HC.parse_worshiptogether(WT_PAGE)
    assert raw.meta["key"] == "G" and raw.meta["tempo"] == 72.0
    chords = [s.chord for ln in raw.lines for s in ln if s.chord]
    assert chords == ["G", "C2", "D/F#", "Em"]


def test_parse_cantonhymn():
    payload = json.dumps({"success": True, "data": {"currentSong": {
        "title": "測試", "lyrics": "Verse 1\n啦[C]啦啦[G]啦\n[Am]啦啦啦[F]啦",
        "songkey": {"label": "C"}, "songyear_of_publication": [{"label": "2001"}],
        "songlanguage": [{"label": "粵語"}], "owner": [{"label": "某團"}]}}})
    raw = HC.parse_cantonhymn(payload)
    assert raw.meta["key"] == "C" and raw.meta["year"] == "2001"
    assert [s.chord for ln in raw.lines for s in ln if s.chord] == ["C", "G", "Am", "F"]


def test_parse_simplyworship():
    wrap = ("<span class=KCNchordWrap style=display:inline-block;><span data-original-chord=c class='KCNchord'>C</span>"
            "<span class=KCNchordWrapQuality></span> /<span class='KCN-inversion KCNchord'>E</span></span>")
    page = ("<div class=KCN-content><h3 class='KCNkeyTitle'>Key of <span class=KCNkey>C</span> "
            "<span class=KCNkeyQuality></span></h3><p>廣東話    3/4    Tempo=80</p><p><strong>Verse1:</strong></p>"
            "<p>啦<span class=KCNchordWrap style=display:inline-block;><span class='KCNchord'>A</span>"
            "<span class=KCNchordWrapQuality>m7</span><span></span></span>啦啦" + wrap + "啦</p></div></article>")
    raw = HC.parse_simplyworship(page)
    assert raw.meta["meter"] == "3/4" and raw.meta["tempo"] == 80.0 and raw.meta["key"] == "C"
    assert [s.chord for ln in raw.lines for s in ln if s.chord] == ["Am7", "C/E"]


def test_parse_ultimateguitar_columns():
    content = ("[Verse 1]\n[tab][ch]G[/ch]       [ch]C[/ch]\nla la la la la[/tab]\n"
               "[ch]D[/ch] [ch]Em[/ch]\n\nChords Used:\n[ch]G[/ch] - 320003\n")
    store = {"store": {"page": {"data": {"tab": {"tonality_name": "G", "capo": 0},
                                         "tab_view": {"wiki_tab": {"content": content}}}}}}
    page = '<div class="js-store" data-content="' + json.dumps(store).replace('"', "&quot;") + '"></div>'
    raw = HC.parse_ultimateguitar(page)
    segs = [ln for ln in raw.lines if any(s.chord for s in ln)]
    assert [(s.chord, s.text) for s in segs[0]] == [("G", "la la la"), ("C", " la la")]
    assert [s.chord for s in segs[1] if s.chord] == ["D", "Em"]
    assert len(segs) == 2                     # the fingering table is skipped


def test_build_chart_has_header_and_no_lyrics():
    song = HC.Song("t", "Test", "Nobody", "english")
    text, info = HC.build_chart(song, HC.parse_worshiptogether(WT_PAGE), "WorshipTogether", "https://x", "2026-11-01")
    assert text is not None, info
    assert HC.check_chart_text(text) == []
    assert "la" not in re.sub(r"(?m)^#.*$", "", text)
    hdr = HC.read_header(text)
    assert hdr["key"].startswith("G") and hdr["meter"].startswith("4/4") and "inferred" in hdr["bars"]


def test_check_chart_text_flags_lyrics():
    good = ("# title: x\n# language: english\n# key: C\n# meter: 4/4\n# source: s u\n# retrieved: d\n"
            "[Verse 1]\n| C | G |\n")
    assert HC.check_chart_text(good) == []
    assert any("not a bar line" in p for p in HC.check_chart_text(good + "la la la\n"))


def test_key_estimate():
    ev = [("G:maj", 4), ("D:maj/3", 4), ("E:min7", 4), ("C:maj7", 4), ("D:maj", 4), ("G:maj", 4)]
    assert HC.estimate_key(ev)[0] == 7                      # G major
    assert HC.estimate_key(ev, page_key=0)[0] == 7          # a wrong page key does not win
    assert HC.key_name(10) == "Bb" and HC.key_name(13) == "C#m"


# ---------------------------------------------------------------------------------------------
# polite fetcher (no network: a fake session)
# ---------------------------------------------------------------------------------------------

class _Resp:
    def __init__(self, code, text="ok"):
        self.status_code, self.text, self.encoding = code, text, "utf-8"


class _Session:
    def __init__(self, codes):
        self.codes = list(codes)
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        return _Resp(self.codes.pop(0))


def test_fetcher_spacing_cache_and_backoff(tmp_path):
    sleeps = []
    f = HC.Fetcher(tmp_path, sleep=sleeps.append)
    f._session = _Session([200, 200])
    assert f.get("https://example.org/a")[0] == "ok"
    assert f.get("https://example.org/b")[0] == "ok"
    assert sleeps and sleeps[-1] > 1.0                    # second request waited ~MIN_GAP
    assert f.get("https://example.org/a") == ("ok", "cache") and f._session.calls == 2
    g = HC.Fetcher(tmp_path, sleep=sleeps.append)
    g._session = _Session([429, 429, 429])
    sleeps.clear()
    assert g.get("https://example.org/c")[0] is None
    assert HC.BACKOFF[0] in sleeps and HC.BACKOFF[1] in sleeps   # backed off before retrying
    h = HC.Fetcher(tmp_path, offline=True)
    assert h.get("https://example.org/zzz") == (None, "offline")


# ---------------------------------------------------------------------------------------------
# the harvested data itself
# ---------------------------------------------------------------------------------------------

def test_test_songs_are_excluded():
    if not HC.CANDIDATES.exists():
        pytest.skip("no candidate list")
    ids = {s.id for s in HC.load_candidates()}
    assert not ids & HC.EXCLUDE_IDS
    files = {p.name.split(".")[0] for _, p in HC.chart_files()}
    assert not files & HC.EXCLUDE_IDS
    for _, p in HC.chart_files():
        title = HC.read_header(p.read_text(encoding="utf-8")).get("title", "").lower()
        assert not any(t in title for t in ("立志擺上", "活出愛", "here i bow", "10,000 reasons"))


def _chart_paths():
    return [p for _, p in HC.chart_files()]


@pytest.mark.skipif(not _chart_paths(), reason="no harvested charts yet")
@pytest.mark.parametrize("path", _chart_paths(), ids=lambda p: p.name)
def test_every_harvested_chart_parses(path):
    text = path.read_text(encoding="utf-8")
    assert HC.check_chart_text(text) == []
    hdr = HC.read_header(text)
    chart = CH.parse(text, int(hdr["beats_per_bar"]))
    assert sum(len(s.bars) for s in chart.sections) >= 4
