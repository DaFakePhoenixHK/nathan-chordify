"""sheets.parsers / sheets.fetch / sheets.sources (synthetic pages only; no network)."""
from __future__ import annotations

import json

import pytest

from nathan_chordify.sheets import fetch as F
from nathan_chordify.sheets import parsers as P
from nathan_chordify.sheets import sources as SRC
from nathan_chordify.theory import chart as CH

# a synthetic CantonHymn song-detail payload ("la" lyrics, inline chords)
CH_JSON = json.dumps({"success": True, "data": {
    "currentSong": {"title": "測試歌", "slug": "%e6%b8%ac%e8%a9%a6%e6%ad%8c", "songkey": {"label": "G"},
                    "songyear_of_publication": [{"label": "2010"}], "owner": [{"label": "某某樂團"}],
                    "songlanguage": [{"label": "粵語"}], "source": "",
                    "lyrics": "Verse 1\n[G]啦啦啦啦 [D/F#]啦啦啦啦\n[Em]啦啦啦啦 [C]啦啦啦啦\n\n副歌\n[C]啦啦啦啦 [D]啦啦啦啦\n[G]啦啦啦啦 [Em]啦啦啦啦\n[C]啦啦啦啦 [D]啦啦啦啦 [G]啦"},
    "currentSongVideosInfo": [{"type": "video_1_youtube_id", "label": "歌詞影片1", "youtubeId": "AAAAAAAAAAA"}],
    "parentSongVideosInfo": [],
    "otherRelatedSongs": [{"songId": 1, "slug": "x-2", "parentSlug": "%e6%b8%ac%e8%a9%a6%e6%ad%8c", "title": "測試歌 (學生版)",
                           "person": "某人", "hasChordPattern": True}],
}}, ensure_ascii=False)


def test_cantonhymn_json_to_chart_and_lyrics():
    raw = P.parse_cantonhymn(CH_JSON)
    assert raw is not None and raw.meta["key"] == "G" and raw.meta["year"] == "2010"
    text, info = P.build_chart_text(raw, source="CantonHymn", url="https://cantonhymn.net/song/x", retrieved="2026-11-03",
                                    title="測試歌", artist="某某樂團")
    assert text is not None and info["key"] == "G" and info["n_events"] >= 8
    hdr = P.read_header(text)
    assert hdr["title"] == "測試歌" and hdr["key"].startswith("G") and hdr["meter"].startswith("4/4")
    body = P.chart_body(text)
    assert body.splitlines()[0] == "[Verse 1]" and "[Chorus]" in body
    assert not P.check_chart_text(text)                      # parses, no lyric line slipped in
    assert "啦" not in text                                  # the chart text never holds lyrics
    lyr = P.raw_to_chordpro(raw)
    assert "[Verse 1]" in lyr and "[G]啦啦啦啦 [D/F#]啦啦啦啦" in lyr and "[Chorus]" in lyr
    moved = P.transpose_chordpro(lyr, 2, 2)                  # G -> A, spelled in A major
    assert "[A]啦啦啦啦 [E/G#]啦啦啦啦" in moved and "[Chorus]" in moved


def test_simplyworship_page_and_search_markup():
    page = ('<html><body><div class=KCN-content><h3><a href="/artist/x/">Some Band</a></h3>'
            '<p>廣東話 4/4 Tempo=65</p><p>Verse 1<br>'
            '<span class=KCNchordWrap><span class=KCNchord>C</span></span>la la la la '
            '<span class=KCNchordWrap><span class=KCNchord>G/B</span></span>la la la la<br>'
            '<span class=KCNchordWrap><span class=KCNchord>Am</span></span>la la la la '
            '<span class=KCNchordWrap><span class=KCNchord>F</span></span>la la la la</p>'
            '<p>Chorus<br><span class=KCNchordWrap><span class=KCNchord>F</span></span>la la la la '
            '<span class=KCNchordWrap><span class=KCNchord>G</span></span>la la la la '
            '<span class=KCNchordWrap><span class=KCNchord>C</span></span>la</p>'
            '</div></article></body></html>')
    raw = P.parse_simplyworship(page)
    assert raw is not None and raw.meta.get("meter") == "4/4" and raw.meta.get("tempo") == 65.0
    text, info = P.build_chart_text(raw, source="SimplyWorship", url="u", retrieved="d", title="t")
    assert text is not None and info["meter"] == "4/4" and info["n_events"] >= 6
    search = ('<article class="clearfix post-1 lyrics type-lyrics"><div><h3><a href="https://simplyworship.hk/lyrics/%e5%9b%9e%e5%ae%b6/?swpmtx=1&#038;x=2">回家</a></h3></div></article>'
              '<article class="post-2 lyrics"><h3><a href="https://simplyworship.hk/lyrics/other/">別的歌</a></h3></article>')

    class Fake:
        def get(self, url, max_age_s=None):
            return search, "200"
    cands = SRC.search_simplyworship("回家", Fake())
    assert [c.url for c in cands] == ["https://simplyworship.hk/lyrics/%e5%9b%9e%e5%ae%b6/"]
    assert cands[0].title == "回家" and cands[0].title_sim == 1.0


def test_cantonhymn_search_json_to_candidates():
    payload = {"success": True, "data": {"total": 3, "page_index": 1, "per_page": 12, "songs": [
        {"songId": 1, "slug": "%e5%9b%9e%e5%ae%b6", "parentSlug": None, "title": "回家", "person": "A",
         "featuredVideoLink": "https://www.youtube.com/watch?v=1Bf0btWKE0w", "hasChordPattern": True},
        {"songId": 2, "slug": "%e5%9b%9e%e5%ae%b6-4", "parentSlug": "%e5%9b%9e%e5%ae%b6", "title": "回家", "person": "B",
         "featuredVideoLink": "https://www.youtube.com/watch?v=Y7boUFbM76k", "hasChordPattern": False},
        {"songId": 3, "slug": "%e5%88%a5%e7%9a%84", "parentSlug": None, "title": "完全不同的歌名", "person": "C",
         "featuredVideoLink": "", "hasChordPattern": True},
    ]}}

    class Fake:
        def get_json(self, url, max_age_s=None):
            assert "advanced-search-songs.php" in url and "filter1_val=%E5%9B%9E%E5%AE%B6" in url
            return payload, "200"
    cands = SRC.search_cantonhymn("回家", Fake(), video_id="Y7boUFbM76k")
    assert [c.slug for c in cands] == ["%e5%9b%9e%e5%ae%b6-4", "%e5%9b%9e%e5%ae%b6"]   # the same-video one first
    assert cands[0].video_match and cands[0].parent_slug == "%e5%9b%9e%e5%ae%b6"
    assert cands[0].fetch_url.endswith("songHierarchyType=child") and cands[0].url.endswith("/%e5%9b%9e%e5%ae%b6/%e5%9b%9e%e5%ae%b6-4")
    assert cands[1].fetch_url.endswith("slug=%e5%9b%9e%e5%ae%b6&songHierarchyType=parent")


def test_cantonhymn_fetch_reads_videos_and_related():
    class Fake:
        def get(self, url, max_age_s=None):
            return CH_JSON, "200"

        def cached_date(self, url):
            return "2026-11-03"
    cand = SRC.Candidate("cantonhymn", "https://cantonhymn.net/song/x", "測試歌", None, "https://cantonhymn.net/api/x", slug="%e6%b8%ac%e8%a9%a6%e6%ad%8c")
    doc = SRC.fetch_cantonhymn(cand, Fake())
    assert doc is not None and doc.meta["youtube_ids"] == ["AAAAAAAAAAA"] and doc.meta["year"] == 2010
    assert doc.meta["key_written"] == "G" and doc.lyrics and "[Chorus]" in doc.lyrics
    assert [r.slug for r in doc.related] == ["x-2"] and doc.related[0].fetch_url.endswith("child")
    sig = SRC.chord_signature(doc.body)
    assert sig[:3] == ("G:maj", "D:maj/3", "E:min")


def test_worshiptogether_slug_guesses_and_url_candidates():
    assert SRC.wt_slug_guesses("10,000 Reasons (Bless the Lord)", "Matt Redman")[:3] == [
        "10-000-reasons", "10-000-reasons-bless-the-lord", "10-000-reasons-matt-redman"]
    c = SRC.candidate_from_url("https://www.worshiptogether.com/songs/here-i-bow/?nav=1")
    assert c is not None and c.source == "worshiptogether" and c.url == "https://www.worshiptogether.com/songs/here-i-bow/"
    c = SRC.candidate_from_url("cantonhymn.net/song/%e5%9b%9e%e5%ae%b6")
    assert c is not None and c.fetch_url.endswith("slug=%e5%9b%9e%e5%ae%b6&songHierarchyType=parent")
    assert SRC.candidate_from_url("https://example.com/chords/x") is None


def test_title_similarity():
    assert SRC.title_similarity("回家", "回家") == 1.0
    assert SRC.title_similarity("回家", "浪子回家") >= 0.9          # contains
    assert SRC.title_similarity("回家", "完全不同的歌名") < 0.3
    assert SRC.title_similarity("Here I Bow", "Here I Bow (Live)") == 1.0
    assert SRC.title_similarity("Here I Bow", "Here I Am To Worship") < 0.5


def test_fetcher_cache_age_and_json(tmp_path):
    f = F.Fetcher(tmp_path, offline=True)
    p = f._path("https://example.com/a")
    p.parent.mkdir(parents=True)
    p.write_text('{"a": 1}', "utf-8")
    assert f.get_json("https://example.com/a") == ({"a": 1}, "cache")
    assert f.get("https://example.com/a", max_age_s=-1) == (None, "offline")  # too old → would refetch → offline
    f.forget("https://example.com/a")
    assert f.get("https://example.com/a") == (None, "offline")
    assert F.is_online_error("offline") and F.is_online_error("error ConnectionError") and not F.is_online_error("404")


def test_chart_body_round_trip_parses():
    body = "[Verse 1]\n| G . D/F# . | Em . C . |\n[Chorus]\n| C | D | G | Em |\n"
    text = "# title: x\n# language: x\n# key: G\n# meter: 4/4\n# beats_per_bar: 4\n# source: pasted\n# retrieved: d\n" + body
    assert P.chart_body(text) == body
    CH.parse(body, 4)
    assert P.check_chart_text(text) == []
