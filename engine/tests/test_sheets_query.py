"""sheets.query: messy video titles -> cleaned title / artist / ranked search terms."""
from __future__ import annotations

import pytest

from nathan_chordify.sheets import query as Q


@pytest.mark.parametrize("title, uploader, want_title, want_artist, want_terms, lang", [
    ("詩歌-回家 [歌詞]", "Kelly:DD", "回家", "Kelly:DD", ["回家"], "cjk"),
    ("〈立志擺上〉建道神學院新祢呈敬拜隊 官方歌詞版MV (曲詞:DJS 編曲:古丹青)", "建道神學院新祢呈敬拜隊",
     "立志擺上", "建道神學院新祢呈敬拜隊", ["立志擺上"], "cjk"),
    ("活出愛 Live Out Love 盛曉玫 Amy Sand 泥土音樂專輯 6 幸福", "泥土音樂Clay Music",
     "活出愛", "盛曉玫", ["活出愛", "Live Out Love"], "cjk"),
    ("Here I Bow (Official Lyric Video) -  Brian & Jenn Johnson | After All These Years", "Bethel Music",
     "Here I Bow", "Brian & Jenn Johnson", ["Here I Bow"], "en"),
    ("Matt Redman - 10,000 Reasons (Bless the Lord)", "Matt Redman",
     "10,000 Reasons", "Matt Redman", ["10,000 Reasons", "Bless the Lord"], "en"),
])
def test_the_five_real_titles(title, uploader, want_title, want_artist, want_terms, lang):
    q = Q.parse_title(title, uploader)
    assert q.title == want_title
    assert q.artist == want_artist
    assert q.terms == want_terms
    assert q.language == lang


def test_quoted_title_wins_and_cjk_terms_come_first():
    q = Q.parse_title("《恩典太美麗》Amazing Grace 讚美之泉 官方MV", "讚美之泉")
    assert q.title == "恩典太美麗"
    assert q.terms[0] == "恩典太美麗"
    assert q.language == "cjk"


def test_mixed_title_and_english_alias_split():
    q = Q.parse_title("回家 Going Home - 永恆音樂事工", "x")
    assert q.title == "回家" and q.artist == "永恆音樂事工"
    assert q.terms == ["回家", "Going Home"]


def test_english_artist_title_convention_and_subtitle_alias():
    q = Q.parse_title("Oceans (Where Feet May Fail) - Hillsong UNITED", "Hillsong UNITED")
    assert q.title == "Oceans" and q.artist == "Hillsong UNITED"
    assert "Where Feet May Fail" in q.terms
    q2 = Q.parse_title("Hillsong UNITED - Oceans (Live)", "Hillsong UNITED")
    assert q2.title == "Oceans" and q2.artist == "Hillsong UNITED"


def test_upload_file_name():
    q = Q.parse_title("03 - 全因為祢.mp3", None, "upload")
    assert q.title == "全因為祢" and q.artist is None and q.language == "cjk"
    q2 = Q.from_meta({"source_type": "upload", "title": "Here_I_Bow.m4a"})
    assert q2.title == "Here I Bow"


def test_uploader_noise_is_cleaned():
    assert Q.clean_uploader("Bethel Music - Topic") == "Bethel Music"
    assert Q.clean_uploader("讚美之泉官方頻道") == "讚美之泉"
    assert Q.clean_uploader(None) is None
