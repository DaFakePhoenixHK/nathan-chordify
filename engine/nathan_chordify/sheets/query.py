"""Song meta (YouTube title + uploader, or an upload's file name) -> ranked search terms.

YouTube titles are messy: "詩歌-回家 [歌詞]", "〈立志擺上〉建道神學院新祢呈敬拜隊 官方歌詞版MV (曲詞:DJS 編曲:古丹青)",
"活出愛 Live Out Love 盛曉玫 Amy Sand 泥土音樂專輯 6 幸福", "Here I Bow (Official Lyric Video) - Brian & Jenn
Johnson | After All These Years", "Matt Redman - 10,000 Reasons (Bless the Lord)".

Rules, in order: a title inside 《》〈〉「」『』 wins; bracketed groups that only describe the video
(官方 / MV / 歌詞 / Official / Lyric Video / Live / Audio / credits such as 曲詞:…) are dropped, other
brackets become a subtitle alias; the part after "|" is an album / channel tail; " - " splits
artist and title (YouTube's "Artist - Title" unless one part carried the video descriptor or
equals the uploader); a run of Chinese followed by a run of English with no separator is a title
and its English alias (活出愛 Live Out Love), the next Chinese run the artist. CJK terms come first
for CJK titles, then the English alias.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_QUOTED = re.compile(r"《([^》]+)》|〈([^〉]+)〉|「([^」]+)」|『([^』]+)』")
_BRACKET = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]|【([^】]*)】|（([^）]*)）|〔([^〕]*)〕")

# a bracketed group that only describes the video / credits (not part of the title)
_NOISE_WORDS = (r"official|lyrics?|video|audio|live|acoustic|cover|mv|hd|4k|remaster(?:ed)?|version|ver\.?|"
                r"feat\.?|ft\.?|karaoke|instrumental|visuali[sz]er|session|chords?|ccli|piano|guitar|demo|"
                r"歌詞|字幕|官方|現場|伴奏|翻唱|高清|曲|詞|編曲|作曲|填詞|主唱|演唱|原唱|原創|版|專輯|詩歌|敬拜|讚美|"
                r"鋼琴|結他|吉他|示範|教學|和弦|和絃|樂譜")
_NOISE_GROUP = re.compile(rf"(?:{_NOISE_WORDS})", re.I)
_CREDIT = re.compile(r"[曲詞編作填主演唱]\s*[:：]")            # "曲詞:DJS 編曲:古丹青"
# standalone noise tokens anywhere (outside brackets)
_NOISE_TOKEN = re.compile(
    r"(?<![A-Za-z])(?:official(?:\s+(?:lyric|music)\s+video)?|lyric\s+video|lyrics?|music\s+video|audio|hd|4k|mv|"
    r"live\s+(?:version|at|in|from|recording|session)\b|acoustic|karaoke|instrumental|visuali[sz]er|remaster(?:ed)?)(?![A-Za-z])|"
    r"官方歌詞版MV|官方歌詞版|歌詞版MV|歌詞版|官方MV|歌詞字幕版|官方|歌詞|字幕|現場版|伴奏版|高清|MV", re.I)
_GENRE_WORDS = frozenset({"詩歌", "敬拜", "讚美", "基督教", "福音", "聖詩", "worship", "hymn", "christian", "gospel",
                          "敬拜詩歌", "詩歌敬拜", "讚美詩歌", "基督教詩歌", "粵語詩歌", "國語詩歌"})
_ALBUM_MARK = re.compile(r"專輯|album|ep\b", re.I)
_UPLOADER_NOISE = re.compile(r"\s*-\s*topic$|\bofficial\b|\bvevo\b|\bchannel\b|官方(?:頻道)?|頻道", re.I)
_SPLIT = re.compile(r"\s+[-–—~〜]\s+|(?<=[一-鿿）」》〉])[-–—](?=[一-鿿（「《〈A-Za-z])|(?<=[A-Za-z0-9）])[-–—](?=[一-鿿（])")
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’&.,!?:]*(?:\s+[A-Za-z0-9][A-Za-z0-9'’&.,!?:]*)*")
_AUDIO_EXT = re.compile(r"\.(mp3|m4a|wav|flac|ogg|opus|aac|wma|mp4|webm|mkv|aiff?)$", re.I)
_TRACK_NO = re.compile(r"^\s*\d{1,3}[\s._-]+")


@dataclass
class Query:
    title: str
    artist: str | None
    terms: list[str]                 # ranked search terms (title first; the English alias next)
    language: str                    # "cjk" | "en"
    aliases: list[str] = field(default_factory=list)
    subtitle: str | None = None

    def to_dict(self) -> dict:
        return {"title": self.title, "artist": self.artist, "terms": list(self.terms), "language": self.language,
                "aliases": list(self.aliases)}


def has_cjk(s: str) -> bool:
    return bool(_CJK.search(s or ""))


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("｜", "|").replace("／", "/").replace("－", "-").replace("　", " ")
    return re.sub(r"\s+", " ", s).strip()


def _is_noise_group(content: str) -> bool:
    c = content.strip()
    if not c:
        return True
    if _CREDIT.search(c):
        return True
    words = re.findall(r"[A-Za-z]+|[一-鿿]+", c)
    if not words:
        return True                                   # "(2019)", "(x2)"
    hits = sum(1 for w in words if _NOISE_GROUP.search(w))
    return hits >= max(1, len(words) // 2)


def _strip_noise_tokens(s: str) -> str:
    s = _NOISE_TOKEN.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—|/:：,，.。")
    return s


def clean_uploader(name: str | None) -> str | None:
    if not name:
        return None
    s = _UPLOADER_NOISE.sub(" ", _norm(name))
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s or None


def _same_name(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    na = re.sub(r"[^a-z0-9一-鿿]+", "", a.lower())
    nb = re.sub(r"[^a-z0-9一-鿿]+", "", b.lower())
    return bool(na) and bool(nb) and (na == nb or na in nb or nb in na)


def _drop_genre(s: str) -> str:
    toks = s.split(" ")
    toks = [t for t in toks if t.lower() not in _GENRE_WORDS]
    return " ".join(toks).strip()


def _mixed_runs(s: str) -> tuple[list[str], list[str]]:
    """(CJK runs, Latin runs) of a string without separators, in order."""
    cjk = [m.group(0) for m in re.finditer(r"[一-鿿][一-鿿，、・·]*", s)]
    latin = [m.group(0).strip() for m in _LATIN_RUN.finditer(s)]
    return cjk, latin


def _mixed_alias(p: str) -> bool:
    """"回家 Going Home": a Chinese run followed by an English run of two or more words."""
    cjk, latin = _mixed_runs(p)
    return bool(cjk) and bool(latin) and len(latin[0].split()) >= 2 and p.strip().startswith(cjk[0][0])


def _split_subtitle(title: str) -> tuple[str, str | None]:
    """"10,000 Reasons (Bless the Lord)" -> ("10,000 Reasons", "Bless the Lord")."""
    m = re.match(r"^(.*?)\s*[(（]([^()（）]+)[)）]\s*$", title)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2).strip()
    return title.strip(), None


def parse_title(title: str, uploader: str | None = None, source_type: str = "youtube") -> Query:
    """Cleaned title / artist / ranked search terms from a video title (or a file name)."""
    s = _norm(title)
    if source_type == "upload":
        s = _TRACK_NO.sub("", _AUDIO_EXT.sub("", s)).replace("_", " ")
    uploader_c = clean_uploader(uploader)

    # 1. a quoted title wins
    quoted = [next(g for g in m.groups() if g) .strip() for m in _QUOTED.finditer(s)]
    rest = _QUOTED.sub(" ", s) if quoted else s

    # 2. bracket groups: drop the video descriptors, keep the rest as a possible subtitle
    subtitles: list[str] = []

    def _bracket(m: re.Match) -> str:
        content = next(g for g in m.groups() if g is not None)
        if _is_noise_group(content):
            return " \x00 "                              # a marker: "this part carried a descriptor"
        subtitles.append(content.strip())
        return f" ({content.strip()}) "
    rest = _BRACKET.sub(_bracket, rest)
    for _ in range(2):                                   # nested brackets
        rest = _BRACKET.sub(_bracket, rest)

    # 3. the album / channel tail after "|"
    segs = [x.strip() for x in rest.split("|")]
    main, tail = segs[0], [x for x in segs[1:] if x]

    # 4. artist / title parts
    parts_raw = [p for p in _SPLIT.split(main) if p and p.strip()]
    marked = [("\x00" in p) for p in parts_raw]
    parts = []
    for p in parts_raw:
        p = _strip_noise_tokens(p.replace("\x00", " "))
        p = _drop_genre(p)
        parts.append(p)
    pairs = [(p, mk) for p, mk in zip(parts, marked) if p]
    parts = [p for p, _ in pairs]
    marked = [mk for _, mk in pairs]

    artist: str | None = None
    aliases: list[str] = []
    title_out: str | None = None
    if quoted:
        title_out = quoted[0]
        aliases += quoted[1:]
        cand = " ".join(parts).strip()
        if cand:
            cjk, latin = _mixed_runs(cand)
            artist = (cjk[0] if cjk else (latin[0] if latin else None))
    elif len(parts) >= 2:
        if any(marked):
            ti = marked.index(True)                      # the part with the video descriptor is the title
        elif any(_same_name(p, uploader_c) for p in parts):
            ti = next(i for i, p in enumerate(parts) if not _same_name(p, uploader_c))
        elif has_cjk(parts[0]) and not has_cjk(parts[1]) and len(parts) == 2 and _LATIN_RUN.fullmatch(parts[1]):
            ti = 0                                        # "回家 - Going Home": title + English alias
            aliases.append(parts[1])
        elif any(_mixed_alias(p) for p in parts) and not all(_mixed_alias(p) for p in parts):
            ti = next(i for i, p in enumerate(parts) if _mixed_alias(p))   # "回家 Going Home - 永恆音樂事工"
        else:
            ti = 1                                        # YouTube's "Artist - Title"
        title_out = parts[ti]
        others = [p for i, p in enumerate(parts) if i != ti and p not in aliases]
        artist = others[0] if others else None
    elif parts:
        p = parts[0]
        if has_cjk(p) and re.search(r"[A-Za-z]", p):
            cjk, latin = _mixed_runs(_ALBUM_MARK.split(p)[0] if _ALBUM_MARK.search(p) else p)
            cjk = [c for c in cjk if c not in _GENRE_WORDS]
            if cjk:
                title_out = cjk[0]
                if latin and len(latin[0].split()) >= 2 and not _NOISE_GROUP.fullmatch(latin[0]):
                    aliases.append(latin[0])
                if len(cjk) >= 2:
                    artist = cjk[1]
                elif len(latin) >= 2:
                    artist = latin[1]
            else:
                title_out = latin[0] if latin else p
        else:
            title_out = p
    else:
        title_out = _strip_noise_tokens(s) or s

    if _mixed_alias(title_out):                           # "回家 Going Home" -> 回家 + alias
        cjk, latin = _mixed_runs(title_out)
        title_out = cjk[0]
        aliases.insert(0, latin[0])
    title_out, sub = _split_subtitle(title_out)
    if sub and not _is_noise_group(sub):
        aliases.append(sub)
    for st in subtitles:
        if st and st != sub and st not in aliases and len(st) >= 2 and not _is_noise_group(st):
            aliases.append(st)
    if title_out.lower().startswith(("intro ", "cover ")):
        title_out = title_out.split(" ", 1)[1]
    if artist is None:
        artist = uploader_c
    if artist and _same_name(artist, title_out):
        artist = uploader_c if not _same_name(uploader_c, title_out) else None
    language = "cjk" if has_cjk(title_out) else "en"
    # ranked terms: the title; the English alias; a non-noise quoted/bracketed alias
    terms: list[str] = []
    for t in [title_out] + aliases:
        t = t.strip()
        if t and t not in terms and len(t) >= 2:
            terms.append(t)
    if language == "cjk":                                 # CJK terms first for CJK titles
        terms.sort(key=lambda t: 0 if has_cjk(t) else 1)
    return Query(title_out, artist, terms, language, [a for a in aliases if a != title_out], sub)


def from_meta(meta: dict) -> Query:
    """The Query for a song's cache/<key>/meta.json (YouTube: title + uploader; upload: file name)."""
    source_type = meta.get("source_type") or "youtube"
    title = meta.get("title") or meta.get("filename") or ""
    return parse_title(title, meta.get("uploader") or meta.get("artist"), source_type)
