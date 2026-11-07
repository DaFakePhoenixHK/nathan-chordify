"""Where chord sheets come from: per-site search and chart fetching (polite, no login).

- CantonHymn (Cantonese / Mandarin): the React search of the site calls
  ``/api/advanced-search-songs.php?generalType=all&recommendType=hot&filter1_target=song_name
  &filter1_type=contain_all_words&filter1_val=<title>&as_page=N`` → JSON ``data.songs[]`` with
  slug, parentSlug (child versions), title, person, featuredVideoLink, hasChordPattern (the
  WordPress ``?s=`` search finds no songs). A song's chart is its song-detail JSON
  ``/api/song-detail.php?slug=<slug>&songHierarchyType=parent|child`` ("[C]" inline chords, key,
  year, ``currentSongVideosInfo[].youtubeId``, ``otherRelatedSongs`` with ``hasChordPattern``).
  A page whose YouTube ids include the recording's id is a certain match (``video_match``).
- Simply Worship (simplyworship.hk): WordPress ``?s=<title>`` → ``<article class="… lyrics …">``
  with the song link; the lyric page has inline chords, key, time signature, tempo.
- WorshipTogether (English): its search is a third-party widget (Cludo), so the song page is
  guessed from the title: ``/songs/<title-slug>/`` then ``/songs/<title-slug>-<artist-slug>/``.
- Ultimate Guitar (English fallback): the public search page, most-voted public "Chords" version.

Politeness: ``fetch.Fetcher`` spaces requests (≥ 1.2 s, CantonHymn ≥ 2.5 s), backs off and caches
pages outside the repo (24 h; search results 7 days). At most ``MAX_PER_SOURCE`` charts per
source are fetched for one song.
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field

from ..ingest.resolve import youtube_id
from . import parsers as P
from .fetch import PAGE_MAX_AGE_S, SEARCH_MAX_AGE_S, Fetcher, is_online_error, today

SOURCES = ("cantonhymn", "simplyworship", "worshiptogether", "ultimateguitar")
SOURCE_NAMES = {"cantonhymn": "CantonHymn", "simplyworship": "SimplyWorship", "worshiptogether": "WorshipTogether",
                "ultimateguitar": "UltimateGuitar", "pasted": "pasted"}
ORDER = {"cjk": ("cantonhymn", "simplyworship"), "en": ("worshiptogether", "ultimateguitar")}
MAX_PER_SOURCE = 6
MIN_TITLE_SIM = 0.5
CH_SEARCH = ("https://cantonhymn.net/api/advanced-search-songs.php?generalType=all&recommendType=hot"
             "&filter1_target=song_name&filter1_type=contain_all_words&filter1_val={q}&as_page={page}")
CH_DETAIL = "https://cantonhymn.net/api/song-detail.php?slug={slug}&songHierarchyType={kind}"
CH_SONG = "https://cantonhymn.net/song/{path}"
SW_SEARCH = "https://simplyworship.hk/?s={q}"
WT_SONG = "https://www.worshiptogether.com/songs/{slug}/"
UG_SEARCH = "https://www.ultimate-guitar.com/search.php?search_type=title&value={q}"


class SourceOffline(Exception):
    """Every request to a site failed (no network, or the site is down / blocking)."""


@dataclass
class Candidate:
    source: str
    url: str                                 # the page a person would open
    title: str
    artist: str | None = None
    fetch_url: str | None = None             # what we actually download (API JSON for CantonHymn)
    year: int | None = None
    video_ids: list[str] = field(default_factory=list)
    title_sim: float = 0.0
    slug: str | None = None
    parent_slug: str | None = None
    has_chart: bool = True
    note: str = ""

    @property
    def id(self) -> str:
        return result_id(self.source, self.url)

    @property
    def video_match(self) -> bool:
        return bool(getattr(self, "_video_match", False))


@dataclass
class ChartDoc:
    text: str                                # chart file text (header + bar text), as written
    body: str                                # the bar text alone
    lyrics: str | None                       # ChordPro-style chords over lyrics (user's cache only)
    meta: dict                               # key_written, key_index, meter, tempo, year, artist, title, youtube_ids, capo, n_events
    related: list[Candidate] = field(default_factory=list)   # other versions found on the page


def result_id(source: str, url_or_text: str) -> str:
    return f"{source}:{hashlib.sha1(url_or_text.encode('utf-8')).hexdigest()[:10]}"


# ---------------------------------------------------------------------------------------- titles

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_STOP = frozenset({"the", "a", "an", "of", "and", "to", "in", "my", "your", "our", "is", "be"})


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[（(\[【][^)）\]】]*[)）\]】]", " ", s)            # bracketed subtitles / notes
    s = re.sub(r"[^\w㐀-鿿]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_similarity(a: str, b: str) -> float:
    """0–1: exact (normalised) match 1; CJK: character overlap of the shorter in the longer;
    English: word overlap (stop words dropped)."""
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if _CJK.search(na) or _CJK.search(nb):
        ca, cb = [c for c in na if not c.isspace()], [c for c in nb if not c.isspace()]
        short, long_ = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
        if not short:
            return 0.0
        if "".join(short) in "".join(long_):
            return 0.95 if len(short) >= 2 else 0.6
        common = sum(1 for c in short if c in long_)
        return 0.9 * common / len(short) * (len(short) / len(long_)) ** 0.5
    wa = [w for w in na.split() if w not in _STOP] or na.split()
    wb = [w for w in nb.split() if w not in _STOP] or nb.split()
    if not wa or not wb:
        return 0.0
    sa, sb = set(wa), set(wb)
    j = len(sa & sb) / len(sa | sb)
    if na in nb or nb in na:
        j = max(j, 0.85)
    return j


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s.lower()).strip("-")
    return s


# ------------------------------------------------------------------------------------- CantonHymn

def _ch_video(song: dict) -> list[str]:
    out = []
    for k in ("featuredVideoLink",):
        v = youtube_id(song.get(k) or "")
        if v:
            out.append(v)
    return out


def search_cantonhymn(term: str, fetcher: Fetcher, video_id: str | None = None, pages: int = 2) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    errors = 0
    for page in range(1, pages + 1):
        url = CH_SEARCH.format(q=urllib.parse.quote(term, safe=""), page=page)
        d, status = fetcher.get_json(url, SEARCH_MAX_AGE_S)
        if d is None:
            if is_online_error(status):
                errors += 1
            break
        data = (d or {}).get("data") or {}
        songs = data.get("songs") or []
        for s in songs:
            slug = s.get("slug") or ""
            if not slug or slug in seen:
                continue
            seen.add(slug)
            sim = title_similarity(term, s.get("title") or "")
            vids = _ch_video(s)
            if sim < MIN_TITLE_SIM and not (video_id and video_id in vids):
                continue
            parent = s.get("parentSlug") or None
            path = f"{parent}/{slug}" if parent else slug
            kind = "child" if parent else "parent"
            c = Candidate("cantonhymn", CH_SONG.format(path=path), s.get("title") or term, s.get("person") or None,
                          CH_DETAIL.format(slug=slug, kind=kind), None, vids, sim, slug, parent,
                          bool(s.get("hasChordPattern")), "")
            c._video_match = bool(video_id and video_id in vids)    # type: ignore[attr-defined]
            out.append(c)
        total = int(data.get("total") or 0)
        per = int(data.get("per_page") or 12)
        if page * per >= total:
            break
    if errors and not out:
        raise SourceOffline("cantonhymn")
    out.sort(key=lambda c: (not c.video_match, not c.has_chart, -c.title_sim))
    return out


def fetch_cantonhymn(cand: Candidate, fetcher: Fetcher) -> ChartDoc | None:
    url = cand.fetch_url or P.ch_api(cand.url, "child" if cand.parent_slug else "parent")
    text, status = fetcher.get(url, PAGE_MAX_AGE_S)
    if text is None and cand.parent_slug and not is_online_error(status):
        text, status = fetcher.get(url.replace("songHierarchyType=child", "songHierarchyType=parent"), PAGE_MAX_AGE_S)
    if text is None:
        if is_online_error(status):
            raise SourceOffline("cantonhymn")
        return None
    raw = P.parse_cantonhymn(text)
    if raw is None and cand.parent_slug:                      # a child slug answered as a parent page
        alt = url.replace("songHierarchyType=child", "songHierarchyType=parent")
        text2, _ = fetcher.get(alt, PAGE_MAX_AGE_S)
        raw = P.parse_cantonhymn(text2) if text2 else None
    if raw is None:
        return None
    import json
    try:
        d = json.loads(text)["data"]
    except (ValueError, KeyError, TypeError):
        d = {}
    vids = [v.get("youtubeId") for v in (d.get("currentSongVideosInfo") or []) if v.get("youtubeId")]
    vids += [v.get("youtubeId") for v in (d.get("parentSongVideosInfo") or []) if v.get("youtubeId")]
    vids += cand.video_ids
    related: list[Candidate] = []
    for s in d.get("otherRelatedSongs") or []:
        slug = s.get("slug") or ""
        if not slug or slug == cand.slug or not s.get("hasChordPattern"):
            continue
        parent = s.get("parentSlug") or None
        path = f"{parent}/{slug}" if parent else slug
        related.append(Candidate("cantonhymn", CH_SONG.format(path=path), s.get("title") or cand.title,
                                 s.get("person") or None, CH_DETAIL.format(slug=slug, kind="child" if parent else "parent"),
                                 None, _ch_video(s), title_similarity(cand.title, s.get("title") or ""), slug, parent, True,
                                 "another version listed on the page"))
    chart, info = P.build_chart_text(raw, source="CantonHymn", url=cand.url, retrieved=fetcher.cached_date(url) or today(),
                                     title=cand.title, artist=cand.artist or raw.meta.get("artist"))
    if chart is None:
        return None
    year = raw.meta.get("year")
    meta = {"key_written": raw.meta.get("key"), "key_index": info["page_key"] if info["page_key"] is not None else info["key_index"],
            "meter": raw.meta.get("meter"), "tempo": info["tempo"], "year": int(year) if str(year or "").isdigit() else None,
            "artist": raw.meta.get("artist") or cand.artist, "title": raw.meta.get("title") or cand.title,
            "youtube_ids": list(dict.fromkeys(v for v in vids if v)), "capo": None, "n_events": info["n_events"]}
    return ChartDoc(chart, P.chart_body(chart), P.raw_to_chordpro(raw), meta, related)


# --------------------------------------------------------------------------------- Simply Worship

_SW_ARTICLE = re.compile(r"<article[^>]*class=\"[^\"]*\blyrics\b[^\"]*\"[^>]*>(.*?)</article>", re.S)
_SW_LINK = re.compile(r"<h\d[^>]*>\s*<a\s+href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.S)


def search_simplyworship(term: str, fetcher: Fetcher) -> list[Candidate]:
    url = SW_SEARCH.format(q=urllib.parse.quote(term, safe=""))
    page, status = fetcher.get(url, SEARCH_MAX_AGE_S)
    if page is None:
        if is_online_error(status):
            raise SourceOffline("simplyworship")
        return []
    out: list[Candidate] = []
    seen: set[str] = set()
    for m in _SW_ARTICLE.finditer(page):
        lm = _SW_LINK.search(m.group(1))
        if not lm:
            continue
        href = html.unescape(lm.group(1)).split("?")[0].split("#")[0]
        title = P._strip_tags(lm.group(2)).strip()
        if not href or href in seen or "/lyrics/" not in href:
            continue
        seen.add(href)
        sim = title_similarity(term, title)
        if sim < MIN_TITLE_SIM:
            continue
        out.append(Candidate("simplyworship", href, title, None, href, None, [], sim))
    out.sort(key=lambda c: -c.title_sim)
    return out


def fetch_simplyworship(cand: Candidate, fetcher: Fetcher) -> ChartDoc | None:
    text, status = fetcher.get(cand.fetch_url or cand.url, PAGE_MAX_AGE_S)
    if text is None:
        if is_online_error(status):
            raise SourceOffline("simplyworship")
        return None
    raw = P.parse_simplyworship(text)
    if raw is None:
        return None
    chart, info = P.build_chart_text(raw, source="SimplyWorship", url=cand.url,
                                     retrieved=fetcher.cached_date(cand.fetch_url or cand.url) or today(),
                                     title=cand.title, artist=cand.artist or raw.meta.get("artist"))
    if chart is None:
        return None
    meta = {"key_written": raw.meta.get("key"), "key_index": info["page_key"] if info["page_key"] is not None else info["key_index"],
            "meter": raw.meta.get("meter"), "tempo": info["tempo"], "year": None, "artist": raw.meta.get("artist") or cand.artist,
            "title": cand.title, "youtube_ids": [v for v in re.findall(r"youtube\.com/embed/([A-Za-z0-9_-]{11})", text)],
            "capo": None, "n_events": info["n_events"]}
    return ChartDoc(chart, P.chart_body(chart), P.raw_to_chordpro(raw), meta)


# -------------------------------------------------------------------------------- WorshipTogether

def wt_slug_guesses(title: str, artist: str | None) -> list[str]:
    t = slugify(re.sub(r"[（(\[【][^)）\]】]*[)）\]】]", " ", title))
    if not t:
        return []
    out = [t]
    full = slugify(title)
    if full and full != t:
        out.append(full)
    if artist:
        a = slugify(re.sub(r"\b(music|band|collective|feat\.?|ft\.?)\b", " ", artist, flags=re.I))
        if a:
            out.append(f"{t}-{a}")
            last = a.split("-")[-1]
            if last and last != a:
                out.append(f"{t}-{last}")
    return list(dict.fromkeys(out))


def search_worshiptogether(title: str, artist: str | None, fetcher: Fetcher) -> list[Candidate]:
    out: list[Candidate] = []
    errors = 0
    for slug in wt_slug_guesses(title, artist)[:4]:
        url = WT_SONG.format(slug=slug)
        page, status = fetcher.get(url, PAGE_MAX_AGE_S)
        if page is None:
            if is_online_error(status):
                errors += 1
                break
            continue
        if 'class="chord-pro-disp"' not in page:
            continue
        m = re.search(r"<title>(.*?)\|", page, re.S)
        page_title = P._strip_tags(m.group(1)).strip() if m else title
        out.append(Candidate("worshiptogether", url, page_title, artist, url, None, [], title_similarity(title, page_title)))
        break                                                     # one page per song
    if errors and not out:
        raise SourceOffline("worshiptogether")
    return out


def fetch_worshiptogether(cand: Candidate, fetcher: Fetcher) -> ChartDoc | None:
    text, status = fetcher.get(cand.fetch_url or cand.url, PAGE_MAX_AGE_S)
    if text is None:
        if is_online_error(status):
            raise SourceOffline("worshiptogether")
        return None
    raw = P.parse_worshiptogether(text)
    if raw is None:
        return None
    chart, info = P.build_chart_text(raw, source="WorshipTogether", url=cand.url,
                                     retrieved=fetcher.cached_date(cand.fetch_url or cand.url) or today(),
                                     title=cand.title, artist=cand.artist or raw.meta.get("ministry"))
    if chart is None:
        return None
    meta = {"key_written": raw.meta.get("key"), "key_index": info["page_key"] if info["page_key"] is not None else info["key_index"],
            "meter": raw.meta.get("meter"), "tempo": info["tempo"], "year": None, "artist": cand.artist or raw.meta.get("ministry"),
            "title": cand.title, "youtube_ids": [v for v in re.findall(r"youtube\.com/embed/([A-Za-z0-9_-]{11})", text)],
            "capo": None, "n_events": info["n_events"]}
    return ChartDoc(chart, P.chart_body(chart), P.raw_to_chordpro(raw), meta)


# --------------------------------------------------------------------------------- Ultimate Guitar

def search_ultimateguitar(title: str, artist: str | None, fetcher: Fetcher) -> list[Candidate]:
    url = UG_SEARCH.format(q=urllib.parse.quote(re.sub(r"[（(\[【][^)）\]】]*[)）\]】]", " ", title).strip(), safe=""))
    page, status = fetcher.get(url, SEARCH_MAX_AGE_S)
    if page is None:
        if is_online_error(status):
            raise SourceOffline("ultimateguitar")
        return []
    pick = P.ug_pick(page, title, artist)
    if not pick or not pick.get("tab_url"):
        return []
    return [Candidate("ultimateguitar", pick["tab_url"], pick.get("song_name") or title, pick.get("artist_name") or artist,
                      pick["tab_url"], None, [], title_similarity(title, pick.get("song_name") or title))]


def fetch_ultimateguitar(cand: Candidate, fetcher: Fetcher) -> ChartDoc | None:
    text, status = fetcher.get(cand.fetch_url or cand.url, PAGE_MAX_AGE_S)
    if text is None:
        if is_online_error(status):
            raise SourceOffline("ultimateguitar")
        return None
    raw = P.parse_ultimateguitar(text)
    if raw is None:
        return None
    chart, info = P.build_chart_text(raw, source="UltimateGuitar", url=cand.url,
                                     retrieved=fetcher.cached_date(cand.fetch_url or cand.url) or today(),
                                     title=cand.title, artist=cand.artist or raw.meta.get("artist"))
    if chart is None:
        return None
    capo = raw.meta.get("capo")
    meta = {"key_written": raw.meta.get("key"), "key_index": info["page_key"] if info["page_key"] is not None else info["key_index"],
            "meter": raw.meta.get("meter"), "tempo": info["tempo"], "year": None, "artist": cand.artist or raw.meta.get("artist"),
            "title": cand.title, "youtube_ids": [], "capo": int(capo) if str(capo or "").isdigit() and int(capo) else None,
            "n_events": info["n_events"]}
    return ChartDoc(chart, P.chart_body(chart), P.raw_to_chordpro(raw), meta)


# ------------------------------------------------------------------------------------ dispatch

_FETCHERS = {"cantonhymn": fetch_cantonhymn, "simplyworship": fetch_simplyworship,
             "worshiptogether": fetch_worshiptogether, "ultimateguitar": fetch_ultimateguitar}


def fetch_chart(cand: Candidate, fetcher: Fetcher) -> ChartDoc | None:
    """The chart of one candidate page (None when the page has no readable chords)."""
    fn = _FETCHERS.get(cand.source)
    if fn is None:
        raise ValueError(f"unknown source {cand.source!r}")
    return fn(cand, fetcher)


def search_source(source: str, terms: list[str], artist: str | None, fetcher: Fetcher,
                  video_id: str | None = None) -> list[Candidate]:
    """Candidates of one source for the ranked terms (the first term that finds anything wins)."""
    for term in terms:
        if source == "cantonhymn":
            found = search_cantonhymn(term, fetcher, video_id)
        elif source == "simplyworship":
            found = search_simplyworship(term, fetcher)
        elif source == "worshiptogether":
            found = search_worshiptogether(term, artist, fetcher)
        elif source == "ultimateguitar":
            found = search_ultimateguitar(term, artist, fetcher)
        else:
            raise ValueError(f"unknown source {source!r}")
        if found:
            return found[:MAX_PER_SOURCE]
    return []


def source_order(language: str, sources: tuple[str, ...] | None = None, terms: list[str] | None = None) -> tuple[str, ...]:
    """Sources to try, in order: the language's own sites first; the English-only sites
    (WorshipTogether, Ultimate Guitar) only when the song has an English title or alias."""
    if sources:
        return sources
    first = ORDER.get(language, ORDER["en"])
    rest = tuple(s for s in SOURCES if s not in first)
    if language == "cjk" and terms is not None and not any(re.search(r"[A-Za-z]{2,}", t) for t in terms):
        rest = tuple(s for s in rest if s not in ("worshiptogether", "ultimateguitar"))
    return first + rest


def candidate_from_url(url: str) -> Candidate | None:
    """A pasted sheet LINK from a supported site → a candidate (None for other sites)."""
    u = (url or "").strip()
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u
    host = (urllib.parse.urlparse(u).hostname or "").lower()
    path = urllib.parse.urlparse(u).path
    if host.endswith("cantonhymn.net") and "/song/" in path:
        parts = [p for p in path.split("/song/", 1)[1].split("/") if p]
        if not parts:
            return None
        slug = urllib.parse.quote(urllib.parse.unquote(parts[-1]), safe="").lower()
        parent = urllib.parse.quote(urllib.parse.unquote(parts[0]), safe="").lower() if len(parts) > 1 else None
        title = urllib.parse.unquote(parts[-1])
        return Candidate("cantonhymn", u.split("?")[0], title, None, CH_DETAIL.format(slug=slug, kind="child" if parent else "parent"),
                         None, [], 1.0, slug, parent, True)
    if host.endswith("simplyworship.hk") and "/lyrics/" in path:
        title = urllib.parse.unquote(path.rstrip("/").split("/")[-1])
        return Candidate("simplyworship", u.split("?")[0], title, None, u.split("?")[0], None, [], 1.0)
    if host.endswith("worshiptogether.com") and "/songs/" in path:
        slug = path.rstrip("/").split("/")[-1]
        return Candidate("worshiptogether", u.split("?")[0], slug.replace("-", " ").title(), None, u.split("?")[0], None, [], 1.0)
    if host.endswith("ultimate-guitar.com") and "/tab/" in path:
        return Candidate("ultimateguitar", u.split("?")[0], path.rstrip("/").split("/")[-1].replace("-", " "), None,
                         u.split("?")[0], None, [], 1.0)
    return None


def chord_signature(body: str, bpb: int = 4) -> tuple[str, ...]:
    """The chord sequence of a bar-text chart (repeats merged) — two charts with the same signature
    are the same chords (the harvester's dedupe rule)."""
    from ..theory import chart as CH
    try:
        chart = CH.parse(body, bpb)
    except CH.ChartError:
        return tuple(re.findall(r"[A-G][#b]?[^\s|]*", body))
    return tuple(h for h, _ in P.chord_events(chart))
