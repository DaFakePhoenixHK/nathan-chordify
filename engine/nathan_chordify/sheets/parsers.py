"""Chord-chart pages -> bar-text charts (chords + section labels only; lyrics are read in memory
to estimate chord lengths and never written).

Sources (all readable without an account):
  - CantonHymn.net song-detail JSON ("[C]" inline chords; original key, year, YouTube ids),
  - Simply Worship (simplyworship.hk) lyric pages (inline chords; key, time signature, tempo),
  - WorshipTogether.com song pages (chords over lyrics, original key, BPM),
  - Ultimate Guitar public "Chords" pages (the most-voted public version).

How bars are inferred when a chart has chords over lyrics but no bar lines:
  1. Syllables are counted from each line's first chord to its end (a Chinese character = 1,
     English words by vowel groups, a pause such as "，" or a wide space = 1, the held last note +
     breath = 3). Words before the first chord are a pickup and ignored; a sung line without chords
     makes the line before it longer (its last chord goes on).
  2. rate = the song's typical syllables per chord ("chords per line ~ bars"), blended with a
     tempo guess when the page gives a BPM (~80/BPM syllables per beat), kept between 4.5
     (Chinese) / 3.5 (English) and 8 syllables per 4/4 bar. A line lasts syllables / rate bars,
     rounded to 1, 2, 4, 6 or 8 (at least one bar per two chords).
  3. Inside the line each chord goes to the bar line nearest to where its syllable falls, in
     order, one chord per position; only a line with more chords than bars also uses half bars.
  4. Chord-only lines (intros, interludes) use the chart's own rhythm when given ("C // G //":
     one slash = one beat; "| F | C G |" bar lines); otherwise one bar per chord.
These durations are approximate; the chord ORDER is what the source says.

Moved here from train/harvest_charts.py (2026-11-03); the harvester imports these names.
"""
from __future__ import annotations

import html
import json
import math
import re
import statistics
import unicodedata
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field

from ..theory import chart as CH
from ..theory import harte as H
from ..theory import key as K

LINE_END = 3.0               # syllables added at a line end (the held last note + breath)
PAUSE = 1.0                  # syllables added at a pause inside a line
NICE_BARS = (1, 2, 4, 6, 8)  # how many bars a sung line may last
RATE_MIN = {True: 1.125, False: 0.875}   # syllables per beat at least (Chinese, English)
RATE_MAX = 2.0                           # syllables per beat at most
TEMPO_SYLLABLES = 80.0                   # syllables per beat ~ 80 / BPM (tempo guess)


# =============================================================================================
# Chart text from each source: a list of lines, each a list of (chord marker, text) segments
# =============================================================================================

@dataclass
class Seg:
    chord: str | None   # raw marker text (may hold several tokens, e.g. "C // G //")
    text: str           # lyric / rhythm text that follows it


Line = list[Seg]


@dataclass
class Raw:
    lines: list[Line]
    meta: dict = field(default_factory=dict)    # key, tempo, meter, capo, artist, title (as the page says)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).replace("\xa0", " ")


def chordpro_line(raw: str) -> Line:
    """'常[F]聽說[Am]久' -> [Seg(None,'常'), Seg('F','聽說'), Seg('Am','久')]."""
    segs: Line = []
    pos = 0
    for m in re.finditer(r"\[([^\]\[]*)\]", raw):
        pre = raw[pos:m.start()]
        if segs:
            segs[-1].text += pre
        elif pre:
            segs.append(Seg(None, pre))
        segs.append(Seg(m.group(1), ""))
        pos = m.end()
    tail = raw[pos:]
    if segs:
        segs[-1].text += tail
    elif tail:
        segs.append(Seg(None, tail))
    return segs


def parse_worshiptogether(page: str) -> Raw | None:
    i = page.find('class="chord-pro-disp"')
    if i < 0:
        return None
    body = page[i:]
    end = body.find("</section>")
    body = body[:end] if end > 0 else body
    lines: list[Line] = []
    for chunk in body.split('<div class="chord-pro-line">')[1:]:
        segs: Line = []
        for m in re.finditer(r'<div class="chord-pro-(note|lyric)">(.*?)</div>', chunk, re.S):
            txt = _strip_tags(m.group(2))
            if m.group(1) == "note":
                segs.append(Seg(txt.strip() or None, ""))
            elif segs:
                segs[-1].text += txt
            else:
                segs.append(Seg(None, txt))
        lines.append(segs)
    meta = {}
    for m in re.finditer(r"<span>\s*([A-Za-z #()]+?)\s*:\s*</span>\s*<p>(.*?)</p>", page, re.S):
        meta[m.group(1).strip().lower()] = _strip_tags(m.group(2)).strip()
    out = {"key": meta.get("original key"), "ministry": meta.get("ministry"), "ccli": meta.get("ccli #")}
    bpm = meta.get("bpm")
    if bpm and re.fullmatch(r"\d+(\.\d+)?", bpm):
        out["tempo"] = float(bpm)
    for k in ("time signature", "meter", "time"):
        if meta.get(k) and re.fullmatch(r"\d+/\d+", meta[k]):
            out["meter"] = meta[k]
    return Raw(lines, out)


def parse_cantonhymn(payload: str) -> Raw | None:
    try:
        d = json.loads(payload)
        song = d["data"]["currentSong"]
    except (ValueError, KeyError, TypeError):
        return None
    if not song or not song.get("lyrics"):
        return None
    lines = [chordpro_line(ln) for ln in song["lyrics"].replace("\r", "").split("\n")]
    meta = {"title": song.get("title")}
    key = song.get("songkey") or {}
    if isinstance(key, dict) and key.get("label"):
        meta["key"] = key["label"]
    years = song.get("songyear_of_publication") or []
    if years:
        meta["year"] = years[0].get("label")
    langs = song.get("songlanguage") or []
    if langs:
        meta["language_page"] = langs[0].get("label")
    owners = song.get("owner") or []
    meta["artist"] = ", ".join(o.get("label", "") for o in owners) or song.get("source")
    return Raw(lines, meta)


def _span_end(s: str, start: int) -> int:
    """Index just after the </span> closing the <span ...> that starts at ``start``."""
    depth = 0
    for m in re.finditer(r"<(/?)span\b[^>]*>", s[start:]):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return start + m.end()
    return len(s)


def parse_simplyworship(page: str) -> Raw | None:
    i = page.find("<div class=KCN-content>")
    if i < 0:
        i = page.find('class="KCN-content"')
    if i < 0:
        return None
    j = page.find("KCN-content", i + 30)
    k = page.find("</article>", i)
    stop = min(x for x in (j, k, len(page)) if x > 0)
    body = page[i:stop]
    meta: dict = {}
    m = re.search(r"class=KCNkey>([^<]*)</span>\s*<span class=KCNkeyQuality>([^<]*)<", body)
    if m:
        meta["key"] = (m.group(1) + m.group(2)).replace("♭", "b").replace("♯", "#").strip()
    # replace every chord wrapper with [chord]
    out, pos = [], 0
    for m in re.finditer(r"<span class=KCNchordWrap", body):
        if m.start() < pos:
            continue
        e = _span_end(body, m.start())
        out.append(body[pos:m.start()])
        out.append("[" + re.sub(r"\s+", "", _strip_tags(body[m.start():e])) + "]")
        pos = e
    out.append(body[pos:])
    body = "".join(out)
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<h3.*?</h3>", "", body, flags=re.S)
    body = re.sub(r"<div class=KCNtranspose>.*?</div>\s*<button[^>]*>[^<]*</button>\s*</div>", "", body, flags=re.S)
    body = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>", "\n", body)
    lines: list[Line] = []
    for raw in _strip_tags(body).split("\n"):
        mm = re.fullmatch(r"\s*(廣東話|粵語|國語|普通話|華語|English|英文)?\s*(\d{1,2}/\d{1,2})?\s*"
                          r"(?:Tempo\s*=?\s*(\d+))?\s*", raw)
        if mm and "meter" not in meta and "tempo" not in meta and (mm.group(2) or mm.group(3)):
            if mm.group(2):
                meta["meter"] = mm.group(2)
            if mm.group(3):
                meta["tempo"] = float(mm.group(3))
            if mm.group(1):
                meta["language_page"] = mm.group(1)
            continue
        if mm or "Show/Hide Chords" in raw or "Transpose" in raw:
            continue
        lines.append(chordpro_line(raw))
    am = re.search(r"<h3><a href=[^>]*/artist/[^>]*>(.*?)</a></h3>", page)
    if am:
        meta["artist"] = _strip_tags(am.group(1)).strip()
    return Raw(lines, meta)


def _ug_store(page: str) -> dict | None:
    m = re.search(r'class="js-store" data-content="(.*?)"', page, re.S)
    if not m:
        return None
    try:
        return json.loads(html.unescape(m.group(1)))
    except ValueError:
        return None


def ug_pick(page: str, title: str, artist: str | None) -> dict | None:
    """Best public 'Chords' result for the title in a UG search page (votes x rating, artist match)."""
    store = _ug_store(page)
    try:
        results = store["store"]["page"]["data"]["results"]
    except (TypeError, KeyError):
        return None
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()  # noqa: E731
    t = norm(re.sub(r"\(.*?\)", "", title))
    artist_words = set(norm(artist).split()) - {"the", "and", "music", "worship"} if artist else set()
    best, best_score = None, -1.0
    for r in results:
        if r.get("type") != "Chords" or r.get("tab_access_type") != "public":
            continue
        name = norm(re.sub(r"\(.*?\)", "", r.get("song_name", "")))
        if name != t and not name.startswith(t) and not t.startswith(name):
            continue
        score = (r.get("votes") or 0) * (r.get("rating") or 0)
        if artist_words & set(norm(r.get("artist_name")).split()):
            score = score * 3 + 1
        if score > best_score:
            best, best_score = r, score
    return best


_FINGERING = re.compile(r"(?:^|\s)[x0-9]{6}(?:\s|$)|\bcapo\b|^\s*[eBGDAE]\|", re.I)


def parse_ultimateguitar(page: str) -> Raw | None:
    store = _ug_store(page)
    try:
        pd = store["store"]["page"]["data"]
        content = pd["tab_view"]["wiki_tab"]["content"]
    except (TypeError, KeyError):
        return None
    tab = pd.get("tab") or {}
    meta = {"key": tab.get("tonality_name") or None, "capo": tab.get("capo"),
            "artist": tab.get("artist_name"), "title": tab.get("song_name")}
    text = content.replace("\r", "").replace("[tab]", "").replace("[/tab]", "")
    rows = text.split("\n")
    lines: list[Line] = []

    def chord_cols(row: str) -> list[tuple[int, str]] | None:
        """Chords of a chord row with their columns, or None if the row is not a chord row."""
        cols, plain, pos = [], "", 0
        for m in re.finditer(r"\[ch\](.*?)\[/ch\]", row):
            plain += row[pos:m.start()]
            cols.append((len(plain), m.group(1)))
            plain += m.group(1)
            pos = m.end()
        plain += row[pos:]
        rest = re.sub(r"\[ch\].*?\[/ch\]", " ", row)
        if not cols or re.search(r"[A-Za-z一-鿿]", re.sub(r"(?i)\b(x\d+|\d+x|n\.?c\.?)\b", "", rest)):
            return None
        return cols

    i = 0
    while i < len(rows):
        row = rows[i]
        if _FINGERING.search(re.sub(r"\[/?ch\]", "", row)):
            i += 1
            continue
        cols = chord_cols(row)
        if cols is None:
            if "[ch]" not in row:
                lines.append([Seg(None, row)])
            i += 1
            continue
        nxt = rows[i + 1] if i + 1 < len(rows) else ""
        lyric_next = (nxt.strip() and "[ch]" not in nxt and not _FINGERING.search(nxt)
                      and not re.fullmatch(r"\s*\[[^\]]+\]\s*", nxt))
        if lyric_next:
            lyric = nxt.ljust(max(c for c, _ in cols) + 1)
            segs: Line = []
            if cols[0][0] > 0:
                segs.append(Seg(None, lyric[:cols[0][0]]))
            for n, (c, ch) in enumerate(cols):
                end = cols[n + 1][0] if n + 1 < len(cols) else len(lyric)
                segs.append(Seg(ch, lyric[c:end]))
            lines.append(segs)
            i += 2
        else:   # chord-only row: keep the text between chords (rhythm slashes, bar lines)
            segs = [Seg(None, "")]
            pos = 0
            for m in re.finditer(r"\[ch\](.*?)\[/ch\]", row):
                segs[-1].text += row[pos:m.start()]
                segs.append(Seg(m.group(1), ""))
                pos = m.end()
            segs[-1].text += row[pos:]
            lines.append(segs)
            i += 1
    return Raw(lines, meta)


# =============================================================================================
# Chord symbols: make every chord one the repo's parser accepts
# =============================================================================================

_RHYTHM = re.compile(r"^(?:/+|\\+|\|+|:?\|\|?:?|\.+|-+|~+|～+|\*+|%|"
                     r"[xX×*]\s*\d+|\d+\s*[xX×]|\(\s*[xX×]?\s*\d+\s*[xX×]?\s*\))$")
_SYM = re.compile(r"^(?P<root>[A-Ga-g])(?P<acc>##|bb|[#b])?(?P<q>[^/]*?)(?:/(?P<bass>[A-Ga-g](?:##|bb|[#b])?))?$")
_Q_FALLBACK = [    # (regex on the quality, replacement): only used when the parser rejects a quality
    (r"^(?:maj|M|Maj)$", ""), (r"^add4$|^add11$|^\(add4\)$|^\(add11\)$", ""),
    (r"^m(?:add4|add11|\(add4\)|\(add11\))$", "m"), (r"^(?:maj13|M13|maj7\(13\)|maj9\(13\))$", "maj9"),
    (r"^(?:m13|min13|m7\(13\))$", "m9"), (r"^(?:m7add4|m7\(add4\)|m7sus4|m7\(11\)|m7/11)$", "m11"),
    (r"^(?:sus2sus4|sus24|sus42|sus2\(4\)|2sus4|4sus2|sus\(2\)|2\(4\))$", "sus4"),
    (r"^(?:4|sus\(4\))$", "sus4"), (r"^(?:7sus2|7\(sus2\))$", "7sus4"), (r"^(?:6sus4|6sus|6\(sus4\))$", "sus4"),
    (r"^(?:m2|m\(2\))$", "madd9"), (r"^(?:add9\(no3\)|5add9|5\(add9\)|2\(no3\)|sus2\(no3\))$", "sus2"),
    (r"^(?:m6/9|m69|m6\(9\))$", "m6"), (r"^(?:maj7sus2|maj7\(sus2\)|maj9sus|maj7sus4)$", "maj7"),
    (r"^(?:2/4|24)$", "sus4"), (r"^(?:m\+5|m#5|m\(#5\))$", "m"), (r"^(?:dim\(maj7\)|°maj7)$", "dim"),
    (r"^(?:9b5|7b5|7\(b5\)|7-5)$", "7"), (r"^(?:9\(b5\)|9#11|9\(#11\)|7#11|7\(#11\))$", "9"),
    (r"^(?:add2add4|add9add11)$", "add9"), (r"^(?:maj7add13|maj7\(add13\)|maj7\(6\))$", "maj7"),
    (r"^(?:7add13|7\(add13\)|7\(6\))$", "7"),
]


_Q_VOCAB = re.compile(r"(?:maj|min|mi|ma|sus|add|dim|aug|no|omit|alt|dom|m|M|j|[0-9#b+°ø()\-/,.Δ])*")


def _clean_token(tok: str) -> str:
    s = unicodedata.normalize("NFKC", tok).strip().strip(",;")
    s = s.replace("♭", "b").replace("♯", "#").replace("–", "-").replace("−", "-").replace("Δ", "maj7")
    while len(s) > 2 and s[0] == "(" and s[-1] == ")" and s.count("(") == 1:
        s = s[1:-1].strip()
    if s.startswith("(") and s.count("(") > s.count(")"):
        s = s[1:]
    if s.endswith(")") and s.count(")") > s.count("("):
        s = s[:-1]
    return s.strip()


def norm_chord(tok: str) -> tuple[str | None, str | None]:
    """A raw chord token -> (symbol the parser accepts, simplification note) or (None, None).

    "D2/F#" -> "Dadd9/F#" (a "2" chord over its own third keeps the third), "Asus" -> "Asus4",
    "Cmaj13" -> ("Cmaj9", "Cmaj13 -> Cmaj9"); "N.C." -> "N"."""
    s = _clean_token(tok)
    if not s:
        return None, None
    if s.upper().replace(" ", "") in ("N.C.", "NC", "N.C", "N/C", "NOCHORD"):
        return "N", None
    s = re.sub(r"(?:sus)?2/4$", "sus4", s)      # "Dsus2/4", "D2/4": both suspensions -> sus4
    m = _SYM.match(s)
    if not m:
        return None, None
    root = m["root"].upper() + (m["acc"] or "")
    q = (m["q"] or "").strip()
    bass = m["bass"]
    if bass:
        bass = bass[0].upper() + bass[1:]
    if len(q) > 12 or not _Q_VOCAB.fullmatch(q.replace(" ", "")):
        return None, None             # "Chorus" is not C + "horus", "Bridge" is not B + "ridge"
    if q == "2" and bass:
        rp, _, bp = H.parse_symbol(root + "/" + bass)
        if (bp - rp) % 12 == 4:     # "2" over its own major third: an add9 chord
            q = "add9"

    def attempt(qq: str) -> str | None:
        try:
            harte = H.symbol_to_harte(root + qq + ("/" + bass if bass else ""))
        except ValueError:
            return None
        return H.harte_to_symbol(harte)

    sym = attempt(q)
    if sym is not None:
        return sym, None
    q2 = q.replace(" ", "")
    for pat, rep in _Q_FALLBACK:
        if re.match(pat, q2):
            sym = attempt(rep)
            if sym is not None:
                return sym, f"{s} -> {sym}"
    # last resort: drop a bracketed extension, then shorten the quality until it parses
    q3 = re.sub(r"\(.*?\)", "", q2)
    for n in range(len(q3), -1, -1):
        sym = attempt(q3[:n])
        if sym is not None:
            return sym, f"{s} -> {sym}"
    return None, None


# =============================================================================================
# Lines -> sections -> beats -> bars
# =============================================================================================

_LABELS = [
    (r"intro(?:duction)?|前奏|序|引子|序曲", "Intro"),
    (r"pre[- ]?chorus|prechorus|pre|pc|前副歌|導歌|导歌", "Pre-Chorus"),
    (r"post[- ]?chorus", "Post-Chorus"),
    (r"chorus|refrain|ch|副歌|b段|副", "Chorus"),
    (r"verse|v|主歌|a段|主", "Verse"),
    (r"bridge|br|橋段|桥段|c段|橋", "Bridge"),
    (r"interlude|instrumental|inst|break|solo|turnaround|turn|間奏|间奏|過門|过门|間|奏", "Interlude"),
    (r"tag|vamp|hook|channel", "Tag"),
    (r"outro|ending|end|coda|尾聲|尾声|結尾|结尾|結束|结束|收尾|尾", "Outro"),
]
_LABEL_RE = re.compile(
    r"^[\[(（【]?\s*(?P<name>" + "|".join(p for p, _ in _LABELS) + r")\s*(?P<num>\d+|[ivx]+)?\s*"
    r"(?:[\])）】])?\s*(?:[(（]?\s*(?:[xX×*]\s*\d+|\d+\s*[xX×])\s*[)）]?)?\s*[:：.\-]?\s*$", re.I)
_REPEAT = re.compile(r"[(（]?\s*(?:[xX×*]\s*\d+|\d+\s*[xX×]|repeat|重唱|重複|重复|反覆|反复)\s*[)）]?", re.I)
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]")


def match_label(text: str) -> str | None:
    t = text.strip()
    if not t or len(t) > 30:
        return None
    m = _LABEL_RE.match(t)
    if not m:
        return None
    name = m["name"].lower()
    canon = next(c for p, c in _LABELS if re.fullmatch(p, name, re.I))
    num = m["num"]
    if num and not num.isdigit():
        num = str({"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}.get(num.lower(), "")) or None
    return f"{canon} {num}" if num else canon


def eng_syllables(word: str) -> int:
    w = word.lower().strip("'")
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if n > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        n -= 1
    if n > 1 and w.endswith("ed") and not w.endswith(("ted", "ded")):
        n -= 1
    return max(1, n)


def syllables(text: str) -> float:
    """Sung syllables in a lyric fragment (+ pauses), used only to estimate chord lengths."""
    t = _REPEAT.sub(" ", text)
    n = float(len(_CJK.findall(t)) + t.count("～") + t.count("~"))
    latin = _CJK.sub(" ", t)
    n += sum(eng_syllables(w) for w in re.findall(r"[A-Za-z']+", latin))
    if _CJK.search(t):
        n += PAUSE * len(re.findall(r"[，。、；！？　]|(?<=\S)\s{2,}(?=\S)|(?<=[一-鿿]) (?=[一-鿿])", t))
    else:
        n += 0.5 * len(re.findall(r"[,;!?]", latin))
    return n


@dataclass
class Tok:
    kind: str           # "chord" | "text"
    value: str          # symbol or text


def expand(line: Line, notes: Counter) -> list[Tok]:
    """Segments -> a flat token list; chord markers holding several tokens ("C // G //") are split,
    marker tokens that are not chords become text (e.g. a "[Chorus]" written in brackets)."""
    out: list[Tok] = []
    for seg in line:
        if seg.chord is not None:
            for t in re.split(r"(\s+|/{2,}|(?<=\s)/(?=\s)|\|)", seg.chord):
                if t is None or not t:
                    continue
                sym, note = norm_chord(t) if t.strip() and not _RHYTHM.match(t.strip()) else (None, None)
                if sym is not None:
                    out.append(Tok("chord", sym))
                    if note:
                        notes[note] += 1
                else:
                    out.append(Tok("text", t if not t.strip() or _RHYTHM.match(t.strip()) else f" {t} "))
        if seg.text:
            out.append(Tok("text", seg.text))
    return out


def _texts(toks: list[Tok]) -> str:
    return "".join(t.value for t in toks if t.kind == "text")


def is_instrumental(toks: list[Tok]) -> bool:
    if not any(t.kind == "chord" for t in toks):
        return False
    txt = _REPEAT.sub(" ", _texts(toks))
    txt = re.sub(r"[/\\|.\-~～:*%()（）\[\]{}，,]", " ", txt)
    return syllables(txt) == 0


@dataclass
class Section:
    label: str | None
    lines: list[list[Tok]]


def sectionize(lines: list[Line], notes: Counter) -> list[Section]:
    sections: list[Section] = []
    cur = Section(None, [])
    for line in lines:
        toks = expand(line, notes)
        first = next((i for i, t in enumerate(toks) if t.kind == "chord"), len(toks))
        lead = _texts(toks[:first])
        label = match_label(lead) if lead.strip() else None
        if label is None and first == len(toks) and not syllables(lead):
            continue                              # blank or decoration line
        if label is not None:
            if cur.lines:
                sections.append(cur)
            cur = Section(label, [])
            toks = toks[first:]
            if not toks:
                continue
        elif first == len(toks) and re.fullmatch(r"\s*" + _REPEAT.pattern + r"\s*", lead, re.I):
            continue
        cur.lines.append(toks)
    if cur.lines:
        sections.append(cur)
    return [s for s in sections if any(t.kind == "chord" for ln in s.lines for t in ln)]


def _split_bar(n: int, m: int) -> list[int]:
    """Beats for n chords sharing one m-beat bar ("| F C |" -> [2, 2]; 3 in 4/4 -> [2, 1, 1])."""
    if n <= 1:
        return [m]
    if n >= m:
        return [1] * n
    if n == 2:
        return [m - m // 2, m // 2]
    return [m - (n - 1)] + [1] * (n - 1)


def _rhythm_seq(toks: list[Tok]) -> list[tuple[str, str]]:
    """A chord-only line -> [("c", symbol) | ("/", "") | ("|", "")]."""
    seq: list[tuple[str, str]] = []
    for t in toks:
        if t.kind == "chord":
            seq.append(("c", t.value))
        else:
            for mark in re.findall(r"\|+|/|\\", t.value):
                seq.append(("|", "") if mark.startswith("|") else ("/", ""))
    return seq


def _slash_counts(seq: list[tuple[str, str]]) -> list[list]:
    counts: list[list] = []
    for k, v in seq:
        if k == "c":
            counts.append([v, 0])
        elif k == "/" and counts:
            counts[-1][1] += 1
    return counts


def _bar_groups(seq: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    groups, cur = [], []
    for k, v in seq + [("|", "")]:
        if k == "|":
            if any(kk == "c" for kk, _ in cur):
                groups.append(cur)
            cur = []
        else:
            cur.append((k, v))
    return groups


def instrumental_items(toks: list[Tok], m: int) -> list[tuple[str, int]]:
    """A chord-only line -> [(symbol, beats)] using its own rhythm marks when present:
    bar lines ("| F | C G |": chords share their bar; "| Bm / / A |": chord + slashes = beats),
    or slashes alone ("C // G //": a slash is a beat; "G / / / C / / /": the chord is beat 1)."""
    seq = _rhythm_seq(toks)
    if any(k == "|" for k, _ in seq):
        out = []
        for g in _bar_groups(seq):
            counts = _slash_counts(g)
            with_chord = [n + 1 for _, n in counts]
            only = [n for _, n in counts]
            if sum(with_chord) == m:
                out += [(c, b) for (c, _), b in zip(counts, with_chord)]
            elif sum(only) == m and all(only):
                out += [(c, b) for (c, _), b in zip(counts, only)]
            else:
                out += list(zip([c for c, _ in counts], _split_bar(len(counts), m)))
        return out
    counts = _slash_counts(seq)
    if any(n for _, n in counts):
        total = sum(n for _, n in counts)
        chord_counts = any(n == 0 for _, n in counts) or (total % m and (total + len(counts)) % m == 0)
        return [(c, n + 1 if chord_counts else n) for c, n in counts]
    return [(c, m) for c, _ in counts]


@dataclass
class Block:
    """One line of a section: a sung line (chords at syllable offsets) or a chord-only line."""
    kind: str                                  # "lyric" | "fixed"
    chords: list[tuple[str, float]]            # lyric: (symbol, syllables from the line's first chord)
                                               # fixed: (symbol, beats)
    span: float = 0.0                          # lyric: syllables from the first chord to the line end (+ holds)
    own: float = 0.0                           # lyric: the same, without the chordless lines that follow


def section_blocks(sec: Section, m: int) -> list[Block]:
    blocks: list[Block] = []
    for toks in sec.lines:
        if is_instrumental(toks):
            blocks.append(Block("fixed", instrumental_items(toks, m)))
            continue
        if not any(t.kind == "chord" for t in toks):
            # a sung line without chords: the previous line's last chord simply goes on longer
            if blocks and blocks[-1].kind == "lyric":
                blocks[-1].span += syllables(_texts(toks)) + LINE_END
            continue
        chords: list[tuple[str, float]] = []
        s = 0.0
        for t in toks:
            if t.kind == "chord":
                chords.append((t.value, s))
            elif chords:                       # text before the first chord is a pickup: ignored
                s += syllables(t.value)
        blocks.append(Block("lyric", chords, s + LINE_END, s + LINE_END))
    return blocks


def song_rate(blocks: list[Block], m: int, cjk: bool, tempo: float | None = None) -> float:
    """Syllables per bar: the song's typical syllables per chord ("chords per line ~ bars"),
    blended (geometric mean) with a tempo guess when the page gives a BPM (about 80/BPM syllables
    per beat: slow songs fit more syllables in a beat), kept between RATE_MIN (Chinese / English)
    and RATE_MAX syllables per beat."""
    r = [b.own / len(b.chords) for b in blocks if b.kind == "lyric" and b.chords]
    lo = RATE_MIN[cjk] * m
    rate = statistics.median(r) if r else lo
    if tempo and tempo > 0:
        rate = math.sqrt(rate * m * min(max(TEMPO_SYLLABLES / tempo, 0.5), 1.6))
    return min(max(rate, lo), RATE_MAX * m)


def line_bars(span: float, n_chords: int, rate: float) -> int:
    x = span / rate
    bars = min(NICE_BARS, key=lambda n: (abs(n - x), n))
    return max(bars, math.ceil(n_chords / 2))


def place_block(b: Block, m: int, rate: float) -> tuple[list[tuple[int, str]], int]:
    """A block -> (chord onsets in beats from the block start, block length in beats)."""
    if b.kind == "fixed":
        out, pos = [], 0
        for sym, beats in b.chords:
            out.append((pos, sym))
            pos += beats
        return out, pos + (-pos) % m
    half = m // 2 if m % 2 == 0 else (m + 1) // 2
    n = len(b.chords)
    bars = line_bars(b.span, n, rate)
    total = bars * m
    # no more chords than bars: chords change on bar lines; otherwise half bars are allowed too
    offs = (0,) if n <= bars or half >= m else (0, half)
    slots = [bar * m + off for bar in range(bars) for off in offs]
    out = [(0, b.chords[0][0])]
    k = 0                                       # index of the slot used by the previous chord
    for i, (sym, off) in enumerate(b.chords[1:], 1):
        target = off / b.span * total if b.span > 0 else slots[min(k + 1, len(slots) - 1)]
        cands = range(k + 1, len(slots) - (n - 1 - i))
        # nearest slot; a bar line is preferred over a half bar by a quarter beat
        k = min(cands, key=lambda j: abs(slots[j] - target) + (0.25 if slots[j] % m else 0.0))
        out.append((slots[k], sym))
    return out, total


def bars_text(onsets: list[tuple[int, str]], end: int, m: int) -> list[str]:
    """Chord onsets (beats) -> one bar-text string per bar; the chord sounding on beat 1 is restated."""
    bars: list[str] = []
    cur = None
    k = 0
    for bar in range(end // m):
        lo, hi = bar * m, (bar + 1) * m
        inbar = []
        while k < len(onsets) and onsets[k][0] < hi:
            inbar.append(onsets[k])
            k += 1
        if not inbar or (len(inbar) == 1 and inbar[0][0] == lo):
            cur = inbar[0][1] if inbar else cur
            bars.append(cur)
            continue
        toks = ["."] * m
        toks[0] = cur if cur else inbar[0][1]
        for p, s in inbar:
            toks[p - lo] = s
            cur = s
        bars.append(" ".join(toks))
    return bars


def to_bar_text(sections: list[Section], m: int, tempo: float | None = None) -> str:
    """Sections -> bar text (see the module docstring, "How bars are inferred")."""
    blocks = [section_blocks(s, m) for s in sections]
    cjk = sum(len(_CJK.findall(_texts(t))) for s in sections for t in s.lines) >         sum(len(re.findall(r"[A-Za-z]+", _texts(t))) for s in sections for t in s.lines)
    rate = song_rate([b for bl in blocks for b in bl], m, cjk, tempo)
    out_lines: list[str] = []
    for sec, bl in zip(sections, blocks):
        onsets, pos = [], 0
        for b in bl:
            ons, length = place_block(b, m, rate)
            onsets += [(pos + p, s) for p, s in ons]
            pos += length
        if not onsets:
            continue
        bars = bars_text(onsets, pos, m)
        if sec.label:
            out_lines.append(f"[{sec.label}]")
        for i in range(0, len(bars), 4):
            out_lines.append("| " + " | ".join(bars[i:i + 4]) + " |")
    return "\n".join(out_lines) + "\n"


# =============================================================================================
# Key from chords, header, files
# =============================================================================================

_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_FLAT_MAJOR = {1, 3, 5, 8, 10}   # Db Eb F Ab Bb


def key_name(k: int) -> str:
    pc = k % 12
    if k >= 12:
        rel = (pc + 3) % 12
        return (_NAMES_FLAT if rel in _FLAT_MAJOR or pc in (0, 2, 5, 7) else _NAMES_SHARP)[pc] + "m"
    return (_NAMES_FLAT if pc in _FLAT_MAJOR else _NAMES_SHARP)[pc]


def parse_key(text: str | None) -> int | None:
    if not text:
        return None
    m = re.match(r"\s*([A-Ga-g])([#b♭♯]?)\s*(m(?!aj)|min|minor)?", text)
    if not m:
        return None
    pc = H.parse_symbol(m.group(1).upper() + m.group(2).replace("♭", "b").replace("♯", "#"))[0]
    return pc + (12 if m.group(3) else 0)


def chord_events(chart: CH.Chart) -> list[tuple[str, float]]:
    """(harte, beats) for each chord change, consecutive repeats merged."""
    m = chart.beats_per_bar
    out: list[list] = []
    for sec in chart.sections:
        for bar in sec.bars:
            if not bar:
                if out:
                    out[-1][1] += m
                continue
            pos = [c.pos if c.pos is not None else i * m / len(bar) for i, c in enumerate(bar)]
            for i, c in enumerate(bar):
                d = (pos[i + 1] if i + 1 < len(bar) else m) - pos[i]
                if out and out[-1][0] == c.harte:
                    out[-1][1] += d
                else:
                    out.append([c.harte, d])
    return [(h, d) for h, d in out]


_MINORISH = frozenset({"min", "m7", "m9", "m11", "madd9", "m6", "mM7"})


def estimate_key(events: list[tuple[str, float]], page_key: int | None = None) -> tuple[int, int]:
    """(key 0..23 used for the chart, best key from the chords alone): duration-weighted
    log P(chord | key) + a bonus when the song starts / ends on the tonic. The page key wins
    when it is (almost) as good as the best."""
    trip = [(H.parse_harte(h), d) for h, d in events if h != "N"]
    if not trip:
        return (page_key if page_key is not None else 0), 0
    score = []
    for k in range(24):
        s = sum(d * K.chord_loglik(r, q, k) for (r, q, b), d in trip)
        for idx, bonus in ((0, 2.0), (-1, 4.0)):
            (r, q, b), _ = trip[idx]
            if r == k % 12 and ((k >= 12) == (q in _MINORISH) or q in ("sus2", "sus4", "5")):
                s += bonus
        s -= 2.0 if k >= 12 else 0.0       # worship songs are mostly major
        score.append(s)
    best = max(range(24), key=score.__getitem__)
    total = sum(d for _, d in trip)
    if page_key is not None and score[best] - score[page_key] <= 0.05 * total:
        return page_key, best
    return best, best


def beats_for_meter(meter: str) -> int:
    num, den = (int(x) for x in meter.split("/"))
    if den == 8 and num % 3 == 0:
        return num // 3 if num >= 9 else num      # 12/8 -> 4 dotted-quarter beats; 6/8 -> 6 eighths
    return num


def slash_meter(raw: Raw) -> str | None:
    """The meter written into the chart's own bar-lined rhythm lines ("| D / / / | Bm / / A |" -> 4/4,
    "| G / / | C / / |" -> 3/4): beats per bar = chords + slashes in that bar."""
    totals = []
    notes: Counter = Counter()
    for line in raw.lines:
        toks = expand(line, notes)
        if not is_instrumental(toks) or "//" in _texts(toks):
            continue        # "F2 // | C/E //" is shorthand, not one slash per beat
        seq = _rhythm_seq(toks)
        if not any(k == "|" for k, _ in seq) or not any(k == "/" for k, _ in seq):
            continue
        for g in _bar_groups(seq):
            counts = _slash_counts(g)
            if any(n for _, n in counts):
                totals.append(sum(n + 1 for _, n in counts))
    if len(totals) >= 3:
        c = Counter(totals).most_common(1)[0][0]
        return {3: "3/4", 4: "4/4", 6: "6/8"}.get(c)
    return None


def clean_header(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip() if s is not None else ""


# =============================================================================================
# Chart text (bar text + header), CantonHymn API, validation
# =============================================================================================

def ch_api(url: str, hierarchy: str = "parent") -> str:
    """CantonHymn song page URL -> its song-detail JSON URL (slug = the last path part)."""
    path = urllib.parse.urlparse(url).path
    slug = path.rstrip("/").split("/song/", 1)[1].rstrip("/").split("/")[-1]
    slug = urllib.parse.quote(urllib.parse.unquote(slug), safe="").lower()
    return f"https://cantonhymn.net/api/song-detail.php?slug={slug}&songHierarchyType={hierarchy}"


def build_chart_text(raw: Raw, *, source: str, url: str, retrieved: str, title: str | None = None,
                     artist: str | None = None, language: str | None = None, meter_hint: str | None = None,
                     tempo_hint: float | None = None, extra_header: list[str] | None = None,
                     min_events: int = 4) -> tuple[str | None, dict]:
    """Raw source lines -> (chart file text or None, info).

    The header follows data/charts/*.chart.txt: title, artist, language, key (with how it was
    found), meter, beats_per_bar, tempo, capo, year, source, retrieved, bars, content, simplified.
    ``meter_hint`` is used when the page gives no time signature (else the bar lines / slashes of
    chord-only lines, else 4/4 is assumed)."""
    notes: Counter = Counter()
    meter, meter_note = raw.meta.get("meter"), "from the page"
    if not meter and meter_hint:
        meter, meter_note = meter_hint, "from the recording"
    if not meter:
        meter, meter_note = slash_meter(raw), "from the bar lines and slashes of its chord-only lines"
    if not meter:
        meter, meter_note = "4/4", "assumed (the page gives none)"
    m = beats_for_meter(meter)
    sections = sectionize(raw.lines, notes)
    if not sections:
        return None, {"error": "no chords found"}
    tempo = raw.meta.get("tempo") or tempo_hint
    body = to_bar_text(sections, m, float(tempo) if tempo else None)
    try:
        chart = CH.parse(body, m)
    except CH.ChartError as e:
        return None, {"error": f"parse: {e}"}
    events = chord_events(chart)
    if len(events) < min_events:
        return None, {"error": f"only {len(events)} chord changes"}
    page_key = parse_key(raw.meta.get("key"))
    k, best = estimate_key(events, page_key)
    key_txt = key_name(k)
    if page_key is None:
        key_src = "estimated from the chords (the page gives none)"
    elif k == page_key and best == page_key:
        key_src = f"page says {raw.meta.get('key')}; the chords agree"
    elif k == page_key:
        key_src = f"page says {raw.meta.get('key')}; chords alone suggest {key_name(best)}"
    else:
        key_src = f"from the chords; page says {raw.meta.get('key')} (the chart is written in another key)"
    hdr = [
        f"# title: {clean_header(title or raw.meta.get('title'))}",
        f"# artist: {clean_header(artist or raw.meta.get('artist'))}",
        f"# language: {clean_header(language or raw.meta.get('language_page'))}",
        f"# key: {key_txt} ({key_src})",
        f"# meter: {meter} ({meter_note})",
        f"# beats_per_bar: {m}",
    ]
    if tempo:
        hdr.append(f"# tempo: {float(tempo):g} BPM")
    if raw.meta.get("capo"):
        hdr.append(f"# capo: {raw.meta['capo']} (chords are guitar shapes; it sounds {raw.meta['capo']} semitones higher)")
    if raw.meta.get("year"):
        hdr.append(f"# year: {clean_header(raw.meta.get('year'))}")
    hdr += [
        f"# source: {source} {url}",
        f"# retrieved: {retrieved}",
        "# bars: inferred (the source has chords over lyrics without bar lines; lengths estimated "
        "from the syllables under each chord - see nathan_chordify/sheets/parsers.py)",
        "# content: chords + section labels only (lyrics removed); private research / training use",
    ]
    if extra_header:
        hdr += extra_header
    if notes:
        hdr.append("# simplified: " + "; ".join(f"{n}" + (f" (x{c})" if c > 1 else "") for n, c in sorted(notes.items())))
    text = "\n".join(hdr) + "\n" + body
    info = {"key": key_txt, "key_index": k, "page_key": page_key, "meter": meter, "beats_per_bar": m,
            "n_events": len(events), "tempo": float(tempo) if tempo else None, "events": events}
    return text, info


def read_header(text: str) -> dict:
    hdr = {}
    for ln in text.splitlines():
        m = re.match(r"^#\s*([a-z_ ]+):\s*(.*)$", ln)
        if m:
            hdr[m.group(1).strip()] = m.group(2).strip()
    return hdr


def chart_body(text: str) -> str:
    """The bar-text lines of a chart file (header comments dropped)."""
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#")).strip() + "\n"


_BAR_LINE = re.compile(r"^\|( [^|]+ \|)+$")


def check_chart_text(text: str) -> list[str]:
    """Problems with a chart file: parse errors, and any line that is neither a comment, a
    section label nor a bar line (so no lyric can sneak in)."""
    problems = []
    hdr = read_header(text)
    try:
        m = int(hdr.get("beats_per_bar", "4"))
        chart = CH.parse(text, m)
        if not chart.sections or not any(s.bars for s in chart.sections):
            problems.append("no bars")
    except (CH.ChartError, ValueError) as e:
        problems.append(f"parse error: {e}")
    for i, ln in enumerate(text.splitlines(), 1):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if re.fullmatch(r"\[[A-Za-z][A-Za-z\- ]*(?: \d+)?\]", s):
            continue
        if not _BAR_LINE.match(s):
            problems.append(f"line {i}: not a bar line")
        elif _CJK.search(s):
            problems.append(f"line {i}: CJK text in a bar line")
    for key in ("title", "language", "key", "meter", "source", "retrieved"):
        if key not in hdr:
            problems.append(f"header: missing '{key}'")
    return problems


# =============================================================================================
# Chords over lyrics (ChordPro-style), for the user's own songs only (cache/<key>/sheets/)
# =============================================================================================

def raw_to_chordpro(raw: Raw) -> str:
    """The page's lines as ChordPro-style text: ``[C]word`` inline chords (normalised symbols),
    section labels as ``[Verse 1]`` on their own line, chord-only lines kept. Lyrics are the
    page's; this text may only be stored under the song's own cache folder."""
    out: list[str] = []
    notes: Counter = Counter()
    for line in raw.lines:
        toks = expand(line, notes)
        first = next((i for i, t in enumerate(toks) if t.kind == "chord"), len(toks))
        lead = _texts(toks[:first])
        label = match_label(lead) if lead.strip() else None
        if label is not None:
            if out and out[-1] != "":
                out.append("")
            out.append(f"[{label}]")
            toks = toks[first:]
        s = "".join(f"[{t.value}]" if t.kind == "chord" else t.value for t in toks)
        s = re.sub(r"[ \t]+", " ", s).strip()
        if s:
            out.append(s)
        elif out and out[-1] != "" and label is None:
            out.append("")
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


_INLINE = re.compile(r"\[([^\[\]]+)\]")


def transpose_chordpro(text: str, t: int, key: int | None = None) -> str:
    """Inline ``[chord]`` markers moved up ``t`` semitones (spelled in ``key`` when given); labels
    and anything that is not a chord stay as they are."""
    def _one(m: re.Match) -> str:
        tok = m.group(1).strip()
        sym, _ = norm_chord(tok)
        if sym is None or sym == "N":
            return m.group(0)
        try:
            r, q, b = H.parse_harte(H.symbol_to_harte(sym), strict=False)
        except ValueError:
            return m.group(0)
        if r < 0:
            return m.group(0)
        r2, b2 = (r + t) % 12, (b + t) % 12
        h = H.to_harte_in_key(r2, q, b2, key) if key is not None else H.to_harte(r2, q, b2)
        return "[" + H.harte_to_symbol(h) + "]"
    if t % 12 == 0 and key is None:
        return text
    return _INLINE.sub(_one, text)
