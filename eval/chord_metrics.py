"""Chord metrics for Nathan_chordify (DESIGN §13.2).

Standard: mir_eval 0.8.2 chord comparisons (root, majmin, majmin_inv, mirex, thirds(_inv),
triads(_inv), sevenths(_inv), tetrads(_inv)) and segmentation (seg, overseg, underseg).

Custom, for what Chordify-style scoring hides:
- short-segment precision / recall / F1 (reference segments <= 1 beat, and separately <= 0.6 s);
- accuracy by duration bin ((0, 1/2], (1/2, 1], (1, 2], (2, 4], > 4 beats);
- boundary F (100 ms and 1/4-beat windows) and the flicker rate;
- bass pitch-class accuracy, incl. out-of-chord basses (F/G, Em/A, E/F#) that *_inv skips;
- recall per idiom tag.

Every metric function takes ``(ref_intervals, ref_labels, est_intervals, est_labels, beats)``:
intervals are (n, 2) arrays in seconds, labels are Harte strings, beats are beat times in seconds.
Estimates are cropped / padded with N to the reference span, as in ``mir_eval.chord.evaluate``.

Label levels used by the custom metrics:
- ``root``: same root;
- ``majmin_inv``: mir_eval majmin_inv; references outside its gamut (aug, dim, sus, out-of-chord
  bass ...) fall back to triads_inv, so passing B+7 or C#dim7 chords are still scored;
- ``sevenths_inv``: mir_eval sevenths_inv, falling back to tetrads_inv.

Hit rule (short segments, idioms): an estimated segment overlaps >= 50 % of the reference segment,
starts within +-max(70 ms, 1/4 beat) of it, and its label matches at the level. Estimated
segments are judged the other way round (>= 50 % of the estimated segment).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import mir_eval
import numpy as np
from mir_eval import chord as mchord
from mir_eval import util as mutil

STANDARD_METRICS = ("root", "majmin", "majmin_inv", "mirex", "thirds", "thirds_inv", "triads",
                    "triads_inv", "sevenths", "sevenths_inv", "tetrads", "tetrads_inv")
SEGMENTATION_METRICS = ("seg", "overseg", "underseg")
LEVELS = ("root", "majmin_inv", "sevenths_inv")
_FALLBACK = {"majmin_inv": "triads_inv", "sevenths_inv": "tetrads_inv"}
DURATION_BINS = (("(0,0.5]", 0.0, 0.5), ("(0.5,1]", 0.5, 1.0), ("(1,2]", 1.0, 2.0),
                 ("(2,4]", 2.0, 4.0), (">4", 4.0, math.inf))
SHORT_BEATS, SHORT_SECONDS = 1.0, 0.6
FLICKER_BEATS = 0.5
ONSET_TOL_S, ONSET_TOL_BEATS = 0.07, 0.25
_EPS = 1e-6
NAN = float("nan")
_NO_CHORD = ("N", "X")
_EXTENSIONS = {"9": (2,), "maj9": (2,), "min9": (2,), "11": (2, 5), "min11": (2, 5),
               "13": (2, 9), "maj13": (2, 9), "min13": (2, 9)}


# ------------------------------------------------------------------------------------- loading

def load_lab(path) -> tuple[np.ndarray, list[str]]:
    """Read a .lab file (start end label)."""
    intervals, labels = mir_eval.io.load_labeled_intervals(str(path))
    return np.asarray(intervals, dtype=float), list(labels)


def load_beats(path) -> np.ndarray:
    """Read beat times from the first column of a beats file (extra columns are ignored)."""
    data = np.loadtxt(str(path), ndmin=2, comments="#")
    return np.asarray(data[:, 0], dtype=float) if data.size else np.zeros(0)


# ---------------------------------------------------------------------------------------- beats

def _grid(beats) -> np.ndarray:
    b = np.unique(np.asarray(beats, dtype=float))
    if len(b) < 2:
        raise ValueError("at least two beat times are needed")
    return b


def beat_position(times, beats) -> np.ndarray:
    """Fractional beat index of each time (linear between beats, extrapolated outside)."""
    b = _grid(beats)
    t = np.atleast_1d(np.asarray(times, dtype=float))
    pos = np.interp(t, b, np.arange(len(b), dtype=float))
    lo, hi = t < b[0], t > b[-1]
    pos[lo] = (t[lo] - b[0]) / (b[1] - b[0])
    pos[hi] = len(b) - 1 + (t[hi] - b[-1]) / (b[-1] - b[-2])
    return pos


def beat_period(times, beats) -> np.ndarray:
    """Local inter-beat interval (s) at each time."""
    b = _grid(beats)
    return np.interp(np.atleast_1d(np.asarray(times, dtype=float)), (b[:-1] + b[1:]) / 2, np.diff(b))


def onset_tolerance(times, beats) -> np.ndarray:
    """+-max(70 ms, 1/4 beat) around each onset."""
    return np.maximum(ONSET_TOL_S, ONSET_TOL_BEATS * beat_period(times, beats))


# --------------------------------------------------------------------------------------- labels

@lru_cache(maxsize=65536)
def _encode(label: str) -> tuple[int, tuple[int, ...], int]:
    root, bitmap, bass = mchord.encode(label)
    return int(root), tuple(int(v) for v in bitmap), int(bass)


def same_chord(a: str, b: str) -> bool:
    """True when two labels spell the same chord ('C' == 'C:maj')."""
    return _encode(a) == _encode(b)


@lru_cache(maxsize=65536)
def compare(ref_label: str, est_label: str, level: str) -> float:
    """1 / 0 match of one label pair at `level`; -1 = reference not comparable.

    'majmin_inv' and 'sevenths_inv' fall back to triads_inv / tetrads_inv outside their gamut."""
    score = float(getattr(mchord, level)([ref_label], [est_label])[0])
    if score < 0 and level in _FALLBACK:
        score = float(getattr(mchord, _FALLBACK[level])([ref_label], [est_label])[0])
    return score


def compare_many(ref_labels, est_labels, level: str) -> np.ndarray:
    return np.array([compare(r, e, level) for r, e in zip(ref_labels, est_labels)], dtype=float)


def bass_pc(label: str) -> int:
    """Absolute bass pitch class; -1 for N / X."""
    root, _, bass = _encode(label)
    return -1 if root < 0 else (root + bass) % 12


@lru_cache(maxsize=4096)
def has_out_of_chord_bass(label: str) -> bool:
    """True for slash chords whose bass is not a chord tone (F/G, Em/A, E/F#)."""
    if label in _NO_CHORD:
        return False
    _, quality, degrees, bass = mchord.split(label)
    tones = {i for i, on in enumerate(mchord.QUALITIES.get(quality, [0] * 12)) if on} | {0}
    tones |= set(_EXTENSIONS.get(quality, ()))
    for d in degrees:
        semis = mchord.scale_degree_to_semitone(d.lstrip("*")) % 12
        (tones.discard if d.startswith("*") else tones.add)(semis)
    return mchord.scale_degree_to_semitone(bass) % 12 not in tones


def _f1(p: float, r: float) -> float:
    if math.isnan(p) or math.isnan(r):
        return 0.0 if (p == 0 or r == 0) else NAN
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else NAN


# ------------------------------------------------------------------------------------- segments

@dataclass
class Segments:
    """Contiguous labelled segments (identical neighbours merged)."""
    start: np.ndarray
    end: np.ndarray
    labels: list[str]
    tags: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def dur(self) -> np.ndarray:
        return self.end - self.start

    def __len__(self) -> int:
        return len(self.labels)

    def overlapping(self, t0: float, t1: float) -> range:
        lo = int(np.searchsorted(self.end, t0 + _EPS, side="left"))
        hi = int(np.searchsorted(self.start, t1 - _EPS, side="left"))
        return range(lo, hi)


def _fill_gaps(intervals, labels) -> tuple[np.ndarray, list[str]]:
    """Sort by onset and fill gaps between segments with N."""
    iv = np.asarray(intervals, dtype=float).reshape(-1, 2)
    order = np.argsort(iv[:, 0], kind="stable")
    out_iv, out_lab = [], []
    for k in order:
        t0, t1 = iv[k]
        if out_iv and t0 > out_iv[-1][1] + _EPS:
            out_iv.append([out_iv[-1][1], t0])
            out_lab.append("N")
        if out_iv and t0 < out_iv[-1][1]:  # overlap: the later segment wins
            out_iv[-1][1] = t0
        if t1 > t0:
            out_iv.append([t0, t1])
            out_lab.append(labels[k])
    return np.array(out_iv, dtype=float).reshape(-1, 2), out_lab


def _align(ref_intervals, ref_labels, est_intervals, est_labels):
    """Reference without gaps; estimate without gaps, cropped / N-padded to the reference span."""
    ref_iv, ref_lab = _fill_gaps(ref_intervals, ref_labels)
    t_min, t_max = float(ref_iv[0, 0]), float(ref_iv[-1, 1])
    est_iv, est_lab = _fill_gaps(est_intervals, est_labels)
    if len(est_lab) == 0:
        est_iv, est_lab = np.array([[t_min, t_max]]), ["N"]
    est_iv, est_lab = mutil.adjust_intervals(est_iv, est_lab, t_min, t_max, mchord.NO_CHORD, mchord.NO_CHORD)
    return ref_iv, list(ref_lab), np.asarray(est_iv, dtype=float), list(est_lab)


def segments(intervals, labels, tags=None, merge: bool = True) -> Segments:
    """Segments with consecutive identical chords merged (their tags are united)."""
    tags = list(tags) if tags is not None else [()] * len(labels)
    start, end, labs, tgs = [], [], [], []
    for (t0, t1), lab, tag in zip(np.asarray(intervals, dtype=float).reshape(-1, 2), labels, tags):
        tag = _as_tags(tag)
        if merge and labs and abs(end[-1] - t0) < _EPS and same_chord(labs[-1], lab):
            end[-1] = t1
            tgs[-1] = tuple(sorted(set(tgs[-1]) | set(tag)))
        else:
            start.append(t0)
            end.append(t1)
            labs.append(lab)
            tgs.append(tag)
    return Segments(np.array(start, dtype=float), np.array(end, dtype=float), labs, tgs)


def _as_tags(tag) -> tuple[str, ...]:
    if not tag:
        return ()
    if isinstance(tag, str):
        return (tag,)
    return tuple(tag)


def _pair_segments(ref_intervals, ref_labels, est_intervals, est_labels):
    ref_iv, ref_lab, est_iv, est_lab = _align(ref_intervals, ref_labels, est_intervals, est_labels)
    return segments(ref_iv, ref_lab), segments(est_iv, est_lab), ref_iv, ref_lab, est_iv, est_lab


def _timeline(ref_iv, ref_lab, est_iv, est_lab):
    """Common sub-intervals of aligned reference and estimate: (intervals, ref, est, durations)."""
    iv, r, e = mutil.merge_labeled_intervals(ref_iv, ref_lab, est_iv, est_lab)
    return np.asarray(iv), list(r), list(e), mutil.intervals_to_durations(iv)


def _weighted(comps: np.ndarray, dur: np.ndarray) -> tuple[float, float]:
    """(correct seconds, scored seconds) over comparable pieces."""
    ok = comps >= 0
    return float(np.sum(comps[ok] * dur[ok])), float(np.sum(dur[ok]))


def _beats_of(segs: Segments, beats) -> np.ndarray:
    if len(segs) == 0:
        return np.zeros(0)
    return beat_position(segs.end, beats) - beat_position(segs.start, beats)


def _is_chord(label: str) -> bool:
    return label not in _NO_CHORD


# -------------------------------------------------------------------------------------- matching

def _find_hit(t0: float, t1: float, label: str, est: Segments, tol: float, level: str) -> int:
    """Index of an estimated segment that hits the reference segment [t0, t1), else -1."""
    for j in est.overlapping(t0, t1):
        ov = min(t1, est.end[j]) - max(t0, est.start[j])
        if (ov >= 0.5 * (t1 - t0) - _EPS and abs(est.start[j] - t0) <= tol + _EPS
                and compare(label, est.labels[j], level) == 1):
            return j
    return -1


def _est_matched(j: int, est: Segments, ref: Segments, ref_tol: np.ndarray, level: str) -> bool:
    """True when estimated segment j is a correct chord at a real reference onset."""
    e0, e1 = est.start[j], est.end[j]
    for i in ref.overlapping(e0, e1):
        ov = min(e1, ref.end[i]) - max(e0, ref.start[i])
        if (ov >= 0.5 * (e1 - e0) - _EPS and abs(e0 - ref.start[i]) <= ref_tol[i] + _EPS
                and _is_chord(ref.labels[i]) and compare(ref.labels[i], est.labels[j], level) == 1):
            return True
    return False


# ------------------------------------------------------------------------------ standard metrics

def standard_scores(ref_intervals, ref_labels, est_intervals, est_labels, beats=None) -> dict:
    """mir_eval chord scores (duration-weighted; NaN when nothing is comparable) + seg/overseg/underseg."""
    ref_iv, ref_lab, est_iv, est_lab = _align(ref_intervals, ref_labels, est_intervals, est_labels)
    _, r, e, dur = _timeline(ref_iv, ref_lab, est_iv, est_lab)
    out = {}
    for name in STANDARD_METRICS:
        correct, scored = _weighted(np.asarray(getattr(mchord, name)(r, e), dtype=float), dur)
        out[name] = _ratio(correct, scored)
    ref_m = mchord.merge_chord_intervals(ref_iv, ref_lab)
    est_m = mchord.merge_chord_intervals(est_iv, est_lab)
    out["underseg"] = float(mchord.underseg(ref_m, est_m))
    out["overseg"] = float(mchord.overseg(ref_m, est_m))
    out["seg"] = min(out["overseg"], out["underseg"])
    return out


# --------------------------------------------------------------------------------- short segments

def short_segment_prf(ref_intervals, ref_labels, est_intervals, est_labels, beats, *,
                      max_beats: float = SHORT_BEATS, max_seconds: float | None = None,
                      levels=LEVELS) -> dict:
    """Short-segment precision / recall / F1 per level.

    Short = at most `max_beats` beats (or `max_seconds` seconds when given). Recall: short
    reference chords hit by the estimate. Precision: of the estimated segments that are short or
    that hit a short reference chord, the share that are a correct chord at a reference onset.
    Each level also lists its misses as [t0, t1, label]."""
    ref, est, *_ = _pair_segments(ref_intervals, ref_labels, est_intervals, est_labels)

    def short(segs: Segments) -> np.ndarray:
        if max_seconds is not None:
            return segs.dur <= max_seconds + _EPS
        return _beats_of(segs, beats) <= max_beats + _EPS

    ref_tol = onset_tolerance(ref.start, beats) if len(ref) else np.zeros(0)
    ref_short = [i for i in np.flatnonzero(short(ref)) if _is_chord(ref.labels[i])]
    est_short = {int(j) for j in np.flatnonzero(short(est)) if _is_chord(est.labels[j])}
    out = {}
    for level in levels:
        hit_est, misses = set(), []
        for i in ref_short:
            j = _find_hit(ref.start[i], ref.end[i], ref.labels[i], est, ref_tol[i], level)
            if j >= 0:
                hit_est.add(j)
            else:
                misses.append([round(float(ref.start[i]), 3), round(float(ref.end[i]), 3), ref.labels[i]])
        hits = len(ref_short) - len(misses)
        cand = est_short | hit_est
        tp = sum(1 for j in cand if j in hit_est or _est_matched(j, est, ref, ref_tol, level))
        p, r = _ratio(tp, len(cand)), _ratio(hits, len(ref_short))
        out[level] = {"n_ref": len(ref_short), "hits": hits, "n_est": len(cand), "tp": tp,
                      "precision": p, "recall": r, "f1": _f1(p, r), "misses": misses}
    return out


# -------------------------------------------------------------------------------- duration bins

def accuracy_by_duration(ref_intervals, ref_labels, est_intervals, est_labels, beats, *, levels=LEVELS) -> dict:
    """Accuracy per reference-duration bin (in beats), seconds-weighted, per level.

    Each bin reports n (reference chords), seconds, and per level acc / correct_s / scored_s."""
    ref, _, ref_iv, ref_lab, est_iv, est_lab = _pair_segments(ref_intervals, ref_labels, est_intervals, est_labels)
    iv, r, e, dur = _timeline(ref_iv, ref_lab, est_iv, est_lab)
    seg_idx = np.searchsorted(ref.start, (iv[:, 0] + iv[:, 1]) / 2, side="right") - 1
    seg_beats = _beats_of(ref, beats)
    comps = {lv: compare_many(r, e, lv) for lv in levels}
    out = {}
    for name, lo, hi in DURATION_BINS:
        in_bin = [i for i in range(len(ref)) if lo + _EPS < seg_beats[i] <= hi + _EPS and _is_chord(ref.labels[i])]
        mask = np.isin(seg_idx, in_bin)
        row = {"n": len(in_bin), "seconds": round(float(np.sum(ref.dur[in_bin])) if in_bin else 0.0, 6)}
        for lv in levels:
            correct, scored = _weighted(comps[lv][mask], dur[mask])
            row[lv] = {"acc": _ratio(correct, scored), "correct_s": correct, "scored_s": scored}
        out[name] = row
    return out


# ------------------------------------------------------------------------------ boundaries, flicker

def boundary_f(ref_intervals, ref_labels, est_intervals, est_labels, beats, *,
               window: float = 0.1, beat_window: float | None = None) -> dict:
    """Chord-change F-measure with a fixed window (s), or a window in beats when `beat_window` is set."""
    ref, est, *_ = _pair_segments(ref_intervals, ref_labels, est_intervals, est_labels)
    rb, eb = ref.start[1:], est.start[1:]
    if beat_window is not None:
        rb = beat_position(rb, beats) if len(rb) else rb
        eb = beat_position(eb, beats) if len(eb) else eb
        window = beat_window
    k = len(mutil.match_events(rb, eb, window + 1e-9)) if len(rb) and len(eb) else 0
    if len(rb) == 0 and len(eb) == 0:
        p = r = 1.0
    else:
        p, r = (_ratio(k, len(eb)) if len(eb) else 0.0), (_ratio(k, len(rb)) if len(rb) else 0.0)
    return {"n_ref": len(rb), "n_est": len(eb), "n_match": k, "precision": p, "recall": r, "f": _f1(p, r)}


def flicker_stats(ref_intervals, ref_labels, est_intervals, est_labels, beats, *,
                  max_beats: float = FLICKER_BEATS, level: str = "majmin_inv") -> dict:
    """Estimated chords shorter than `max_beats` that are not a correct chord at a reference onset."""
    ref, est, ref_iv, *_ = _pair_segments(ref_intervals, ref_labels, est_intervals, est_labels)
    ref_tol = onset_tolerance(ref.start, beats) if len(ref) else np.zeros(0)
    short = np.flatnonzero(_beats_of(est, beats) < max_beats - _EPS)
    bad = [int(j) for j in short if _is_chord(est.labels[j]) and not _est_matched(int(j), est, ref, ref_tol, level)]
    minutes = float(ref_iv[-1, 1] - ref_iv[0, 0]) / 60.0
    return {"count": len(bad), "minutes": minutes, "rate": _ratio(len(bad), minutes),
            "times": [round(float(est.start[j]), 3) for j in bad]}


def flicker_rate(ref_intervals, ref_labels, est_intervals, est_labels, beats, **kw) -> float:
    """Unmatched estimated segments under 1/2 beat, per minute (target < 2)."""
    return flicker_stats(ref_intervals, ref_labels, est_intervals, est_labels, beats, **kw)["rate"]


# ------------------------------------------------------------------------------------------- bass

def _bass_pieces(ref_intervals, ref_labels, est_intervals, est_labels):
    ref, _, ref_iv, ref_lab, est_iv, est_lab = _pair_segments(ref_intervals, ref_labels, est_intervals, est_labels)
    iv, r, e, dur = _timeline(ref_iv, ref_lab, est_iv, est_lab)
    rb = np.array([bass_pc(x) for x in r])
    eq = rb == np.array([bass_pc(x) for x in e])
    seg_idx = np.searchsorted(ref.start, (iv[:, 0] + iv[:, 1]) / 2, side="right") - 1
    return ref, r, rb, eq, dur, seg_idx


def bass_accuracy(ref_intervals, ref_labels, est_intervals, est_labels, beats=None) -> dict:
    """Bass pitch-class accuracy: seconds-weighted (acc) and per reference chord (seg_acc: right
    for at least half of the chord)."""
    ref, _, rb, eq, dur, seg_idx = _bass_pieces(ref_intervals, ref_labels, est_intervals, est_labels)
    valid = rb >= 0
    correct, scored = float(np.sum(dur[valid & eq])), float(np.sum(dur[valid]))
    chords = [i for i in range(len(ref)) if _is_chord(ref.labels[i])]
    seg_ok = sum(1 for i in chords if np.sum(dur[(seg_idx == i) & eq]) >= 0.5 * ref.dur[i] - _EPS)
    return {"acc": _ratio(correct, scored), "correct_s": correct, "scored_s": scored,
            "seg_acc": _ratio(seg_ok, len(chords)), "seg_correct": seg_ok, "n_seg": len(chords)}


def out_of_chord_bass_accuracy(ref_intervals, ref_labels, est_intervals, est_labels, beats=None) -> dict:
    """Bass accuracy over reference chords whose bass is not a chord tone (F/G, Em/A): the
    mir_eval *_inv metrics leave these out. Seconds-weighted; NaN when there are none."""
    ref, r, rb, eq, dur, seg_idx = _bass_pieces(ref_intervals, ref_labels, est_intervals, est_labels)
    ooc = np.array([has_out_of_chord_bass(x) for x in r], dtype=bool)
    correct, scored = float(np.sum(dur[ooc & eq])), float(np.sum(dur[ooc]))
    n_seg = sum(1 for lab in ref.labels if has_out_of_chord_bass(lab))
    return {"acc": _ratio(correct, scored), "correct_s": correct, "scored_s": scored, "n_seg": n_seg}


# ----------------------------------------------------------------------------------------- idioms

def idiom_recall(ref_intervals, ref_labels, est_intervals, est_labels, beats, ref_tags, *, levels=LEVELS) -> dict:
    """Recall per idiom tag. `ref_tags` is aligned with ref_labels (None, a tag, or a list of tags);
    a tagged reference chord is found when the short-segment hit rule holds."""
    _, est, *_ = _pair_segments(ref_intervals, ref_labels, est_intervals, est_labels)
    ref_iv = np.asarray(ref_intervals, dtype=float).reshape(-1, 2)
    out: dict[str, dict] = {}
    for (t0, t1), label, tags in zip(ref_iv, ref_labels, ref_tags):
        tags = _as_tags(tags)
        if not tags or not _is_chord(label):
            continue
        tol = float(onset_tolerance([t0], beats)[0])
        hit = {lv: _find_hit(t0, t1, label, est, tol, lv) >= 0 for lv in levels}
        for tag in tags:
            row = out.setdefault(tag, {"n": 0, "hits": {lv: 0 for lv in levels}})
            row["n"] += 1
            for lv in levels:
                row["hits"][lv] += int(hit[lv])
    for row in out.values():
        row["recall"] = {lv: _ratio(row["hits"][lv], row["n"]) for lv in levels}
    return out


# ------------------------------------------------------------------------------------------- all

def evaluate_all(ref_intervals, ref_labels, est_intervals, est_labels, beats, ref_tags=None) -> dict:
    """Every metric of §13.2 that needs only labels and beats."""
    args = (ref_intervals, list(ref_labels), est_intervals, list(est_labels))
    out = {
        "standard": standard_scores(*args),
        "shortseg_1beat": short_segment_prf(*args, beats, max_beats=SHORT_BEATS),
        "shortseg_0.6s": short_segment_prf(*args, beats, max_seconds=SHORT_SECONDS),
        "duration_bins": accuracy_by_duration(*args, beats),
        "boundary_100ms": boundary_f(*args, beats, window=0.1),
        "boundary_qbeat": boundary_f(*args, beats, beat_window=0.25),
        "flicker": flicker_stats(*args, beats),
        "bass": bass_accuracy(*args, beats),
        "ooc_bass": out_of_chord_bass_accuracy(*args, beats),
    }
    if ref_tags is not None:
        out["idioms"] = idiom_recall(*args, beats, ref_tags)
    return out
