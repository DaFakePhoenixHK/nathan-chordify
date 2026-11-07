"""Key estimation (WP6, DESIGN §5.4).

Keys are ints 0..23 (0–11 major tonic pc, 12–23 minor tonic pc + 12), as in ``KeyTrack.per_beat``.

``estimate`` (pipeline step 2) = §5.4 steps 1–2:

1. Beat-synchronous ``chroma_cqt`` of the harmonic stem (hop 4096 @ 22.05 kHz). Each of the 24
   keys gets the mean of its Krumhansl–Kessler and Temperley (Kostka–Payne) correlations with
   the whole-song chroma, plus 0.3 × the correlation of the bass-salience histogram with a
   tonic-weighted profile (the mean profile with its tonic weight doubled). These global scores
   are the HMM's initial distribution.
2. Local key: a 24-state HMM over beats. Emission = log-softmax over the keys of cos/T (T = 0.02),
   cos = cosine of an 8-beat chroma window with the key profile (mean of the unit-norm KK and
   Temperley profiles). Self-transition 0.995; moves to the relative, parallel, dominant and
   subdominant keys and same-mode moves up 1 or 2 semitones are twice as likely as other moves.
   Viterbi path; key segments shorter than 16 beats are merged into the neighbour that explains
   them better. Two corrections (eval/reports/M2.md: 回家 read a third of the song as G#m / E /
   C#m because worship songs sit on vi, IV and ii for long stretches):
   - each beat's chroma sits in 8 overlapping windows, so the path uses the emissions / 8
     (``path_evidence``): one beat of chroma is counted once, not 8 times;
   - moves to a *close* key (scale one note away: vi, IV, V, ii, iii — ``close_keys``) cost
     CLOSE_PENALTY more nats. Chroma cannot tell those apart once the song borrows ♭VII / iv or
     tonicises vi, so they are left to the chord-based step 3. Step-ups (+1/+2) and distant
     modulations change ≥ 2 notes and keep the design's weights.

   ``global_*`` is the key holding the most beats; ``global_conf`` is its probability under
   softmax(global scores / 0.05), i.e. how clearly the whole-song profile (chroma + bass) says so.

``refine`` (pipeline step 10) = steps 3–4, on decoded chords (``Segment.cand`` indexes
``theory.vocab.candidates().cands``):

3. Chord-key HMM over beats. Emission = duration-weighted log P(chord | key) of the main-layer
   chords (``MAJOR_CHORDS`` / ``MINOR_CHORDS``: degree × triad family, so I/IV/V/vi dominate,
   secondary dominants and borrowed chords are rare, and a minor key needs its major V — the
   cadence — to beat its relative major), plus PROFILE_BONUS per beat for the global key where
   the profile path agrees with it (ties such as Am–F–C–G follow the profile). The path starts
   and ends in the global key and pays REFINE_SWITCH nats per change (ln 2 less for favoured
   moves), so a change needs sustained chord support: a 4-bar V/vi → vi tonicisation or a
   section dwelling on IV with ♭VII / iv stays home, a step-up or a real modulation does not.
   The profile's own local keys get no bonus: a change found by chroma alone is dropped.
   Runs shorter than 16 beats are merged again.
4. Truck-driver test: at each section boundary in the last two thirds of the song (``sections``
   when given, else 8-bar blocks), the next section's chords transposed down 1 and 2 semitones are
   compared beat by beat with each earlier section; ≥ 70 % equal classes (and better than the
   untransposed comparison) marks a ``step_up`` at the boundary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import BassTrack, KeyChange, KeyTrack, RhythmGrid, Segment, Stems

SR = 22050
CHROMA_HOP = 4096
N_KEYS = 24

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Temperley, "Music and Probability" (2007): Kostka–Payne corpus profiles
TP_MAJOR = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
TP_MINOR = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])

BASS_WEIGHT = 0.3
EMIT_T = 0.02
INIT_T = 0.05            # temperature of the global scores used as the initial distribution
P_STAY = 0.995
FAVOURED = 2.0           # relative weight of the favoured moves
WINDOW_BEATS = 8
MIN_SEG_BEATS = 16
CLOSE_PENALTY = 12.0     # extra nats for a profile-HMM move to a close key (see ``close_keys``)

STEP_MATCH, STEP_MIN_BEATS = 0.7, 8
BLOCK_BARS = 8

# Chord-based key model (refine, step 3): P(chord | key) per scale degree (semitones above the
# tonic) and triad family, pop/worship-style. I, IV, V and vi carry most of the mass; secondary
# dominants (II, III, VI = V/V, V/vi, V/ii), borrowed chords (♭VII, iv, ♭VI, ♭III) and passing
# diminished chords are possible but rare, so they colour a key without moving it. In minor, the
# major V (raised leading tone) is the cadential evidence that separates a key from its relative
# major, where the same chord is III, a secondary dominant.
MAJOR_CHORDS = {0: {"maj": .25, "min": .01, "aug": .006}, 1: {"maj": .002, "dim": .006},
                2: {"min": .055, "maj": .025, "dim": .004}, 3: {"maj": .005, "dim": .006},
                4: {"min": .03, "maj": .015}, 5: {"maj": .20, "min": .025}, 6: {"dim": .006},
                7: {"maj": .19, "min": .006, "aug": .003}, 8: {"maj": .015}, 9: {"min": .12, "maj": .01},
                10: {"maj": .04}, 11: {"dim": .01, "min": .004, "maj": .004}}
MINOR_CHORDS = {0: {"min": .25, "maj": .012}, 1: {"maj": .006}, 2: {"dim": .02, "min": .01, "maj": .005},
                3: {"maj": .10, "aug": .003}, 5: {"min": .10, "maj": .02}, 6: {"dim": .005},
                7: {"maj": .12, "min": .045}, 8: {"maj": .13}, 10: {"maj": .12}, 11: {"dim": .02}}
CHORD_FLOOR = 0.002      # any other degree × family
REFINE_SWITCH = 16.0     # nats of chord evidence a key change must pay for (ln 2 less for favoured moves)
PROFILE_BONUS = 0.4      # nats per beat for the global key where the profile path agrees with it:
                         # decides chord ties such as relative keys (Am–F–C–G); none for profile-only changes


# --------------------------------------------------------------------------------------------
# keys and profiles
# --------------------------------------------------------------------------------------------

def tonic(k: int) -> int:
    return k % 12


def is_minor(k: int) -> bool:
    return k >= 12


def make_key(tonic_pc: int, minor: bool) -> int:
    return tonic_pc % 12 + (12 if minor else 0)


def key_name(k: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[tonic(k)]}:{'minor' if is_minor(k) else 'major'}"


def _rotations(major: np.ndarray, minor: np.ndarray) -> np.ndarray:
    """[24, 12] profiles: row k = the profile of key k."""
    return np.stack([np.roll(major, t) for t in range(12)] + [np.roll(minor, t) for t in range(12)])


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


KK = _rotations(KK_MAJOR, KK_MINOR)
TP = _rotations(TP_MAJOR, TP_MINOR)
PROFILE = _unit(0.5 * (_unit(KK) + _unit(TP)))            # [24,12] HMM emission profile


def _bass_profile() -> np.ndarray:
    base_maj, base_min = 0.5 * (_unit(KK_MAJOR) + _unit(TP_MAJOR)), 0.5 * (_unit(KK_MINOR) + _unit(TP_MINOR))
    base_maj, base_min = base_maj.copy(), base_min.copy()
    base_maj[0] *= 2.0
    base_min[0] *= 2.0
    return _rotations(base_maj, base_min)


BASS_PROFILE = _bass_profile()


def _corr(x: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    """Pearson correlation of one 12-vector with each row of ``profiles``."""
    xc = x - x.mean()
    pc = profiles - profiles.mean(axis=1, keepdims=True)
    den = np.linalg.norm(xc) * np.linalg.norm(pc, axis=1)
    return np.where(den > 0, pc @ xc / np.maximum(den, 1e-12), 0.0)


def favoured_moves(k: int) -> set[int]:
    """Relative, parallel, dominant, subdominant, and same-mode +1 / +2 semitones."""
    t, m = tonic(k), is_minor(k)
    rel = make_key(t + 9, True) if not m else make_key(t + 3, False)
    return {rel, make_key(t, not m)} | {make_key(t + d, m) for d in (7, 5, 1, 2)}


def transition_matrix() -> np.ndarray:
    """[24,24] row-stochastic key transitions per beat."""
    A = np.zeros((N_KEYS, N_KEYS))
    for k in range(N_KEYS):
        w = np.ones(N_KEYS)
        w[list(favoured_moves(k))] = FAVOURED
        w[k] = 0.0
        A[k] = (1.0 - P_STAY) * w / w.sum()
        A[k, k] = P_STAY
    return A


def signature(k: int) -> int:
    """Position of the key signature on the circle of fifths (C = 0, G = 1, …; a minor key
    takes its relative major's)."""
    t = (tonic(k) + 3) % 12 if is_minor(k) else tonic(k)
    return (7 * t) % 12


def close_keys(k: int) -> set[int]:
    """Keys whose scale differs from ``k``'s by at most one note (signature within one fifth):
    for B major, B, G#m, E, C#m, F# and D#m. Step-ups (+1 = 5 notes, +2 = 2 notes) are not close."""
    return {j for j in range(N_KEYS) if min((signature(j) - signature(k)) % 12, (signature(k) - signature(j)) % 12) <= 1}


def profile_log_transitions() -> np.ndarray:
    """[24,24] log transitions of the profile (chroma) HMM: ``transition_matrix`` with moves to a
    close key made e^CLOSE_PENALTY times less likely. Chroma cannot tell a key from its close
    keys once the music borrows chords (♭VII and iv in B major are diatonic to E major; G#m
    shares all of B major's notes), so those moves are left to the chord-based ``refine``."""
    with np.errstate(divide="ignore"):
        log_A = np.log(transition_matrix())
    for k in range(N_KEYS):
        log_A[k, [j for j in close_keys(k) if j != k]] -= CLOSE_PENALTY
    return log_A


def classify_change(a: int, b: int) -> str:
    if is_minor(a) == is_minor(b) and (tonic(b) - tonic(a)) % 12 in (1, 2):
        return "step_up"
    if tonic(a) == tonic(b):
        return "parallel"
    if (not is_minor(a) and b == make_key(tonic(a) + 9, True)) or (is_minor(a) and b == make_key(tonic(a) + 3, False)):
        return "relative"
    return "modulation"


# --------------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------------

def harmonic_chroma(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalised ``chroma_cqt`` [F,12] of a file (hop 4096 @ 22.05 kHz) and frame centre times."""
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    ch = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=CHROMA_HOP, norm=None)
    times = librosa.frames_to_time(np.arange(ch.shape[1]), sr=SR, hop_length=CHROMA_HOP)
    return ch.T, times


def beat_chroma(chroma: np.ndarray, times: np.ndarray, beats: np.ndarray, ibi: float) -> np.ndarray:
    """Sum frames into beats ([b, b+1), the last beat lasting ``ibi``), then max-normalise.

    Beats without a frame take the frame nearest their midpoint; near-silent beats
    (< −30 dB of the loud beats) become all-zero (no evidence).
    """
    B = len(beats)
    if B == 0:
        return np.zeros((0, 12))
    ends = np.append(beats[1:], beats[-1] + ibi)
    idx = np.searchsorted(beats, times, side="right") - 1
    ok = (idx >= 0) & (times < ends[np.clip(idx, 0, B - 1)])
    out = np.zeros((B, 12))
    np.add.at(out, idx[ok], chroma[ok])
    empty = np.bincount(idx[ok], minlength=B)[:B] == 0
    if empty.any() and len(times):
        mids = 0.5 * (beats[empty] + ends[empty])
        out[empty] = chroma[np.abs(times[None, :] - mids[:, None]).argmin(axis=1)]
    energy = out.sum(axis=1)
    loud = np.percentile(energy, 95) if B else 0.0
    out[energy < 1e-3 * loud] = 0.0
    mx = out.max(axis=1, keepdims=True)
    return np.divide(out, mx, out=np.zeros_like(out), where=mx > 0)


def bass_histogram(bass: BassTrack | None) -> np.ndarray | None:
    """Pitch-class histogram of the bass salience, weighted by a 40 dB energy gate."""
    if bass is None or bass.salience is None or len(bass.salience) == 0:
        return None
    e = np.asarray(bass.energy_db, dtype=float)
    w = np.clip((e - (e.max() - 40.0)) / 40.0, 0.0, 1.0) if e.size else np.ones(len(bass.salience))
    h = (np.asarray(bass.salience) * w[:, None]).sum(axis=0)
    return h / h.sum() if h.sum() > 0 else None


def global_scores(chroma_sum: np.ndarray, bass_hist: np.ndarray | None) -> np.ndarray:
    """§5.4 step 1: [24] = mean(KK corr, Temperley corr) + 0.3 · bass corr."""
    s = 0.5 * (_corr(chroma_sum, KK) + _corr(chroma_sum, TP))
    if bass_hist is not None:
        s = s + BASS_WEIGHT * _corr(bass_hist, BASS_PROFILE)
    return s


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def emissions(bc: np.ndarray) -> np.ndarray:
    """[B,24] log-softmax(cos/T) of the 8-beat chroma window [b−4, b+4) against each key."""
    B = len(bc)
    cs = np.vstack([np.zeros((1, 12)), np.cumsum(bc, axis=0)])
    lo = np.clip(np.arange(B) - WINDOW_BEATS // 2, 0, B)
    hi = np.clip(np.arange(B) + WINDOW_BEATS // 2, 0, B)
    win = cs[hi] - cs[lo]
    norm = np.linalg.norm(win, axis=1, keepdims=True)
    cos = np.divide(win, norm, out=np.zeros_like(win), where=norm > 0) @ PROFILE.T
    return _log_softmax(cos / EMIT_T)


def path_evidence(bc: np.ndarray) -> np.ndarray:
    """[B,24] per-beat evidence for the local-key Viterbi: ``emissions`` / WINDOW_BEATS.

    Each beat's chroma sits in WINDOW_BEATS overlapping windows, so the raw emissions count it
    WINDOW_BEATS times; dividing by the window length counts it once (for path decisions this is
    log-softmax(cos / (T·WINDOW_BEATS)) up to a per-beat constant).
    """
    return emissions(bc) / WINDOW_BEATS


# --------------------------------------------------------------------------------------------
# HMM
# --------------------------------------------------------------------------------------------

def viterbi(log_e: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray,
            log_end: np.ndarray | None = None) -> np.ndarray:
    """Best key path; ``log_end`` (optional) scores the last key, like a final transition."""
    B = len(log_e)
    delta = log_pi + log_e[0]
    back = np.zeros((B, N_KEYS), dtype=np.int64)
    for b in range(1, B):
        cand = delta[:, None] + log_A
        back[b] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + log_e[b]
    path = np.empty(B, dtype=np.int64)
    path[-1] = int((delta if log_end is None else delta + log_end).argmax())
    for b in range(B - 1, 0, -1):
        path[b - 1] = back[b, path[b]]
    return path


def runs(path: np.ndarray) -> list[list[int]]:
    """[[key, b0, b1), ...] runs of equal keys."""
    out: list[list[int]] = []
    for b, k in enumerate(path):
        if out and out[-1][0] == k:
            out[-1][2] = b + 1
        else:
            out.append([int(k), b, b + 1])
    return out


def merge_short(path: np.ndarray, score: np.ndarray, min_len: int = MIN_SEG_BEATS) -> np.ndarray:
    """Merge runs shorter than ``min_len`` beats into the neighbour whose key scores higher there."""
    path = path.copy()
    while True:
        rs = runs(path)
        short = [i for i, r in enumerate(rs) if r[2] - r[1] < min_len]
        if len(rs) <= 1 or not short:
            return path
        i = min(short, key=lambda j: rs[j][2] - rs[j][1])
        k, b0, b1 = rs[i]
        nbrs = [rs[j][0] for j in (i - 1, i + 1) if 0 <= j < len(rs)]
        path[b0:b1] = max(nbrs, key=lambda n: score[b0:b1, n].sum())


def _bar_at(grid: RhythmGrid, b: int) -> int:
    bars = np.asarray(grid.bar_of_beat)
    if len(bars) == 0:
        return 0
    if bars[b] >= 0:
        return int(bars[b])
    after = np.nonzero(bars[b:] >= 0)[0]
    if len(after):
        return int(bars[b + after[0]])
    before = np.nonzero(bars[:b] >= 0)[0]
    return int(bars[before[-1]]) if len(before) else 0


def _changes(path: np.ndarray, grid: RhythmGrid, step_ups: set[int] = frozenset()) -> list[KeyChange]:
    out = []
    rs = runs(path)
    for prev, cur in zip(rs, rs[1:]):
        b = cur[1]
        kind = "step_up" if b in step_ups else classify_change(prev[0], cur[0])
        out.append(KeyChange(t=float(grid.beats[b]), bar=_bar_at(grid, b), from_key=prev[0], to_key=cur[0],
                             kind=kind))
    return out


def _global(path: np.ndarray, scores: np.ndarray) -> tuple[int, float]:
    """The key holding the most beats (ties → better global score; no beats → best score) and
    its probability under softmax(scores / INIT_T)."""
    p = np.exp(_log_softmax(scores / INIT_T))
    if len(path) == 0:
        k = int(p.argmax())
    else:
        k = int(np.argmax(np.bincount(path, minlength=N_KEYS) + 1e-6 * scores))
    return k, float(p[k])


def estimate_from_chroma(bc: np.ndarray, grid: RhythmGrid, bass_hist: np.ndarray | None = None) -> KeyTrack:
    """Steps 1–2 on beat chroma [B,12] (one row per ``grid.beats``)."""
    bc = np.asarray(bc, dtype=float)
    scores = global_scores(bc.sum(axis=0) if len(bc) else np.ones(12), bass_hist)
    if len(bc) == 0:
        k, conf = _global(np.zeros(0, dtype=np.int64), scores)
        return KeyTrack(tonic(k), "minor" if is_minor(k) else "major", conf, np.zeros(0, dtype=np.int64), [])
    log_e = path_evidence(bc)
    path = viterbi(log_e, profile_log_transitions(), _log_softmax(scores / INIT_T))
    path = merge_short(path, log_e)
    k, conf = _global(path, scores)
    return KeyTrack(global_tonic=tonic(k), global_mode="minor" if is_minor(k) else "major",
                    global_conf=conf, per_beat=path, changes=_changes(path, grid))


def estimate(stems: Stems, grid: RhythmGrid, bass: BassTrack | None) -> KeyTrack:
    """§5.4 steps 1–2: global key scores and the local-key HMM over ``grid.beats``."""
    chroma, times = harmonic_chroma(stems.harmonic)
    beats = np.asarray(grid.beats, dtype=float)
    bc = beat_chroma(chroma, times, beats, float(grid.ibi_ref) if grid.ibi_ref else 0.5)
    if len(beats) == 0:          # no beats: the global key from the whole-song chroma only
        scores = global_scores(chroma.sum(axis=0), bass_histogram(bass))
        k, conf = _global(np.zeros(0, dtype=np.int64), scores)
        return KeyTrack(tonic(k), "minor" if is_minor(k) else "major", conf, np.zeros(0, dtype=np.int64), [])
    return estimate_from_chroma(bc, grid, bass_histogram(bass))


# --------------------------------------------------------------------------------------------
# refinement on decoded chords (steps 3–4)
# --------------------------------------------------------------------------------------------

def _family(quality: str) -> str:
    from . import templates as T
    if quality in T.DIM_FAMILY:
        return "dim"
    if quality in T.AUG_FAMILY:
        return "aug"
    if quality in T.MINOR_FAMILY:
        return "min"
    if quality in T.SUS4_FAMILY or quality in ("sus2", "5"):
        return "sus"
    return "maj"


FAMILIES = ("maj", "min", "dim", "aug")


def _chord_table(spec: dict[int, dict[str, float]]) -> np.ndarray:
    """[12 degrees, 4 families] probabilities from a sparse spec, floored and normalised."""
    t = np.full((12, len(FAMILIES)), CHORD_FLOOR)
    for deg, fams in spec.items():
        for fam, p in fams.items():
            t[deg, FAMILIES.index(fam)] = p
    return t / t.sum()


CHORD_TABLE = {False: _chord_table(MAJOR_CHORDS), True: _chord_table(MINOR_CHORDS)}


def chord_loglik(root: int, quality: str, k: int) -> float:
    """log P(chord | key ``k``) from its scale degree and triad family (``MAJOR_CHORDS`` /
    ``MINOR_CHORDS``). sus and power chords have no third: the likelier of maj / min on that root."""
    row = CHORD_TABLE[is_minor(k)][(root - tonic(k)) % 12]
    fam = _family(quality)
    p = max(row[0], row[1]) if fam == "sus" else row[FAMILIES.index(fam)]
    return float(np.log(p))


def _chords(segments: list[Segment], cands) -> list[tuple[float, float, int, str, int]]:
    """(t0, t1, root, quality, bass) of the main-layer chords (all chords if none is main)."""
    chosen = [s for s in segments if s.role == "chord" and s.layer == "main"]
    chosen = chosen or [s for s in segments if s.role == "chord"]
    out = []
    for s in sorted(chosen, key=lambda s: s.t0):
        c = cands[s.cand]
        if c.quality != "N" and c.root >= 0:
            out.append((float(s.t0), float(s.t1), int(c.root), c.quality, int(c.bass)))
    return out


def _beat_bounds(grid: RhythmGrid) -> tuple[np.ndarray, np.ndarray]:
    beats = np.asarray(grid.beats, dtype=float)
    ibi = float(grid.ibi_ref) if grid.ibi_ref else (float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5)
    return beats, np.append(beats[1:], beats[-1] + ibi)


def chord_evidence(chords, grid: RhythmGrid) -> np.ndarray:
    """[B,24] per-beat chord log-likelihood: Σ over the chords sounding in the beat of
    (fraction of the beat they cover) × log P(chord | key). Beats without a chord are all 0."""
    beats, ends = _beat_bounds(grid)
    if not chords:
        return np.zeros((len(beats), N_KEYS))
    ll = np.array([[chord_loglik(r, q, k) for k in range(N_KEYS)] for _, _, r, q, _ in chords])
    t0 = np.array([c[0] for c in chords])
    t1 = np.array([c[1] for c in chords])
    ov = np.clip(np.minimum(t1[None, :], ends[:, None]) - np.maximum(t0[None, :], beats[:, None]), 0.0, None)
    return (ov / np.maximum(ends - beats, 1e-6)[:, None]) @ ll


def refine_transitions() -> np.ndarray:
    """[24,24] log transition scores of the chord-key HMM: 0 to stay, −REFINE_SWITCH to change
    key (+ ln FAVOURED for the favoured moves, as in the profile HMM)."""
    log_A = np.full((N_KEYS, N_KEYS), -float(REFINE_SWITCH))
    for k in range(N_KEYS):
        log_A[k, list(favoured_moves(k))] += np.log(FAVOURED)
        log_A[k, k] = 0.0
    return log_A


def _beat_labels(chords, grid: RhythmGrid) -> list[tuple[int, str, int] | None]:
    """The chord (root, quality, bass) sounding at each beat's midpoint (None = no chord)."""
    beats, ends = _beat_bounds(grid)
    out: list[tuple[int, str, int] | None] = []
    j = 0
    for m in 0.5 * (beats + ends):
        while j < len(chords) and chords[j][1] <= m:
            j += 1
        c = chords[j] if j < len(chords) and chords[j][0] <= m < chords[j][1] else None
        out.append(None if c is None else (c[2], c[3], c[4]))
    return out


def _section_starts(grid: RhythmGrid, sections: list[dict] | None) -> list[int]:
    beats = np.asarray(grid.beats, dtype=float)
    B = len(beats)
    if sections:
        starts = {int(np.clip(np.searchsorted(beats, float(s["start"]) - 1e-3), 0, B - 1)) for s in sections}
    elif grid.bars:
        starts = {grid.bars[i][0] for i in range(0, len(grid.bars), BLOCK_BARS)}
    else:
        starts = set(range(0, B, 4 * BLOCK_BARS))
    starts.add(0)
    return sorted(s for s in starts if 0 <= s < B)


def _class_of(cands) -> dict[tuple[int, str, int], int]:
    return {(c.root, c.quality, c.bass): c.cls for c in cands if c.root >= 0}


def _match(a: list, b: list, shift: int, cls: dict) -> tuple[float, int]:
    """Fraction of beats where ``a`` transposed down ``shift`` has the same class as ``b``."""
    hit = tot = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        xs = ((x[0] - shift) % 12, x[1], (x[2] - shift) % 12)
        cx, cy = cls.get(xs, xs), cls.get(y, y)
        tot += 1
        hit += cx == cy
    return (hit / tot if tot else 0.0), tot


def truck_driver(path: np.ndarray, labels: list, grid: RhythmGrid, sections: list[dict] | None,
                 cls: dict) -> tuple[np.ndarray, set[int]]:
    """§5.4 step 4: step-ups of +1/+2 semitones at section boundaries in the last two thirds.

    Returns the new per-beat keys and the beats where a step-up starts.
    """
    path = path.copy()
    beats = np.asarray(grid.beats, dtype=float)
    B = len(beats)
    starts = _section_starts(grid, sections)
    bounds = list(zip(starts, starts[1:] + [B]))
    first_third = beats[0] + (beats[-1] - beats[0]) / 3.0
    steps: set[int] = set()
    for i, (a, b) in enumerate(bounds):
        if i == 0 or beats[a] < first_third:
            continue
        sec = labels[a:b]
        best = (0.0, 0, -1)                    # (match, shift, earlier section index)
        same = 0.0
        for j, (c, d) in enumerate(bounds[:i]):
            m0, n0 = _match(sec, labels[c:d], 0, cls)
            if n0 >= STEP_MIN_BEATS:
                same = max(same, m0)
            for shift in (1, 2):
                m, n = _match(sec, labels[c:d], shift, cls)
                if n >= STEP_MIN_BEATS and m > best[0]:
                    best = (m, shift, j)
        m, shift, j = best
        if m < STEP_MATCH or m <= same:
            continue
        c, d = bounds[j]
        k_from = int(np.bincount(path[c:d], minlength=N_KEYS).argmax())
        k_to = make_key(tonic(k_from) + shift, is_minor(k_from))
        pc0, _ = bounds[i - 1]
        x = a - 1
        while x >= pc0 and path[x] == k_to:
            x -= 1
        if x < pc0:                                     # the previous section is already in k_to
            continue
        path[x + 1:a] = k_from                          # an early HMM change moves to the boundary
        keep = {k_from, k_to, int(path[a - 1])}
        x = a
        while x < B and path[x] in keep:                # up to the next genuine modulation
            path[x] = k_to
            x += 1
        steps.add(a)
    return path, steps


def refine(key: KeyTrack, segments: list[Segment], grid: RhythmGrid, sections: list[dict] | None) -> KeyTrack:
    """§5.4 steps 3–4 on the decoded chords. Returns a new KeyTrack (``key`` is not modified)."""
    from . import vocab                          # lazy: WP6 vocab (§4 interface: candidates().cands)
    path = np.asarray(key.per_beat, dtype=np.int64).copy()
    B = len(path)
    if B == 0 or not segments:
        return key
    cands = vocab.candidates().cands
    chords = _chords(segments, cands)
    if not chords:
        return key

    # step 3: chord-key HMM (chord log-likelihood + a bonus for the global key where the profile
    # agrees), anchored to the global key at both ends: every key change, also one at the start
    # or the end of the song, has to be paid for by the chords
    score = chord_evidence(chords, grid)
    g = make_key(key.global_tonic, key.global_mode == "minor")
    score[path == g, g] += PROFILE_BONUS
    log_A = refine_transitions()
    path = merge_short(viterbi(score, log_A, log_A[g], log_A[:, g]), score)

    # step 4: truck-driver step-ups
    path, steps = truck_driver(path, _beat_labels(chords, grid), grid, sections, _class_of(cands))

    counts = np.bincount(path, minlength=N_KEYS)
    k = int(counts.argmax())
    if counts[g] == counts[k]:
        k = g
    conf = key.global_conf if k == g else float(counts[k] / B)   # new global key: its beat share
    return KeyTrack(global_tonic=tonic(k), global_mode="minor" if is_minor(k) else "major",
                    global_conf=conf, per_beat=path, changes=_changes(path, grid, steps))
