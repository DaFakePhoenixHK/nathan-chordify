"""Core internal types and interfaces (DESIGN §4). Owned by WP0.

Every dataclass here is a shared contract between work packages: field names, types and
defaults follow DESIGN.md v1.1 §4 exactly. Change them only through a contract-change request.

The chart types at the end (``ChartChord``, ``ChartSection``, ``Chart``) are the result of
``theory.chart.parse`` for the bar-text format of DESIGN §0.1.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np

Progress = Callable[[str, float, str], None]     # (stage, pct 0..1, message)


@dataclass(frozen=True)
class AudioAsset:
    key: str                  # "yt-Y7boUFbM76k" | "sha256-<hex64>" | "url-<extractor>-<id>"
    dir: Path                 # cache/<key>/
    wav: Path                 # cache/<key>/audio.wav  (44.1 kHz, stereo, PCM_16; evictable, re-decoded from source.*)
    play: Path                # cache/<key>/play.m4a   (AAC 192k, faststart) for the web player
    duration_s: float
    title: str
    artist: str | None
    source_type: Literal["youtube", "url", "upload"]
    source_url: str | None


@dataclass(frozen=True)
class Stems:
    mix: Path                 # stems/*.flac (16-bit FLAC)
    vocals: Path
    drums: Path
    bass: Path
    other: Path
    harmonic: Path            # bass + other (no vocals, no drums)
    model: str                # "htdemucs_ft" | "none" (debug fast mode: harmonic = mix, bass = low-passed mix)


@dataclass
class RhythmGrid:
    beats: np.ndarray         # [B] seconds
    beat_conf: np.ndarray     # [B]
    bar_of_beat: np.ndarray   # [B] int (−1 for pickup beats and free-time beats)
    beat_in_bar: np.ndarray   # [B] int, 1-based (0 in free time)
    bars: list[tuple[int, int]]          # (first_beat_idx, beats_in_bar); mixed meters allowed
    meter: list[tuple[int, int, int]]    # (from_bar, numerator, denominator)
    bpm: float
    ibi_ref: float            # median inter-beat interval (s); scales evidence in free-time cells
    subdiv: np.ndarray        # [B] cells per beat ∈ {2, 3, 4, 6}, constant within a section (§5.3)
    free: list[tuple[float, float]]      # free-time regions (s), drawn unmetered
    cell_t: np.ndarray        # [G+1] cell boundaries in seconds
    cell_beat: np.ndarray     # [G+1] boundaries in beat units (beat index + fraction; NaN inside free time)
    cell_pos: np.ndarray      # [G] int8: 0 downbeat, 1 mid-bar beat, 2 beat, 3 half/third, 4 quarter/sixth, 5 free time
    edited: bool = False      # True when user grid edits were applied


@dataclass
class RhythmOverrides:        # built from the edit ops (§5.13); rhythm.track re-applies them
    tempo_factor: float = 1.0                                       # 0.5 | 1 | 2
    downbeat_shift: int = 0
    moved_beats: dict[int, float] = field(default_factory=dict)     # beat idx → new time (s)
    tapped_beats: list[float] | None = None                         # replaces the tracker (tap tempo, GT beats)
    meter: list[tuple[int, int, int]] = field(default_factory=list) # (from_bar, num, den)
    subdiv: list[tuple[int, int]] = field(default_factory=list)     # (from_bar, cells per beat)
    free: list[tuple[float, float]] = field(default_factory=list)   # forced free-time regions


@dataclass
class ChordEvidence:          # one model on one input, at the model's native hop
    name: str                 # "lv@mix", "lv@harmonic", "ace@mix", "ace@harmonic", "btc@harmonic"
    frame_t: np.ndarray       # [T] frame centre times (s), already corrected by offset_s
    offset_s: float           # measured adapter latency (§5.5)
    root: np.ndarray | None   # [T,13] probs (12 pcs + N)
    bass: np.ndarray | None   # [T,13]
    notes: np.ndarray | None  # [T,12] independent probabilities
    heads: dict[str, np.ndarray]           # model-specific heads, e.g. lv: triad/seventh/ninth/...
    head_labels: dict[str, list[str]]
    label_probs: np.ndarray | None         # [T,V] for label classifiers (BTC-CL)
    labels: list[str] | None               # Harte labels of label_probs columns


@dataclass
class NoteEvents:
    onset: np.ndarray         # [N]
    offset: np.ndarray        # [N]
    pitch: np.ndarray         # [N]
    velocity: np.ndarray      # [N]
    offset_pedal: np.ndarray  # [N] offset extended to the sustain-pedal release
    source: str               # "piano_amt@other"


@dataclass
class BassTrack:
    frame_t: np.ndarray       # [T] (11.6 ms hop)
    salience: np.ndarray      # [T,12] pitch-class salience, rows sum to 1
    energy_db: np.ndarray     # [T]
    notes: list[tuple[float, float, int, float, str]]  # (t0, t1, midi, conf, "bass_stem" | "amt_lh"), latency-compensated


@dataclass
class KeyChange:
    t: float
    bar: int
    from_key: int             # 0..23 as in KeyTrack.per_beat
    to_key: int
    kind: Literal["step_up", "modulation", "relative", "parallel"]


@dataclass
class KeyTrack:
    global_tonic: int
    global_mode: Literal["major", "minor"]
    global_conf: float
    per_beat: np.ndarray      # [B] int 0..23 (0–11 major tonic pc, 12–23 minor)
    changes: list[KeyChange]


@dataclass(frozen=True)
class Candidate:
    root: int                 # pcs 0..11; N is (-1, "N", -1)
    quality: str              # quality from §5.8
    bass: int
    pcs: frozenset[int]       # template tones (required + optional) ∪ {bass}
    cls: int                  # equivalence-class id (§5.8)


@dataclass
class Vocab:
    cands: list[Candidate]
    tau: np.ndarray           # [V,12] template probabilities (§5.9, sym)
    classes: list[list[int]]  # class id → member candidate indices


@dataclass
class SymParams:              # §5.7
    tau: float = 0.4
    beta: float = 0.5
    alpha: float = 0.7
    theta: float = 0.25
    split_midi: int = 55
    min_velocity: int = 20
    min_dur_s: float = 0.06


@dataclass
class DecoderParams:          # decode/params.py; every constant of §5.9–5.12 lives here
    kappa_b: float = 10.0                                   # evidence per beat; FIXED while the rest is tuned
    # T_s per source (decode/calibrate.py); M2 calibration cal-m2-0001: lv = NLL fit on mini-D2,
    # bas = real-audio midpoint (the D2 fit 0.39 does not transfer) — models/calibration/cal-m2-0001.json
    temp: dict[str, float] = field(default_factory=lambda: {"lv": 0.7069, "bas": 1.0, "ace": 0.5801, "sym": 0.597})  # ace: WP4b NLL fit on mini-D2 (eval/ace_eval.py fit); sym: M5 NLL fit on mini-D2 (activity-weighted; eval/reports/m5_amt_numbers.json)
    clip: float = -6.0
    w_src: dict[str, float] = field(default_factory=lambda: {"lv": 1.0, "ace": 1.0, "btc": 0.5, "sym": 1.0, "bas": 1.68})
    w_short: float = 0.3      # M2 tuning (mini-D2): 1.0 let the fast bass outvote lv in ½-beat segments
    lam: dict[int, float] = field(default_factory=lambda: {0: 2.0, 1: 2.5, 2: 3.0, 3: 4.5, 4: 6.0, 5: 4.0})
    lam_scale: float = 0.76   # §5.9 break-even 0.70 (half-beat A–P–B); M2 tuning is flat in 0.5–0.8
    lam_main_mult: float = 1.5
    w_prior: float = 0.6      # M2 tuning (mini-D2 wants ~0.3, 回家's checklist ~1; equal D2 score at 0.6)
    prior_cap: float = 3.0
    w_dur: float = 1.0
    dur_median_beats: float = 2.0
    dur_sigma: float = 0.9
    pt_bonus: float = 1.0
    rep_bonus: float = 1.0
    slider: tuple[float, float, float] = (-0.594, 0.0, 1.768)  # (a, b, c) of min_detail (§5.11), M2 fit: confidence only
    demote: float = 0.3
    prune_top: int = 10
    prune_cap: int = 250
    bass_event_max_s: float = 0.3
    # M5 piano transcription (§5.6, §5.7; pipeline.py). Both need cache/<key>/notes.npz (stems/other.flac
    # transcribed; analyze runs the AMT only when one of them is on); without notes they do nothing.
    # use_sym: the symbolic chroma as the "sym" source, its onsets (dch) for the phantom rule and its
    # presence for the support tests. amt_bass: the left-hand bass fallback where the bass stem is quiet.
    # Both OFF: they failed the §13.6 M5 component gate on top of lv+ACE (mini-D2 short-segment F1
    # +0.013 at best, < 0.02; 回家 checklist no gain, more false positives) — eval/reports/m5_amt_numbers.json.
    use_sym: bool = False
    amt_bass: bool = False


@dataclass
class Evidence:
    C: np.ndarray             # [K] candidate indices (the pruned set, §5.8)
    E: np.ndarray             # [G,K] nats
    E_short: np.ndarray       # [G,K] the same with the fast sources up-weighted (w_short); used by segments ≤ ½ beat
    parts: dict[str, np.ndarray]   # calibrated ℓ_s [G,K] per source (debugging, and "sources" in the JSON)


@dataclass
class Segment:
    c0: int                   # cells [c0, c1)
    c1: int
    t0: float
    t1: float
    cand: int                 # display candidate (index into Vocab.cands)
    cls: int
    layer: Literal["main", "passing"]
    role: Literal["chord", "bass_event"] = "chord"
    conf: float = 0.0         # class-level mean posterior
    salience: float | None = None
    min_detail: float | None = None
    alternatives: list[tuple[int, float]] = field(default_factory=list)   # (cand, class conf), other classes
    suggestion: bool = False
    edited: bool = False


@dataclass
class DecodeResult:
    segments: list[Segment]
    post: np.ndarray          # [G,K] cell marginals
    post_cls: np.ndarray      # [G,n_cls] class marginals (sum over members)
    pb: np.ndarray            # [G] boundary probability
    score: float              # Viterbi path score


@dataclass
class EditOp:                 # one line of cache/<key>/edits.jsonl (append-only, §5.13)
    id: str                   # uuid4 made by the client; idempotency key
    seq: int                  # assigned by the server on append
    ts: str                   # ISO 8601
    device: str               # "desktop" | "phone-<n>"
    kind: Literal["set_chord", "delete_change", "insert_rest", "accept_suggestion", "reject_suggestion",
                  "set_meter", "set_subdiv", "set_free_time", "shift_downbeat", "move_beat",
                  "half_double_tempo", "tap_beats", "set_section", "set_key", "revert"]
    target: dict              # {"chord_id": …} | {"t0": …, "t1": …} | {"bar": …} | {"op_id": …}
    value: dict               # e.g. {"harte": "B:aug(b7)"}


# ---------------------------------------------------------------------------------------------
# Bar-text charts (DESIGN §0.1; theory/chart.py)
# ---------------------------------------------------------------------------------------------

class ChartChord(NamedTuple):
    """One chord change inside a bar: a ``(beat_pos, harte)`` pair.

    ``pos`` is the onset in beats from the start of the bar (0-based; fractions come from
    bracketed beats such as ``(A7 Em6)``), or None for a bar written without dots that is split
    evenly over however many beats the bar turns out to have (``| B E |``).
    """
    pos: float | None
    harte: str                # canonical Harte label ("N" for no chord)


@dataclass
class ChartSection:
    label: str | None         # "Verse 1", or None for bars before the first label
    bars: list[list[ChartChord]] = field(default_factory=list)
    unsure: set[tuple[int, int]] = field(default_factory=set)   # (bar idx in section, chord idx in bar) written with "?"


@dataclass
class Chart:
    sections: list[ChartSection] = field(default_factory=list)
    beats_per_bar: int = 4

    @property
    def bars(self) -> list[list[ChartChord]]:
        """All bars of all sections, in order."""
        return [bar for s in self.sections for bar in s.bars]
