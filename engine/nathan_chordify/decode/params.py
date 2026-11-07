"""Decoder parameters (DESIGN §4 ``DecoderParams``) and the fixed constants of §5.9–5.12.

``DecoderParams`` itself lives in ``types.py`` (WP0 contract) and is re-exported here. The
constants below are the numbers §5.9–5.12 write inline; they are not tuned in M2 and the
``DecoderParams`` contract has no fields for them (see the WP7 report).
"""
from __future__ import annotations

from dataclasses import replace

from ..types import DecoderParams

__all__ = ["DecoderParams", "fast_rescan", "default"]

# --- §5.9 fusion -------------------------------------------------------------------------------
HOP_S = 256 / 22050                 # the 11.6 ms evidence timeline (§5.6, §5.9)
INPUT_WEIGHTS = {"harmonic": 0.6, "mix": 0.4}   # one source per model: p = 0.6·p@harm + 0.4·p@mix
MODEL_BASS_WEIGHT = 0.5             # a model's own bass head counts half (§5.6, review A3)
EXT_FLOOR = 1e-4                    # floor of the lv extension heads and the ACE note terms
BAS_MIX = 0.9                       # bas(g, b) = log(0.9·P_bass + 0.1/12)
BASS_REL_DB = 24.0                  # bas activity: soft step this far below the song's loud bass level
BASS_REL_WIDTH_DB = 3.0             # … with this width (logistic scale, dB)
DEFAULT_TEMP = 1.0                  # T_s of a source that has no fitted temperature
SYM_ACT_FULL = 2.0                  # sym activity = min(1, Σ_pc x / this): two notes' presence = fully active (M5)
#: Id of the fitted temperatures that ship as ``DecoderParams.temp`` defaults (written to
#: analysis.json "calibration"; details in models/calibration/<id>.json and eval/reports/M2.md).
CALIBRATION_ID = "cal-m2-0001"

# --- §5.10 decoding ----------------------------------------------------------------------------
T_SELF = -0.3                       # re-entry of the same candidate (merged later)
# Forward–backward temperature (the Viterbi path does not depend on it). M2 on mini-D2: 2.0 brings
# the cell-level ECE from 0.165 to 0.085 (accuracy unchanged), but makes passing-chord confidence
# useless for the Detail slider (AUC 0.59, precision FALLS with conf), so 1.0 ships; see M2.md.
POST_TEMP = 1.0
STEP_BASS_BONUS = 0.6               # 1 ≤ bass distance ≤ 2 semitones
HCDF_BONUS = 0.5                    # an HCDF peak within ±60 ms of the boundary
HCDF_WINDOW_S = 0.06
SHORT_BEATS = 0.5                   # segments ≤ ½ beat score with E_short
PT_MAX_BEATS = 1.0                  # the PT bonus applies to segments ≤ 1 beat
LMAX_BARS = 4                       # Lmax = 4 bars of cells (64 at 4/4, subdiv 4)
LMAX_FREE_S = 16.0                  # … or 16 s in free time
FREE_BEAT_CELLS = 3                 # main layer: 3 free-time cells (0.6 s) act as a beat

# --- §5.11 layers ------------------------------------------------------------------------------
PASSING_MAX_BEATS = 2.0             # a disagreeing detail run ≤ 2 beats is a passing candidate
PROMOTE_MARGIN = 0.10               # long disagreement: promote when class posterior wins by 0.10
RICHER_COVER = 0.5                  # a richer same-chord detail run relabels M when it covers ≥ 50 %
MIN_DETAIL_RANGE = (0.05, 1.0)

# --- §5.12 post-processing ---------------------------------------------------------------------
BASS_EVENT_MAX_BEATS = 1.0          # step 1: main-layer segments ≤ 1 beat (passing ones ≤ 2), shorter than N (M2)
PHANTOM_MAX_BEATS = 0.5
APA_MAX_BEATS = 1.0
SUPPORT_MIN = 0.5                   # sym presence / lv@harmonic tone presence for added tones
ONSET_MIN = 0.1                     # an onset feature (dch) above this counts as a new onset
N_ALTERNATIVES = 3


def default() -> DecoderParams:
    """A fresh ``DecoderParams`` with the §4 defaults."""
    return DecoderParams()


def fast_rescan(p: DecoderParams) -> DecoderParams:
    """§5.11 "Re-scan this section (more sensitive)": λ_scale × 0.6, w_prior × 0.5, PT bonus +1.5.

    The region (selection ± 2 bars) and the fixed boundary chords are passed to
    ``hsmm.decode(region=…, boundary=…)``; ``hsmm.rescan_region`` computes the cell range.
    """
    return replace(p, lam_scale=p.lam_scale * 0.6, w_prior=p.w_prior * 0.5, pt_bonus=1.5,
                   temp=dict(p.temp), w_src=dict(p.w_src), lam=dict(p.lam))
