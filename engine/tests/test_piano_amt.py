"""Piano transcription (DESIGN §5.7) and the left-hand bass fallback (§5.6), M5.

Unit tests on synthetic notes (pedal offsets, the §5.7 filters, the notes.npz cache, the
left-hand "lowest sounding note" line) and one smoke test of the real network on a short excerpt
of 回家's piano stem (skipped without CUDA, the checkpoint or the cached stem).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

from nathan_chordify import config
from nathan_chordify.notes import bass as nb
from nathan_chordify.notes import piano_amt as amt
from nathan_chordify.types import BassTrack, NoteEvents

HUIJIA_OTHER = config.CACHE_DIR / "yt-Y7boUFbM76k" / "stems" / "other.flac"


def ev(rows, pedals=()) -> NoteEvents:
    return amt.to_events(np.array(rows, dtype=float).reshape(-1, 4), np.array(pedals, dtype=float).reshape(-1, 2))


# ---------------------------------------------------------------------------------------------
# pedal offsets and filters
# ---------------------------------------------------------------------------------------------

def test_pedal_extends_a_key_released_while_the_pedal_is_down():
    ne = ev([(0.0, 0.5, 60, 80),          # released at 0.5, pedal down 0.2–2.0 → 2.0
             (2.5, 2.7, 62, 80),          # released before the next pedal → unchanged
             (3.0, 3.4, 64, 80)],         # pedal pressed at 3.5, after the release → unchanged
            pedals=[(0.2, 2.0), (3.5, 4.0)])
    assert ne.offset_pedal == pytest.approx([2.0, 2.7, 3.4])


def test_pedal_extension_stops_at_the_next_strike_of_the_same_key():
    ne = ev([(0.0, 0.3, 60, 80), (1.0, 1.3, 60, 80), (1.0, 1.3, 64, 80)], pedals=[(0.0, 3.0)])
    assert ne.offset_pedal == pytest.approx([1.0, 3.0, 3.0])
    assert np.all(ne.offset_pedal >= ne.offset)


def test_filters_velocity_duration_and_range():
    ne = ev([(0.0, 1.0, 60, 19),          # too soft
             (0.0, 0.05, 60, 80),         # 50 ms: too short
             (0.0, 1.0, 32, 80),          # below A1 (33)
             (0.0, 1.0, 97, 80),          # above C7 (96)
             (0.0, 0.06, 33, 20),         # every bound inclusive
             (0.0, 1.0, 96, 127)])
    f = amt.filter_notes(ne)
    assert list(f.pitch) == [33, 96]
    assert f.source == "piano_amt@other"


def test_cache_round_trip(tmp_path):
    notes = np.array([[0.5, 0.9, 60, 70], [0.1, 0.4, 40, 10]], dtype=float)
    amt.save_cache(tmp_path, notes, np.array([[0.0, 1.0]]), {"input": "other.flac"})
    raw = amt.load_cached(tmp_path, raw=True)
    assert list(raw.pitch) == [60, 40] and raw.offset_pedal == pytest.approx([1.0, 1.0])   # both released under the pedal
    kept = amt.load_cached(tmp_path)
    assert list(kept.pitch) == [60]                                  # velocity 10 is filtered
    assert amt.load_cached(tmp_path / "missing") is None
    assert amt.song_dir(tmp_path / "k" / "stems" / "other.flac") == (tmp_path / "k").resolve()


def test_missing_checkpoint_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(amt, "CHECKPOINT", tmp_path / "nope.pth")
    with pytest.raises(FileNotFoundError, match="fetch_models.py --all"):
        amt.checkpoint()


# ---------------------------------------------------------------------------------------------
# left hand (§5.6)
# ---------------------------------------------------------------------------------------------

def test_left_hand_line_is_the_lowest_sounding_note():
    # B1 held under the pedal while the left hand arpeggiates F#2, B2, D#3 above it: the bass stays B;
    # then E1 (lower) takes over at once
    ne = ev([(0.0, 0.4, 35, 90), (0.5, 0.7, 42, 70), (1.0, 1.2, 47, 70), (1.5, 1.7, 51, 70),
             (2.0, 2.4, 28, 90)], pedals=[(0.1, 1.95), (2.1, 3.0)])
    line = nb.left_hand_line(ne)
    assert [(round(a, 2), round(b, 2), m) for a, b, m, _ in line] == [(0.0, 1.95, 35), (2.0, 3.0, 28)]
    assert line[0][3] == pytest.approx(90 / 127)


def test_left_hand_line_follows_the_next_note_when_the_root_is_released():
    ne = ev([(0.0, 0.5, 35, 90), (0.2, 1.0, 42, 60)])                # no pedal: B1 released at 0.5
    line = nb.left_hand_line(ne)
    assert [(a, b, m) for a, b, m, _ in line] == [(0.0, 0.5, 35), (0.5, 1.0, 42)]
    assert line[1][3] == pytest.approx(0.47, abs=0.01)


def test_a_long_pedal_does_not_hold_the_old_root_across_a_chord_change():
    # B1 under an 8 s pedal; the next chord's root C#2 (higher) is struck 3.7 s later
    ne = ev([(0.0, 0.3, 35, 90), (1.0, 1.2, 42, 60), (3.7, 4.0, 37, 90)], pedals=[(0.0, 8.0)])
    line = nb.left_hand_line(ne)
    assert [(round(a, 2), m) for a, _, m, _ in line] == [(0.0, 35), (3.7, 37)]


def test_left_hand_only_where_the_bass_stem_is_quiet():
    hop = nb.HOP / nb.SR
    ft = np.arange(int(6.0 / hop)) * hop
    e = np.where(ft < 3.0, -20.0, -80.0)                              # bass stem plays 0–3 s, then silence
    bt = BassTrack(frame_t=ft, salience=np.full((len(ft), 12), 1 / 12), energy_db=e,
                   notes=[(0.0, 2.9, 40, 0.9, "bass_stem"), (4.5, 5.0, 43, 0.4, "bass_stem")])  # 2nd = bleed
    ne = ev([(1.0, 1.5, 33, 90), (4.0, 5.5, 38, 90)])
    out = nb.with_left_hand(bt, ne)
    assert bt.notes[1][4] == "bass_stem" and len(bt.notes) == 2       # the stem track is not modified
    srcs = [(round(a, 1), m, s) for a, _, m, _, s in out.notes]
    assert (0.0, 40, "bass_stem") in srcs and (4.0, 38, "amt_lh") in srcs
    assert all(not (s == "amt_lh" and a < 3.0) for a, _, s in srcs)  # loud stem: the left hand is ignored
    assert all(not (s == "bass_stem" and a > 4.0) for a, _, s in srcs)  # quiet stem: its notes are dropped
    assert nb.with_left_hand(bt, None) is bt


# ---------------------------------------------------------------------------------------------
# smoke test on real audio (GPU)
# ---------------------------------------------------------------------------------------------

def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:          # pragma: no cover
        return False


@pytest.mark.skipif(not _cuda(), reason="needs CUDA")
@pytest.mark.skipif(not amt.CHECKPOINT.exists(), reason="run scripts/fetch_models.py --all")
@pytest.mark.skipif(not HUIJIA_OTHER.exists(), reason="回家's stems are not cached")
def test_amt_smoke_on_a_real_piano_excerpt(tmp_path):
    y, sr = sf.read(str(HUIJIA_OTHER), dtype="float32", always_2d=True, start=60 * 44100, stop=70 * 44100)
    wav = tmp_path / "excerpt.flac"
    sf.write(str(wav), y, sr)
    ne = amt.transcribe(wav, "cuda", cache_dir=tmp_path)
    assert (tmp_path / amt.CACHE_NAME).exists()
    assert len(ne.onset) >= 10                                       # 回家 60–70 s: piano chords B → B+ → E
    assert np.all((ne.pitch >= amt.PITCH_LO) & (ne.pitch <= amt.PITCH_HI))
    assert np.all((ne.onset >= 0) & (ne.onset <= 10.0)) and np.all(ne.offset_pedal >= ne.offset)
    assert np.all(ne.velocity >= amt.MIN_VELOCITY)
    pcs = np.bincount(ne.pitch % 12, minlength=12)
    assert pcs[[11, 3, 6, 4, 8]].sum() > 0.5 * pcs.sum()               # mostly B, D#, F#, E, G# (B major)
    t = time.perf_counter()
    again = amt.transcribe(wav, "cuda", cache_dir=tmp_path)          # served from notes.npz
    assert time.perf_counter() - t < 1.0 and np.array_equal(again.pitch, ne.pitch)
