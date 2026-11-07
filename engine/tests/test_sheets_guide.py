"""sheets.guide: the sheet bonus, flags, the analysis-only refine and the diff → ops."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from nathan_chordify import config, schema
from nathan_chordify.server.stub import fixture_paths
from nathan_chordify.sheets import guide as G
from nathan_chordify.theory.vocab import candidates, index_of
from nathan_chordify.types import RhythmGrid


def fixture() -> schema.Analysis:
    return schema.Analysis.model_validate_json(fixture_paths()[0].read_text("utf-8"))


def tiny_grid(n_beats: int = 8, ibi: float = 0.5, sub: int = 2) -> RhythmGrid:
    beats = np.arange(n_beats) * ibi
    G_ = n_beats * sub
    cell_t = np.arange(G_ + 1) * (ibi / sub)
    cell_beat = np.arange(G_ + 1) / sub
    cell_pos = np.array([0 if (g % (4 * sub)) == 0 else (2 if g % sub == 0 else 3) for g in range(G_)], dtype=np.int8)
    return RhythmGrid(beats=beats, beat_conf=np.ones(n_beats), bar_of_beat=np.arange(n_beats) // 4,
                      beat_in_bar=np.arange(n_beats) % 4 + 1, bars=[(0, 4), (4, 4)], meter=[(0, 4, 4)], bpm=60 / ibi,
                      ibi_ref=ibi, subdiv=np.full(n_beats, sub), free=[], cell_t=cell_t, cell_beat=cell_beat,
                      cell_pos=cell_pos)


def test_sheet_bonus_targets_the_family_and_spares_the_edges():
    grid = tiny_grid()
    vocab = candidates()
    bonus = G.sheet_bonus(grid, vocab, [(0.0, 2.0, "C:maj"), (2.0, 4.0, "N")], kappa_b=10.0)
    assert bonus.shape == (16, len(vocab.cands))
    c_maj = index_of(0, "maj", 0)
    c_add9 = index_of(0, "add9", 0)
    c_min = index_of(0, "min", 0)
    g_maj = index_of(7, "maj", 7)
    mid = 3                                                     # cell 0.75–1.0 s: well inside the C chord
    per_beat = 10.0 * 0.5                                       # κ_b × cell length in beats
    assert bonus[mid, c_maj] == pytest.approx(G.W_SHEET * per_beat)
    assert bonus[mid, c_add9] == pytest.approx(G.W_SHEET * per_beat)        # same family (Cadd9 ~ C)
    assert bonus[mid, c_min] == pytest.approx(G.W_SHEET_ROOT * per_beat)    # same root only
    assert bonus[mid, g_maj] == 0.0
    assert bonus[0, c_maj] == pytest.approx(0.5 * G.W_SHEET * per_beat)     # the edge cells get only their inner half
    assert bonus[7, c_maj] == pytest.approx(0.5 * G.W_SHEET * per_beat)
    assert not bonus[8:].any()                                              # N gives nothing


def test_flags_and_agreement():
    main = [(0.0, 2.0, "C:maj"), (2.0, 4.0, "G:maj"), (4.0, 6.0, "A:min")]
    sheet = [(0.0, 2.0, "C:maj(9)"), (2.0, 4.0, "E:min"), (4.0, 5.0, "A:min7"), (5.0, 6.0, "F:maj")]
    flagged, agree = G.flags(sheet, main)
    assert agree == 2 and [f["sheet"] for f in flagged] == ["E:min", "F:maj"]
    assert flagged[0]["detected"] == "G:maj" and flagged[0]["t0"] == 2.0


def test_refine_from_analysis_takes_the_sheet_only_where_unsure():
    a = fixture()
    mains = [c for c in sorted(a.chords, key=lambda c: c.start) if c.layer == "main" and c.harte != "N"]
    sure = next(c for c in mains if c.conf >= 0.95 and c.end - c.start > 1.0)
    unsure = next(c for c in mains if c.conf < 0.6 and c.end - c.start > 0.5)
    other = lambda h: "D:min" if not h.startswith("D:min") else "F:maj"   # noqa: E731
    sheet = [(sure.start, sure.end, other(sure.harte)), (unsure.start, unsure.end, other(unsure.harte))]
    ref = G.refine_from_analysis(a, sheet)
    assert ref.method == "analysis"
    assert [f["t0"] for f in ref.flagged] == [round(sure.start, 3)]          # the sure one is flagged, kept
    changed = [c for c in ref.main if c[2] == other(unsure.harte)]
    assert changed and abs(changed[0][0] - unsure.start) < 1e-6              # the unsure one took the sheet
    ops, n_changed, n_kept = G.diff_ops(G.main_timeline(a), ref.main, "sheet-x", "p")
    assert n_changed == 1 and ops[0]["kind"] == "set_chord" and ops[0]["value"]["group"] == "sheet-x"
    assert ops[0]["target"]["mode"] == "range" and ops[0]["value"]["harte"] == other(unsure.harte)


def test_diff_ops_merges_pieces_and_keeps_equal_chords():
    cur = [(0.0, 1.0, "C:maj"), (1.0, 2.0, "G:maj"), (2.0, 3.0, "A:min"), (3.0, 4.0, "F:maj")]
    target = [(0.0, 1.0, "C:maj"), (1.0, 3.0, "G:maj"), (3.0, 4.0, "F:maj")]
    ops, n_changed, n_kept = G.diff_ops(cur, target, "g", "p")
    assert n_changed == 1 and n_kept == 2
    assert ops[0]["target"]["t0"] == 2.0 and ops[0]["target"]["t1"] == 3.0 and ops[0]["value"]["harte"] == "G:maj"
    # a hole in the current layer is filled
    ops2, n2, _ = G.diff_ops([(0.0, 1.0, "C:maj")], [(0.0, 2.0, "C:maj")], "g", "p")
    assert n2 == 1 and ops2[0]["target"]["t0"] == 1.0
    # tiny boundary jitter is ignored
    ops3, n3, k3 = G.diff_ops([(0.0, 1.0, "C:maj"), (1.0, 2.0, "G:maj")], [(0.0, 1.05, "C:maj"), (1.05, 2.0, "G:maj")], "g", "p")
    assert n3 == 0 and k3 == 2


def test_sheet_and_section_ops_and_full_label():
    ops = G.sheet_ops([(0.0, 1.0, "B:maj"), (1.0, 2.0, "N"), (2.0, 3.0, "F#:maj/5")], "sheet-a", "p", 0.75)
    assert [o["value"]["harte"] for o in ops] == ["B:maj", "N", "F#:maj/5"] and all(o["value"]["conf"] == 0.75 for o in ops)
    sec = G.section_ops([(0.0, "Verse 1"), (8.0, "Chorus")], "sheet-a", "p")
    assert sec[1]["kind"] == "set_section" and sec[1]["value"] == {"label": "Chorus", "group": "sheet-a"}
    assert G.full_label("F#:maj/5", 11) == "F#/C#" and G.full_label("N", 11) == "N"


REAL = config.CACHE_DIR / "yt-WLtqi4s29LQ"


@pytest.mark.skipif(not (REAL / "decode_state.v1.npz").exists() or os.environ.get("NCHORD_SKIP_REAL") == "1",
                    reason="needs the real cached decode state of 立志擺上")
def test_refine_state_on_the_real_cache_keeps_or_raises_agreement():
    """Sanity check (a)/(b) in miniature: guided by its own GT chart the decode agrees at least as much
    with eval/gt as unguided; flagged chords are a minority."""
    from nathan_chordify import pipeline
    from nathan_chordify.sheets import align as A
    gt_lab = Path(config.EVAL_DIR) / "gt" / "yt-WLtqi4s29LQ.lab"
    if not gt_lab.exists():
        pytest.skip("no eval/gt lab")
    rows = [ln.split("\t") for ln in gt_lab.read_text("utf-8").splitlines() if ln.strip()]
    gt = [(float(a), float(b), h) for a, b, h in rows if h not in ("N", "X")]
    st = pipeline.load_state(REAL)
    audio = A.audio_from_state(st)
    text = (Path(config.EVAL_DIR) / "gt" / "yt-WLtqi4s29LQ.chart.txt").read_text("utf-8")
    al = A.align_chart_text(text, audio, 4)
    assert al.verdict == "good" and al.transpose == 0
    run, segs, ev, key = pipeline.decode_state(st, pipeline.params_from(None))
    a0, _ = pipeline._assemble(st, pipeline.params_from(None), "full", run, segs, ev, key, {})
    ref = G.refine_state(st, al.chords)

    def root_acc(main):
        n = ok = 0
        for t0, t1, h in gt:
            for tm in np.arange(t0 + 0.25, t1, 0.5):
                est = next((d for a, b, d in main if a <= tm < b), "N")
                n += 1
                ok += G._triple(h)[0] == G._triple(est)[0]
        return ok / n
    assert root_acc(ref.main) >= root_acc(G.main_timeline(a0)) - 1e-9
    assert len(ref.flagged) < 0.3 * ref.n_sheet and ref.method == "state"
