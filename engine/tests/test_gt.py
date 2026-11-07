"""M4 Ground-truth mode (DESIGN §8.4, §11, §13.1; review E1/E3): server/gt.py + theory.chart.align.

Chart import → alignment on tapped beats → draft edit layer → review → eval/gt export (+ songs.yaml),
and the exported chart read back gives the same chords.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from nathan_chordify import config, schema
from nathan_chordify.server import edits as ED
from nathan_chordify.theory import chart as CH

from test_server import client, env, put_cached_song  # noqa: F401  (env is a fixture)

KEY = "yt-AAAAAAAAAAA"
IBI = 60 / 65.4

CHART = """[Verse 1]
| B . . (. B+7) | Emaj7 . A7 Em6 |
| C#m7 D#m7 | C#m7/E Em/G? |
[Chorus]   (M)
| G#m F# | E F# B |
| E/F# . . . | B |
"""


def tapped(start=2.95, n=60, jitter=0.0):
    rng = np.random.default_rng(1)
    return [round(start + i * IBI + (rng.normal(0, jitter) if jitter else 0.0), 4) for i in range(n)]


def gt_chords(c) -> list[schema.ChordOut]:
    from nathan_chordify.server.gt import gt_chords as G
    return G(schema.Analysis.model_validate(c.get(f"/api/tracks/{KEY}").json()))


# ---- theory.chart.align -------------------------------------------------------------------------

def test_align_honours_dots_brackets_even_splits_and_holds():
    ch = CH.parse("| B . . (. B+7) | Emaj7 . A7 Em6 | C#m7 D#m7 | . | % |")
    beats = 10.0 + np.arange(24) * 0.5
    al = CH.align_chart(ch, beats)
    got = [(a.harte, a.t0, a.t1) for a in al]
    assert got[:6] == [("B:maj", 10.0, 11.75), ("B:aug(b7)", 11.75, 12.0), ("E:maj7", 12.0, 13.0),
                       ("A:7", 13.0, 13.5), ("E:min6", 13.5, 14.0), ("C#:min7", 14.0, 15.0)]
    assert got[6:] == [("D#:min7", 15.0, 20.0)]             # held bar (and % of it): D#m7 runs on
    al = CH.align_chart(CH.parse("| C#m7 D#m7 | % |"), beats)
    assert [(a.harte, a.t0, a.t1) for a in al] == [("C#:min7", 10.0, 11.0), ("D#:min7", 11.0, 12.0),
                                                   ("C#:min7", 12.0, 13.0), ("D#:min7", 13.0, 14.0)]
    ch = CH.parse("| B . . (. B+7) | Emaj7 . A7 Em6 |")
    segs = CH.align(ch, beats)
    assert len(segs) == 5 and segs[1].c0 == 3 and segs[1].c1 == 4 and segs[0].conf == 1.0
    assert all(s.cand >= 0 for s in segs)


def test_align_with_start_anchors_unsure_and_uneven_beats():
    text = "[Intro @3.0]\n| B | E? |   (L)\n[Verse @12]\n| G#m F# |\n"
    ac = CH.parse_annotated(text)
    assert ac.anchors == [3.0, 12.0] and [s.label for s in ac.chart.sections] == ["Intro", "Verse"]
    assert ac.bar_conf == ["L", "L", "H"]
    beats = np.cumsum(np.r_[1.0, np.full(30, 0.9) + np.tile([0.05, -0.05], 15)])   # wobbly tapped beats
    al = CH.align_chart(ac.chart, beats, anchors=ac.anchors, bar_conf=ac.bar_conf)
    assert al[0].t0 == pytest.approx(beats[int(np.argmin(abs(beats - 3.0)))])
    assert al[1].unsure and al[1].conf == "L"
    assert al[1].t1 == pytest.approx(beats[int(np.argmin(abs(beats - 3.0))) + 8])   # ends at its section's end
    v0 = int(np.argmin(abs(beats - 12)))
    assert (al[2].t0, al[3].t0) == (pytest.approx(beats[v0]), pytest.approx(beats[v0 + 2]))


def test_chart_from_times_round_trip():
    beats = 2.0 + np.arange(40) * 0.75
    ch = CH.parse(CHART)
    al = CH.align_chart(ch, beats)
    lay = CH.layout(ch, beats)
    back = CH.chart_from_times([(a.t0, a.t1, a.harte, a.unsure) for a in al], beats, lay.bar_start, 4,
                               lay.section_of_bar, [s.label for s in ch.sections])
    al2 = CH.align_chart(CH.parse(CH.format(back)), beats)
    assert [(a.harte, a.t0, a.t1, a.unsure) for a in al2] == [(a.harte, a.t0, a.t1, a.unsure) for a in al]


# ---- routes -------------------------------------------------------------------------------------

def test_gt_chart_import_review_export_round_trip(env):
    put_cached_song(KEY)
    eval_dir = config.EVAL_DIR
    with client() as c:
        st = c.get(f"/api/tracks/{KEY}/gt").json()
        assert st["status"] == "none" and st["beats"] == [] and st["chart"] is None
        # tapped beats (independent of our grid), stored sorted with near-duplicates merged
        beats = tapped(jitter=0.01)
        r = c.put(f"/api/tracks/{KEY}/gt/beats", json={"beats": beats[::-1] + [beats[5] + 0.02]})
        assert r.status_code == 200 and r.json()["beats"] == sorted(beats) and r.json()["status"] == "draft"
        assert c.put(f"/api/tracks/{KEY}/gt/beats", json={"beats": [1.0, 1e6]}).status_code == 400
        # paste the chart: aligned to the tapped beats, one edit-op group
        r = c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": CHART, "import_id": "imp1", "device": "desktop"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_chords"] == 16 and body["gt"]["chart"]["source"] == "tapped"
        assert [u["harte"] for u in body["unsure"]] == ["E:min/b3"]
        a = schema.Analysis.model_validate(body["analysis"])
        assert a.gt.status == "draft" and a.gt.tapped_beats and a.gt.chart == CHART
        g = [ch for ch in a.chords if ch.edited]
        assert g[0].start == pytest.approx(beats[0]) and g[0].harte == "B:maj"
        assert g[1].harte == "B:aug(b7)" and g[1].start == pytest.approx(beats[3] + 0.5 * (beats[4] - beats[3]), abs=1e-3)
        em = next(ch for ch in g if ch.harte.startswith("E:min/"))
        assert em.conf == pytest.approx(0.35) and em.label.full == "Em/G"
        assert next(ch for ch in g if ch.harte == "G#:min").conf == pytest.approx(0.75)   # an (M) line
        assert [s.label for s in a.sections] == ["Verse 1", "Chorus"]
        # GT timeline = main layer + Nathan's chords; no passing guesses inside the chart
        seq = [ch for ch in gt_chords(c) if beats[0] - 1e-3 <= ch.start < beats[32] - 1e-3]
        assert [ch.harte for ch in seq] == ["B:maj", "B:aug(b7)", "E:maj7", "A:7", "E:min6", "C#:min7", "D#:min7",
                                            "C#:min7/b3", "E:min/b3", "G#:min", "F#:maj", "E:maj", "F#:maj", "B:maj",
                                            "E:maj/2", "B:maj"]
        assert seq[-1].end == pytest.approx(beats[32], abs=1e-3)

        # a retried import (same id) appends nothing
        rev = body["rev"]
        r = c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": CHART, "import_id": "imp1", "replace": False})
        assert r.json()["rev"] == rev and r.json()["appended"] == 0

        # correct one chord by ear, then review → files + songs.yaml
        b7 = next(ch for ch in a.chords if ch.harte == "B:aug(b7)")
        c.post(f"/api/tracks/{KEY}/edits", json={"ops": [{"id": "fix1", "kind": "set_chord", "target": {
            "chord_id": b7.id, "t0": b7.start, "t1": b7.end, "layer": b7.layer}, "value": {"text": "B7"}}]})
        r = c.post(f"/api/tracks/{KEY}/gt/review", json={"reviewed": True})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["reviewed"] and out["files"]["lab"] == f"eval/gt/{KEY}.lab"
        lab = (eval_dir / "gt" / f"{KEY}.lab").read_text("utf-8").splitlines()
        rows = [ln.split("\t") for ln in lab]
        assert all(float(x[0]) < float(x[1]) for x in rows)
        assert any(x[2] == "B:7" and float(x[0]) == pytest.approx(b7.start, abs=1e-3) for x in rows)
        assert not any(x[2] == "B:aug(b7)" for x in rows)
        bt = np.loadtxt(eval_dir / "gt" / f"{KEY}.beats", comments="#", ndmin=2)
        assert np.allclose(bt[:, 0], sorted(beats), atol=1e-3)
        assert list(bt[:8, 1]) == [1, 2, 3, 4, 1, 2, 3, 4]
        chart_txt = (eval_dir / "gt" / f"{KEY}.chart.txt").read_text("utf-8")
        assert "REVIEWED" in chart_txt and "[Verse 1 @" in chart_txt and "Em/G?" in chart_txt
        # the exported chart, laid on the same beats, gives back the GT chords
        ac = CH.parse_annotated(chart_txt)
        back = CH.align_chart(ac.chart, np.asarray(sorted(beats)), anchors=ac.anchors)
        want = [(ch.harte, round(ch.start, 2)) for ch in gt_chords(c) if beats[0] - 1e-3 <= ch.start < beats[31]]
        assert [(x.harte, round(x.t0, 2)) for x in back][:len(want)] == want
        songs = yaml.safe_load((eval_dir / "songs.yaml").read_text("utf-8"))
        entry = next(s for s in songs["songs"] if s["key"] == KEY)
        assert entry["reviewed"] is True and entry["tapped_beats"] is True and entry["lab"] == f"eval/gt/{KEY}.lab"
        a = schema.Analysis.model_validate(c.get(f"/api/tracks/{KEY}").json())
        assert a.gt.status == "reviewed"
        # un-review keeps the files, flips the flag
        c.post(f"/api/tracks/{KEY}/gt/review", json={"reviewed": False})
        songs = yaml.safe_load((eval_dir / "songs.yaml").read_text("utf-8"))
        assert next(s for s in songs["songs"] if s["key"] == KEY)["reviewed"] is False
        # GET export returns one file (and rewrites them all)
        r = c.get(f"/api/tracks/{KEY}/gt/export?fmt=beats")
        assert r.status_code == 200 and r.text.startswith("# nathan_chordify") and "attachment" in r.headers["content-disposition"]


def test_gt_reimport_replaces_previous_and_undo_restores(env):
    put_cached_song(KEY)
    with client() as c:
        c.put(f"/api/tracks/{KEY}/gt/beats", json={"beats": tapped()})
        c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": "| B | E | F# | B |", "import_id": "one"})
        r = c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": "| G#m |", "import_id": "two"}).json()
        a = schema.Analysis.model_validate(r["analysis"])
        assert [ch.harte for ch in a.chords if ch.edited] == ["G#:min"]       # import one was reverted
        # undo the second import (revert its group, which also undoes its revert of the first)
        r = c.post(f"/api/tracks/{KEY}/edits", json={"ops": [{"id": "u2", "kind": "revert", "target": {"group": "gt-two"}}]}).json()
        a = schema.Analysis.model_validate(r["analysis"])
        assert [ch.harte for ch in a.chords if ch.edited] == ["B:maj", "E:maj", "F#:maj", "B:maj"]


def test_gt_detected_beats_lab_import_and_errors(env):
    base = put_cached_song(KEY)
    with client() as c:
        # no tapped beats: the chart goes on the detected grid from the first downbeat
        r = c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": "| C#m7 . . . |"})
        assert r.status_code == 200 and r.json()["gt"]["chart"]["source"] == "detected"
        first_down = min(b.t for b in base.beats if b.bar >= 0 and b.beat == 1)
        a = schema.Analysis.model_validate(r.json()["analysis"])
        e = next(ch for ch in a.chords if ch.edited)
        assert e.start == pytest.approx(first_down) and e.harte == "C#:min7"
        r = c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": "| C#m7 |", "start": 30.0})
        e = next(ch for ch in schema.Analysis.model_validate(r.json()["analysis"]).chords if ch.edited)
        assert abs(e.start - 30.0) < 0.6
        assert c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": "| C#m7", "import_id": "bad"}).status_code == 400
        assert c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": "| Hx |"}).json()["code"] == "bad_chart"
        assert c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": "| C |", "beats": "tapped"}).status_code == 409
        # a .lab (DESIGN §11 /gt/import name)
        r = c.post(f"/api/tracks/{KEY}/gt/import", json={"format": "lab", "text": "40.0 41.5 F#/A#\n41.5 43 B:maj7\n"})
        assert r.status_code == 200, r.text
        a = schema.Analysis.model_validate(r.json()["analysis"])
        got = [(ch.start, ch.end, ch.harte) for ch in a.chords if ch.edited]
        assert got == [(40.0, 41.5, "F#:maj/3"), (41.5, 43.0, "B:maj7")]
        # export without tapped beats warns that the .beats file isn't independent
        out = c.post(f"/api/tracks/{KEY}/gt/export", json={}).json()
        assert any("tapped" in w for w in out["warnings"]) and out["reviewed"] is False
        assert c.get(f"/api/tracks/{KEY}/gt/draft-chart").json()["kind"] == "current"
        assert c.get("/api/tracks/yt-BBBBBBBBBBB/gt").status_code == 404


def test_gt_partial_retap_and_existing_draft_chart_is_kept(env):
    put_cached_song(KEY)
    gt_dir = config.EVAL_DIR / "gt"
    gt_dir.mkdir(parents=True)
    (gt_dir / f"{KEY}.chart.txt").write_text("# hand-made draft\n[Intro @2.9]\n| B |   (L)\n", "utf-8")
    with client() as c:
        d = c.get(f"/api/tracks/{KEY}/gt/draft-chart").json()
        assert d["kind"] == "current" and "hand-made" in d["text"]
        beats = tapped(n=40)
        c.put(f"/api/tracks/{KEY}/gt/beats", json={"beats": beats})
        fix = [round(b + 0.03, 4) for b in beats[10:20]]
        st = c.put(f"/api/tracks/{KEY}/gt/beats", json={"beats": fix, "t0": beats[10] - 0.1, "t1": beats[19] + 0.1}).json()
        assert st["beats"] == beats[:10] + fix + beats[20:]
        r = c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": d["text"]}).json()
        assert r["gt"]["chart"]["unsure"] and r["gt"]["chart"]["unsure"][0]["flag"] == "L"
        c.post(f"/api/tracks/{KEY}/gt/export", json={})
        assert (gt_dir / f"{KEY}.chart.draft.txt").read_text("utf-8").startswith("# hand-made draft")
        assert c.get(f"/api/tracks/{KEY}/gt/draft-chart").json()["kind"] == "draft"
        # re-tapping after the import marks the alignment stale
        c.put(f"/api/tracks/{KEY}/gt/beats", json={"beats": tapped(start=3.1, n=40)})
        assert c.get(f"/api/tracks/{KEY}/gt").json()["chart"]["stale"] is True
    assert not (ED.read_ops(config.CACHE_DIR / KEY) == [])


def test_gt_chart_on_detected_beats_exports_on_them_even_after_a_partial_tap(env):
    base = put_cached_song(KEY)
    with client() as c:
        text = "[Verse @30]\n| B B7sus4 | G#m7/F# Em/A |\n| C#m7 . D#m7 . | E/F# |\n"
        r = c.post(f"/api/tracks/{KEY}/gt/chart", json={"text": text}).json()
        a = schema.Analysis.model_validate(r["analysis"])
        first = next(ch for ch in a.chords if ch.edited)
        under = [ch for ch in base.chords if ch.layer == "main" and ch.end > first.start and ch.start < first.end]
        assert first.alternatives and first.alternatives[0].harte in {u.harte for u in under}   # our guess, one click away
        c.put(f"/api/tracks/{KEY}/gt/beats", json={"beats": [10.0, 10.7, 11.4]})                 # a few taps only
        out = c.post(f"/api/tracks/{KEY}/gt/export", json={}).json()
        assert any("cover only" in w for w in out["warnings"])
        chart_txt = (config.EVAL_DIR / "gt" / f"{KEY}.chart.txt").read_text("utf-8")
        ac = CH.parse_annotated(chart_txt)
        det = np.asarray(sorted(b.t for b in base.beats if b.bar >= 0))
        back = CH.align_chart(ac.chart, det, anchors=ac.anchors)
        orig = CH.align_chart(CH.parse_annotated(text).chart, det, anchors=[30.0])
        assert [(x.harte, round(x.t0, 2)) for x in back[:len(orig)]] == [(x.harte, round(x.t0, 2)) for x in orig]
        bt = np.loadtxt(config.EVAL_DIR / "gt" / f"{KEY}.beats", comments="#", ndmin=2)
        assert list(bt[:, 0]) == [10.0, 10.7, 11.4]
