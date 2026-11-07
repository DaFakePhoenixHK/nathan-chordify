"""M4 chord editor (DESIGN §4 EditOp, §5.13 edit rules, §11; review W4): server/edits.py.

Unit tests of ``apply_ops`` on the 回家 fixture, then the routes (append-only log, idempotent
retries, two devices interleaving, revert, typed entry) through the stub server.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from nathan_chordify import config, schema
from nathan_chordify.export import lab_lines
from nathan_chordify.server import edits as ED
from nathan_chordify.server import stub
from nathan_chordify.theory import display as DISP

from test_server import client, env, put_cached_song  # noqa: F401  (env is a fixture)

KEY = "yt-AAAAAAAAAAA"


@pytest.fixture(scope="module")
def base() -> schema.Analysis:
    fa, _ = stub.fixture_paths()
    return schema.Analysis.model_validate_json(fa.read_text("utf-8"))


def op(seq, kind, target=None, value=None, oid=None, device="desktop"):
    return {"id": oid or f"op{seq}", "seq": seq, "ts": "2026-10-13T00:00:00Z", "device": device, "kind": kind,
            "target": target or {}, "value": value or {}}


def chord(a, cid):
    return next(c for c in a.chords if c.id == cid)


def target_of(c):
    return {"chord_id": c.id, "t0": c.start, "t1": c.end, "layer": c.layer}


def mains(a):
    return sorted((c for c in a.chords if c.layer == "main"), key=lambda c: c.start)


def assert_main_tiles(a):
    m = mains(a)
    for x, y in zip(m, m[1:]):
        assert abs(x.end - y.start) < 2e-3, (x.id, x.end, y.id, y.start)


# ---- apply ---------------------------------------------------------------------------------------

def test_build_display_matches_theory_display(base):
    assert ED.build_display(base) == DISP.build(base)


def test_no_ops_is_the_analysis(base):
    a, notes = ED.apply_ops(base, [])
    assert a.chords == base.chords and notes == []
    assert a.edits.rev == 0


def test_set_chord_respells_revoices_and_transposes(base):
    c = chord(base, "c0012")                      # E/F# at 28.74 s, key B major
    text = ED.normalize_op({"id": "x1", "kind": "set_chord", "target": target_of(c), "value": {"text": "F#/A#"}})
    assert text["value"]["harte"] == "F#:maj/3"
    a, display = ED.apply(base, [{**text, "seq": 1, "ts": ""}])
    e = chord(a, "c0012")
    assert (e.harte, e.root, e.bass, e.quality) == ("F#:maj/3", "F#", "A#", "maj")
    assert e.label.full == "F#/A#" and e.label.standard == "F#/A#" and e.label.simple == "F#"
    assert e.degree.nashville == "5/7"
    assert e.edited and e.conf == 1.0 and (e.start, e.end) == (c.start, c.end)
    assert e.alternatives[0].harte == c.harte              # the old chord is one click away
    assert e.voicing.lh and e.voicing.lh[0] % 12 == 10     # LH plays A#
    assert display.variants["1:auto"].chords["c0012"] == ["G", "G/B", "G/B"]   # transpose +1
    assert display.variants["0:flat"].chords["c0012"][2] in ("F#/A#", "Gb/Bb")
    assert display.voicings["1"]["c0012"].lh[0] % 12 == 11
    bl = next(b for b in a.bass_line if abs(b.start - c.start) < 1e-3)
    assert bl.note == "A#" and bl.midi % 12 == 10
    assert a.edits.rev == 1 and a.stats.n_main == base.stats.n_main


def test_typed_text_at_a_shift_is_untransposed():
    o = ED.normalize_op({"id": "x", "kind": "set_chord", "target": {"t0": 1, "t1": 2}, "value": {"text": "G/B", "shift": 1}})
    assert o["value"]["harte"] == "F#:maj/3"


def test_range_set_splits_and_cuts_passing(base):
    host = chord(base, "c0004")                    # E7 10.28–13.98 with passing c0005 inside
    inner = chord(base, "c0005")
    a, _ = ED.apply_ops(base, [op(7, "set_chord", {"t0": 11.0, "t1": 12.5}, {"harte": "C#:min7"})])
    ids = {c.id for c in a.chords}
    assert "c0005" not in ids and inner.start > 11.0
    left, new, right = chord(a, "c0004"), chord(a, "u7"), chord(a, "c0004~7")
    assert (left.start, left.end) == (host.start, 11.0) and left.harte == host.harte
    assert (new.start, new.end, new.harte, new.layer) == (11.0, 12.5, "C#:min7", "main")
    assert (right.start, right.end, right.harte) == (12.5, host.end, host.harte)
    assert new.start_beat is not None and new.bar >= 0 and new.dur_beats > 0
    assert_main_tiles(a)
    d = ED.build_display(a)
    assert set(d.variants["0:auto"].chords) == ids


def test_delete_change_and_insert_rest(base):
    c = chord(base, "c0012")
    prev = chord(base, "c0011")
    a, _ = ED.apply_ops(base, [op(1, "delete_change", target_of(c))])
    assert "c0012" not in {x.id for x in a.chords}
    assert chord(a, "c0011").end == c.end
    assert_main_tiles(a)
    p = chord(base, "c0005")
    a, _ = ED.apply_ops(base, [op(1, "delete_change", target_of(p))])
    assert "c0005" not in {x.id for x in a.chords} and chord(a, "c0004").end == chord(base, "c0004").end
    a, _ = ED.apply_ops(base, [op(1, "insert_rest", target_of(prev), {"harte": "N"})])
    r = chord(a, "c0011")
    assert r.harte == "N" and r.root is None and r.label.full == "N" and r.voicing.lh == []


def test_revert_redo_and_later_op_wins(base):
    c = chord(base, "c0012")
    ops = [op(1, "set_chord", target_of(c), {"harte": "F#:maj/3"}),
           op(2, "set_chord", target_of(c), {"harte": "D#:min7"})]
    a, _ = ED.apply_ops(base, ops)
    assert chord(a, "c0012").harte == "D#:min7"                     # same target: the later op wins
    ops.append(op(3, "revert", {"op_id": "op2"}))
    a, _ = ED.apply_ops(base, ops)
    assert chord(a, "c0012").harte == "F#:maj/3"
    ops.append(op(4, "revert", {"op_id": "op3"}))                  # redo = revert the revert
    a, _ = ED.apply_ops(base, ops)
    assert chord(a, "c0012").harte == "D#:min7"
    ops += [op(5, "revert", {"op_id": "op4"}), op(6, "revert", {"op_id": "op1"})]
    a, _ = ED.apply_ops(base, ops)
    assert chord(a, "c0012") == chord(base, "c0012").model_copy(update={"voicing": chord(a, "c0012").voicing})
    active, reverted = ED.active_ops(ops)
    assert [o["id"] for o in active] == [] and reverted >= {"op1", "op2", "op4"}


def test_group_revert(base):
    ops = [op(1, "set_chord", {"t0": 30.0, "t1": 31.0}, {"harte": "A:7", "group": "g1"}),
           op(2, "set_chord", {"t0": 31.0, "t1": 32.0}, {"harte": "B:7", "group": "g1"}),
           op(3, "set_chord", {"t0": 40.0, "t1": 41.0}, {"harte": "C:maj"})]
    a, _ = ED.apply_ops(base, ops + [op(4, "revert", {"group": "g1"})])
    hs = {c.harte for c in a.chords if c.edited}
    assert "A:7" not in hs and "B:7" not in hs and any(c.id == "u3" for c in a.chords)
    a, _ = ED.apply_ops(base, ops + [op(4, "revert", {"group": "g1"}), op(5, "revert", {"op_id": "op4"})])
    assert {"u1", "u2", "u3"} <= {c.id for c in a.chords}


def test_target_falls_back_to_time_when_the_id_is_gone(base):
    c = chord(base, "c0012")
    t = {**target_of(c), "chord_id": "c9999"}
    a, notes = ED.apply_ops(base, [op(1, "set_chord", t, {"harte": "A:maj"})])
    assert chord(a, "c0012").harte == "A:maj" and notes == []
    t = {"chord_id": "c9999", "t0": c.start + 0.1, "t1": c.start + 0.3, "layer": "main"}
    a, _ = ED.apply_ops(base, [op(2, "set_chord", t, {"harte": "A:maj"})])   # small span: a range edit
    assert chord(a, "u2").harte == "A:maj" and chord(a, "c0012").harte == c.harte
    a, notes = ED.apply_ops(base, [op(3, "delete_change", {"chord_id": "c9999"})])
    assert notes == [{"op": "op3", "note": "chord not found"}]


def test_half_and_double_tempo(base):
    a, _ = ED.apply_ops(base, [op(1, "half_double_tempo", value={"factor": 2})])
    assert a.tempo.bpm == pytest.approx(base.tempo.bpm * 2, abs=0.02)
    metered = [b for b in base.beats if b.bar >= 0]
    assert len([b for b in a.beats if b.bar >= 0]) == 2 * len(metered)
    assert len(a.bars) == 2 * len(base.bars)
    assert sum(x.beats for x in a.bars) == 2 * sum(x.beats for x in base.bars)
    assert all(x.beats == 4 for x, o in zip(a.bars[::2], base.bars) if o.beats == 4)   # 4/4 stays 4/4
    assert a.tempo.subdivision[0].cells_per_beat == 2
    assert [(c.id, c.start, c.end, c.harte) for c in a.chords] == [(c.id, c.start, c.end, c.harte) for c in base.chords]
    c = chord(a, "c0012")
    assert c.dur_beats == pytest.approx(2 * chord(base, "c0012").dur_beats, abs=0.05)
    back, _ = ED.apply_ops(base, [op(1, "half_double_tempo", value={"factor": 2}),
                                  op(2, "half_double_tempo", value={"factor": 0.5})])
    assert back.beats == base.beats and back.bars == base.bars
    h, _ = ED.apply_ops(base, [op(1, "half_double_tempo", value={"factor": 0.5})])
    assert h.tempo.bpm == pytest.approx(base.tempo.bpm / 2, abs=0.02)
    assert len(h.bars) == (len([b for b in base.bars if not b.free]) + 1) // 2 + len([b for b in base.bars if b.free])


def test_sections_and_key(base):
    bar5 = next(b for b in base.bars if b.i == 5)
    a, _ = ED.apply_ops(base, [op(1, "set_section", {"t0": bar5.start}, {"label": "Verse 1"}),
                               op(2, "set_section", {"bar": 9}, {"label": "Pre-chorus"})])
    assert [(s.label, s.source) for s in a.sections] == [("Verse 1", "user"), ("Pre-chorus", "user")]
    assert next(b for b in a.bars if b.i == 5).section == "Verse 1"
    assert next(b for b in a.bars if b.i == 8).section == "Verse 1"
    assert next(b for b in a.bars if b.i == 9).section == "Pre-chorus"
    a, _ = ED.apply_ops(base, [op(1, "set_key", value={"key": "C:major"})])
    assert [(s.tonic, s.mode) for s in a.key.segments] == [("C", "major")] and a.key.changes == []
    c = chord(a, "c0012")                          # E/F# keeps its pitch, its degrees follow the key
    assert c.label.full == "E/F#" and c.degree.root == 4
    a, _ = ED.apply_ops(base, [op(1, "set_key", value={"key": "B:major"}),
                               op(2, "set_key", {"t0": 200.0}, {"key": "C#:major"})])
    assert a.key.changes[-1].kind == "step_up" and a.key.changes[-1].to == "C#:major"


def test_suggestions_accept_and_reject(base):
    sg = chord(base, "c0005").model_copy(update={"id": "sg-1", "suggestion": True, "start": 50.0, "end": 50.4,
                                                 "harte": "C:dim7", "min_detail": 0.05})
    sg2 = sg.model_copy(update={"id": "sg-2", "start": 60.0, "end": 60.4, "harte": "A:aug"})
    a, _ = ED.apply_ops(base, [], [sg, sg2])
    assert {c.id for c in a.chords if c.suggestion} == {"sg-1", "sg-2"}
    ops = [op(1, "accept_suggestion", {"chord_id": "sg-1", "t0": 50.0, "t1": 50.4}, {"harte": "C:dim7"}),
           op(2, "reject_suggestion", {"chord_id": "sg-2", "t0": 60.0, "t1": 60.4}, {"harte": "A:aug"})]
    a, _ = ED.apply_ops(base, ops, [sg, sg2])
    acc = chord(a, "sg-1")
    assert not acc.suggestion and acc.edited and acc.layer == "passing"
    assert "sg-2" not in {c.id for c in a.chords}
    a, _ = ED.apply_ops(base, ops)                 # suggestions gone (e.g. re-analysed): re-inserted from the op
    new = chord(a, "a1")
    assert (new.start, new.end, new.layer, new.min_detail) == (50.0, 50.4, "passing", 0.05)
    assert new.harte == "C:dim7" and new.label.full in ("C°7", "B#°7")


@pytest.mark.parametrize("bad", [
    {"kind": "set_chord", "target": {"t0": 1, "t1": 2}, "value": {"text": "Hxyz"}},
    {"kind": "set_chord", "target": {"t0": 3, "t1": 2}, "value": {"harte": "C"}},
    {"kind": "set_chord", "target": {}, "value": {"harte": "C"}},
    {"kind": "half_double_tempo", "value": {"factor": 3}},
    {"kind": "revert", "target": {}},
    {"kind": "set_key", "value": {"key": "H major"}},
    {"kind": "nonsense"},
])
def test_bad_ops_are_refused(bad):
    with pytest.raises(Exception) as ei:
        ED.normalize_op({"id": "x", **bad})
    assert getattr(ei.value, "status", None) == 400


def test_later_kinds_are_501():
    with pytest.raises(Exception) as ei:
        ED.normalize_op({"id": "x", "kind": "move_beat", "target": {}, "value": {}})
    assert ei.value.status == 501


# ---- routes --------------------------------------------------------------------------------------

def post(c, ops, device="desktop"):
    r = c.post(f"/api/tracks/{KEY}/edits", json={"ops": ops, "device": device})
    assert r.status_code == 200, r.text
    return r.json()


def test_edit_routes_append_only_and_idempotent(env):
    a = put_cached_song(KEY)
    c12 = chord(a, "c0012")
    with client() as c:
        assert c.get(f"/api/tracks/{KEY}/edits").json() == {"rev": 0, "n_ops": 0, "ops": []}
        o = {"id": "11111111-aaaa", "kind": "set_chord", "target": target_of(c12), "value": {"text": "F#/A#"}}
        r = post(c, [o])
        assert r["rev"] == 1 and r["appended"] == [{"id": "11111111-aaaa", "seq": 1}]
        assert chord(schema.Analysis.model_validate(r["analysis"]), "c0012").harte == "F#:maj/3"
        assert r["display"]["variants"]["1:auto"]["chords"]["c0012"][2] == "G/B"
        r2 = post(c, [o])                                           # a retry: nothing new
        assert r2["rev"] == 1 and r2["appended"] == [] and r2["duplicates"] == [{"id": "11111111-aaaa", "seq": 1}]
        lines = (config.CACHE_DIR / KEY / "edits.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["seq"] == 1
        g = c.get(f"/api/tracks/{KEY}").json()
        assert chord(schema.Analysis.model_validate(g), "c0012").harte == "F#:maj/3" and g["edits"]["rev"] == 1
        orig = c.get(f"/api/tracks/{KEY}?original=1").json()
        assert chord(schema.Analysis.model_validate(orig), "c0012").harte == c12.harte
        d = c.get(f"/api/tracks/{KEY}/display").json()
        assert d["variants"]["1:auto"]["chords"]["c0012"] == ["G", "G/B", "G/B"]
        assert c.get(f"/api/tracks/{KEY}/display?original=1").json()["variants"]["1:auto"]["chords"]["c0012"] != d["variants"]["1:auto"]["chords"]["c0012"]
        lab = c.get(f"/api/tracks/{KEY}/export/lab?detail=0.8").text
        assert any(line.endswith("F#:maj/3") and line.startswith(f"{c12.start:.3f}") for line in lab.splitlines())
        j = c.get(f"/api/tracks/{KEY}/export/json").json()
        assert chord(schema.Analysis.model_validate(j), "c0012").edited
        # undo
        r = post(c, [{"id": "undo-1", "kind": "revert", "target": {"op_id": "11111111-aaaa"}}])
        assert chord(schema.Analysis.model_validate(r["analysis"]), "c0012").harte == c12.harte
        h = c.get(f"/api/tracks/{KEY}/edits").json()
        assert [(o["seq"], o["kind"], o["active"]) for o in h["ops"]] == [(1, "set_chord", False), (2, "revert", True)]
        assert [o["seq"] for o in c.get(f"/api/tracks/{KEY}/edits?since=1").json()["ops"]] == [2]
        # refused ops leave the log alone
        bad = c.post(f"/api/tracks/{KEY}/edits", json={"ops": [{"id": "z1", "kind": "set_chord", "target": {"t0": 1, "t1": 2},
                                                                "value": {"harte": "C"}}, {"id": "z2", "kind": "nope"}]})
        assert bad.status_code == 400
        assert c.post(f"/api/tracks/{KEY}/edits", json={"ops": [{"id": "z3", "kind": "move_beat"}]}).status_code == 501
        assert c.post(f"/api/tracks/{KEY}/edits", json={"ops": [{"id": "dup", "kind": "revert", "target": {"op_id": "x"}},
                                                                {"id": "dup", "kind": "revert", "target": {"op_id": "x"}}]}).status_code == 400
        assert c.get(f"/api/tracks/{KEY}/edits").json()["rev"] == 2
        assert c.get("/api/tracks/yt-BBBBBBBBBBB/edits").status_code == 404


def test_two_devices_interleaved_merge_without_loss(env):
    a = put_cached_song(KEY)
    m = mains(a)
    with client() as c:
        desk = [{"id": f"d{i}", "kind": "set_chord", "target": target_of(m[10 + i]), "value": {"harte": "A:7"}}
                for i in range(6)]
        phone = [{"id": f"p{i}", "kind": "set_chord", "target": target_of(m[30 + i]), "value": {"harte": "C:min7"}}
                 for i in range(6)]
        # interleaved batches, with retries of already-sent ops (an offline phone replaying its queue)
        post(c, desk[:2], "desktop")
        post(c, phone[:3], "phone-1")
        post(c, desk[1:4], "desktop")
        post(c, phone[:5], "phone-1")
        post(c, [{"id": "d-undo", "kind": "revert", "target": {"op_id": "p0"}}], "desktop")
        r = post(c, desk + phone, "desktop")
        assert r["rev"] == 13 and r["n_ops"] == 13
        final = schema.Analysis.model_validate(r["analysis"])
        for i in range(6):
            assert chord(final, m[10 + i].id).harte == "A:7"
        assert chord(final, m[30].id).harte == m[30].harte                       # reverted
        for i in range(1, 6):
            assert chord(final, m[30 + i].id).harte == "C:min7"
        ops = ED.read_ops(config.CACHE_DIR / KEY)
        assert [o["seq"] for o in ops] == list(range(1, 14)) and len({o["id"] for o in ops}) == 13
        assert {o["device"] for o in ops} == {"desktop", "phone-1"}
        # deterministic: replaying the log on the stored analysis gives the same song
        again, _ = ED.apply_ops(ED.load_base(config.CACHE_DIR / KEY), ops)
        assert again.model_dump(mode="json", by_alias=True) == {**r["analysis"], "gt": again.gt.model_dump()}

        # two devices posting at the same moment: every op lands once, seqs stay unique
        def burst(dev, n):
            for i in range(n):
                post(c, [{"id": f"{dev}-{i}", "kind": "set_section", "target": {"bar": i % 20}, "value": {"label": dev}}], dev)
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda dv: burst(*dv), [("desktop", 15), ("phone-2", 15)]))
        ops = ED.read_ops(config.CACHE_DIR / KEY)
        assert len(ops) == 43 and [o["seq"] for o in ops] == list(range(1, 44))


def test_theory_parse_and_vocab(env):
    with client() as c:
        r = c.post("/api/theory/parse", json={"text": "F#/A#", "key": "B:major"}).json()
        assert r["ok"] and r["harte"] == "F#:maj/3" and r["shown"] == ["F#", "F#/A#", "F#/A#"]
        assert r["degree"]["nashville"] == "5/7"
        r = c.post("/api/theory/parse", json={"text": "G/B", "key": "B:major", "shift": 1}).json()
        assert r["harte"] == "F#:maj/3" and r["shown"] == ["G", "G/B", "G/B"] and r["label"]["full"] == "F#/A#"
        r = c.post("/api/theory/parse", json={"text": "Bxyz", "key": "B"}).json()
        assert r["ok"] is False and r["error"] and r["hint"]
        r = c.post("/api/theory/parse", json={"text": "n.c.", "key": "B"}).json()
        assert r["ok"] and r["harte"] == "N"
        r = c.post("/api/theory/parse", json={"hartes": ["B:aug(b7)", "E:min6", "junk"], "key": "B:major", "shift": 1}).json()
        assert r["labels"] == {"B:aug(b7)": ["C", "C+", "C+7"], "E:min6": ["Fm", "Fm6", "Fm6"]}
        r = c.post("/api/theory/parse", json={"texts": ["C#m7", "E/F#"], "key": "B:major"}).json()
        assert [x["harte"] for x in r["results"]] == ["C#:min7", "E:maj/2"]
        assert c.post("/api/theory/parse", json={"text": "C", "key": "Q"}).status_code == 400
        v = c.get("/api/theory/vocab?key=B:major&shift=0").json()
        assert [x["name"] for x in v["roots"]][:3] == ["C", "C#", "D"] and v["roots"][10]["name"] == "A#"
        assert len(v["qualities"]) == 32
        for q in v["qualities"]:                     # every picker suffix reads back as its quality
            p = c.post("/api/theory/parse", json={"text": "D" + q["suffix"], "key": "B:major"}).json()
            assert p["ok"] and p["quality"] == q["q"], q
        assert c.get("/api/theory/vocab?key=Eb:major").json()["roots"][1]["name"] == "Db"


def test_rescan_suggestions_persist_until_accepted(env, monkeypatch):
    from nathan_chordify import pipeline
    a = put_cached_song(KEY)
    sug = chord(a, "c0005").model_copy(update={"id": "c0100", "suggestion": True, "start": 70.0, "end": 70.45,
                                               "harte": "C:dim7", "min_detail": 0.05})
    result = a.model_copy(update={"chords": a.chords + [sug]})
    monkeypatch.setattr(pipeline, "redecode", lambda k, params, region, **kw: result, raising=False)
    with client() as c:
        post(c, [{"id": "older", "kind": "set_chord", "target": {"t0": 69.0, "t1": 72.0}, "value": {"harte": "E:maj"}}])
        r = c.post(f"/api/tracks/{KEY}/redecode", json={"region": [65.0, 75.0]})
        assert r.status_code == 200 and r.json()["n_suggestions"] == 1
        got = schema.Analysis.model_validate(r.json()["analysis"])
        s = next(x for x in got.chords if x.suggestion)
        assert s.id.startswith("sg-") and s.layer == "passing"
        assert s.id in r.json()["display"]["variants"]["0:auto"]["chords"]
        g = schema.Analysis.model_validate(c.get(f"/api/tracks/{KEY}").json())
        assert any(x.suggestion for x in g.chords)                 # still pending on reload
        r = post(c, [{"id": "acc1", "kind": "accept_suggestion", "target": {"chord_id": s.id, "t0": s.start, "t1": s.end},
                      "value": {"harte": s.harte}}])
        acc = chord(schema.Analysis.model_validate(r["analysis"]), s.id)
        assert not acc.suggestion and acc.edited
        lab = c.get(f"/api/tracks/{KEY}/export/lab?detail=0.8").text
        assert any("C:dim7" in line for line in lab.splitlines())
        # an edit made after the re-scan does cut a pending suggestion
        assert c.post(f"/api/tracks/{KEY}/redecode", json={"region": [65.0, 75.0]}).status_code == 200
        r = post(c, [{"id": "newer", "kind": "set_chord", "target": {"t0": 69.5, "t1": 71.5}, "value": {"harte": "A:maj"}}])
        assert not any(x["start"] == 70.0 and x["harte"] == "C:dim7" for x in r["analysis"]["chords"])
