"""server/mixes + GET /api/tracks/{key}/audio?mix=…: practice mixes made from the cached stems.

A fake ``encode`` stands in for ffmpeg (counts calls, writes bytes); one test runs the real ffmpeg on
one-second synthetic stems when tools/ffmpeg is present.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nathan_chordify import config
from nathan_chordify.server import mixes as MIX

from test_server import client, env, put_cached_song  # noqa: F401  (env is a fixture)

KEY = "yt-MIXTEST0001"


def write_stems(d: Path, names=("vocals", "drums", "bass", "other", "harmonic"), seconds: float = 1.0) -> Path:
    sd = d / "stems"
    sd.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(44100 * seconds)) / 44100
    for i, n in enumerate(names):
        y = 0.2 * np.sin(2 * np.pi * (110 * (i + 1)) * t)
        sf.write(sd / f"{n}.flac", np.stack([y, y], axis=1), 44100, subtype="PCM_16")
    return sd


def test_mix_made_once_cached_and_served_with_ranges(env, monkeypatch):
    a = put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    write_stems(d)
    calls: list[list[Path]] = []

    def fake_encode(inputs, out):
        calls.append(list(inputs))
        out.write_bytes(b"m4a" * 5000)

    monkeypatch.setattr(MIX, "encode", fake_encode)
    with client() as c:
        st = c.get(f"/api/tracks/{KEY}/mixes").json()["mixes"]
        assert st == {"original": "ready", "novocals": "makeable", "harmony": "makeable"}
        r = c.head(f"/api/tracks/{KEY}/audio?mix=novocals")           # HEAD makes it too
        assert r.status_code == 200 and int(r.headers["content-length"]) == 15000
        assert r.headers["content-type"] == "audio/mp4"
        assert (d / "play_novocals.m4a").exists()
        assert [p.name for p in calls[0]] == ["bass.flac", "drums.flac", "other.flac"]
        r = c.get(f"/api/tracks/{KEY}/audio?mix=novocals", headers={"Range": "bytes=0-9"})
        assert r.status_code == 206 and r.content == b"m4am4am4am"
        assert len(calls) == 1                                        # cached: no second encode
        r = c.get(f"/api/tracks/{KEY}/audio?mix=harmony")
        assert r.status_code == 200
        assert [p.name for p in calls[1]] == ["harmonic.flac"]       # bass + other already summed
        assert c.get(f"/api/tracks/{KEY}/mixes").json()["mixes"]["harmony"] == "ready"
        r = c.get(f"/api/tracks/{KEY}/audio?mix=original")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/mp4"
        assert c.get(f"/api/tracks/{KEY}/audio?mix=karaoke").status_code == 400
    assert a.source.key == KEY


def test_harmony_falls_back_to_bass_plus_other(env, monkeypatch):
    put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    write_stems(d, names=("vocals", "drums", "bass", "other"))
    calls = []
    monkeypatch.setattr(MIX, "encode", lambda inputs, out: (calls.append([p.name for p in inputs]), out.write_bytes(b"x" * 10)))
    with client() as c:
        assert c.get(f"/api/tracks/{KEY}/audio?mix=harmony").status_code == 200
    assert calls == [["bass.flac", "other.flac"]]


def test_no_stems_is_409_with_a_plain_hint(env):
    put_cached_song(KEY)
    with client() as c:
        r = c.get(f"/api/tracks/{KEY}/audio?mix=novocals")
        assert r.status_code == 409
        j = r.json()
        assert j["code"] == "no_stems" and "without separating the voice" in j["msg"] and "Re-analyse" in j["hint"]
        assert c.head(f"/api/tracks/{KEY}/audio?mix=harmony").status_code == 409
        assert c.get(f"/api/tracks/{KEY}/mixes").json()["mixes"]["novocals"] == "no_stems"
        assert c.get("/api/tracks/yt-NOSUCHSONG00/audio?mix=novocals").status_code == 404
    # fast mode leaves ".absent" markers in the Stems record, never files: still no stems
    d = config.CACHE_DIR / KEY
    (d / "stems").mkdir()
    (d / "stems" / "bass_lp300.flac").write_bytes(b"\0")
    assert MIX.stem_inputs(d, "novocals") is None
    with pytest.raises(MIX.NoStems):
        MIX.ensure(d, "novocals")


def test_concurrent_requests_make_the_file_once(env, monkeypatch):
    put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    write_stems(d)
    n = {"calls": 0}

    def slow_encode(inputs, out):
        n["calls"] += 1
        time.sleep(0.3)
        out.write_bytes(b"y" * 100)

    monkeypatch.setattr(MIX, "encode", slow_encode)
    results: list[Path] = []
    threads = [threading.Thread(target=lambda: results.append(MIX.ensure(d, "novocals"))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert n["calls"] == 1 and len(results) == 4 and all(p == d / "play_novocals.m4a" for p in results)


def test_encode_failure_is_reported(env, monkeypatch):
    put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    write_stems(d)

    def bad_encode(inputs, out):
        raise MIX.MixError("ffmpeg could not make the mix: boom")

    monkeypatch.setattr(MIX, "encode", bad_encode)
    with client() as c:
        r = c.get(f"/api/tracks/{KEY}/audio?mix=novocals")
        assert r.status_code == 500 and r.json()["code"] == "mix_failed"
    assert not (d / "play_novocals.m4a").exists()


@pytest.mark.skipif(not config.FFMPEG.exists(), reason="tools/ffmpeg not installed")
def test_real_ffmpeg_mixes_synthetic_stems(tmp_path):
    d = tmp_path / "song"
    d.mkdir()
    write_stems(d, seconds=0.5)
    out = MIX.ensure(d, "novocals")
    assert out.exists() and out.stat().st_size > 1000
    data = out.read_bytes()
    assert b"ftyp" in data[:64]                     # an MP4/M4A container, faststart (moov near the start)
    assert b"moov" in data[:4096]
    out2 = MIX.ensure(d, "harmony")
    assert out2.exists() and out2 != out


# ---- pitch-shifted copies (?transpose=, 2026-11-04: "original audio or transposed audio") -----------

def fake_shift(calls: list):
    def _shift(src: Path, out: Path, n: int):
        calls.append((src.name, out.name, n))
        out.write_bytes(b"s" * 2000)
    return _shift


def test_shift_path_and_name_round_trip(tmp_path):
    assert MIX.shift_path(tmp_path, "original", -3).name == "play_t-3.m4a"
    assert MIX.shift_path(tmp_path, "novocals", 2).name == "play_novocals_t+2.m4a"
    assert MIX.shift_path(tmp_path, "harmony", 0).name == "play_harmony.m4a"
    assert MIX.parse_shift_name("play_t-3.m4a") == ("original", -3)
    assert MIX.parse_shift_name("play_novocals_t+2.m4a") == ("novocals", 2)
    assert MIX.parse_shift_name("play_harmony_t+11.m4a") == ("harmony", 11)
    assert MIX.parse_shift_name("play.m4a") is None
    assert MIX.parse_shift_name("play_novocals.m4a") is None
    assert MIX.parse_shift_name("play_t-3.tmp.m4a") is None
    assert [f for f in MIX.shift_filters(-3) if f.startswith("rubberband=pitch=0.84089")]
    assert MIX.shift_filters(2)[-1].startswith("aresample=48000,asetrate=53878.")


def test_transposed_audio_made_once_from_the_mix_and_served(env, monkeypatch):
    put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    write_stems(d)
    shifts: list = []
    encodes: list = []
    monkeypatch.setattr(MIX, "shift", fake_shift(shifts))
    monkeypatch.setattr(MIX, "encode", lambda inputs, out: (encodes.append([p.name for p in inputs]), out.write_bytes(b"m" * 3000)))
    with client() as c:
        # 0 = the plain file, nothing made
        r = c.get(f"/api/tracks/{KEY}/audio?transpose=0")
        assert r.status_code == 200 and len(r.content) == 10_000 and shifts == []
        # HEAD makes the shifted original and waits for it
        r = c.head(f"/api/tracks/{KEY}/audio?transpose=-3")
        assert r.status_code == 200 and r.headers["content-type"] == "audio/mp4"
        assert int(r.headers["content-length"]) == 2000
        assert shifts == [("play.m4a", "play_t-3.m4a", -3)] and (d / "play_t-3.m4a").exists()
        r = c.get(f"/api/tracks/{KEY}/audio?transpose=-3", headers={"Range": "bytes=0-3"})
        assert r.status_code == 206 and r.content == b"ssss"
        assert len(shifts) == 1                                       # cached
        # a mix + transpose: the mix is made first, then shifted from it
        r = c.get(f"/api/tracks/{KEY}/audio?mix=novocals&transpose=%2B2")
        assert r.status_code == 200 and len(r.content) == 2000
        assert encodes == [["bass.flac", "drums.flac", "other.flac"]]
        assert shifts[-1] == ("play_novocals.m4a", "play_novocals_t+2.m4a", 2)
        assert c.get(f"/api/tracks/{KEY}/audio?mix=novocals&transpose=2").status_code == 200
        assert len(shifts) == 2
        st = c.get(f"/api/tracks/{KEY}/mixes").json()
        assert st["mixes"]["novocals"] == "ready"
        assert st["transposes_ready"] == {"novocals": [2], "original": [-3]}
        # out of range / not a number: 400 with a plain message
        for bad in ("7", "-7", "abc", "1.5"):
            r = c.get(f"/api/tracks/{KEY}/audio?transpose={bad}")
            assert r.status_code == 400 and r.json()["code"] == "bad_transpose", bad
            assert "semitones" in r.json()["msg"]
        assert c.head(f"/api/tracks/{KEY}/audio?transpose=12").status_code == 400
        assert c.get(f"/api/tracks/{KEY}/audio?transpose=6").status_code == 200
        assert c.get(f"/api/tracks/{KEY}/audio?transpose=-6").status_code == 200
        assert c.get("/api/tracks/yt-NOSUCHSONG00/audio?transpose=-3").status_code == 404
    # no stems: the mix fails before any shift
    put_cached_song("yt-MIXTEST0002")
    with client() as c:
        r = c.get("/api/tracks/yt-MIXTEST0002/audio?mix=harmony&transpose=-1")
        assert r.status_code == 409 and r.json()["code"] == "no_stems"
    assert all(s[1] != "play_harmony_t-1.m4a" for s in shifts)


def test_shift_failure_is_500_shift_failed(env, monkeypatch):
    put_cached_song(KEY)
    d = config.CACHE_DIR / KEY

    def bad_shift(src, out, n):
        raise MIX.ShiftError("ffmpeg could not shift the audio: boom")

    monkeypatch.setattr(MIX, "shift", bad_shift)
    with client() as c:
        r = c.get(f"/api/tracks/{KEY}/audio?transpose=2")
        assert r.status_code == 500
        j = r.json()
        assert j["code"] == "shift_failed" and "new key" in j["msg"] and "Original" in j["hint"]
    assert not (d / "play_t+2.m4a").exists()


def test_concurrent_transposed_requests_make_the_file_once(env, monkeypatch):
    put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    n = {"calls": 0}

    def slow_shift(src, out, k):
        n["calls"] += 1
        time.sleep(0.3)
        out.write_bytes(b"z" * 50)

    monkeypatch.setattr(MIX, "shift", slow_shift)
    results: list[Path] = []
    threads = [threading.Thread(target=lambda: results.append(MIX.ensure(d, "original", 4))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert n["calls"] == 1 and len(results) == 4 and all(p == d / "play_t+4.m4a" for p in results)


def test_a_song_keeps_at_most_12_shifted_files_dropping_the_least_recently_used(env, monkeypatch):
    import os

    put_cached_song(KEY)
    d = config.CACHE_DIR / KEY
    write_stems(d)
    monkeypatch.setattr(MIX, "shift", fake_shift([]))
    monkeypatch.setattr(MIX, "encode", lambda inputs, out: out.write_bytes(b"m" * 10))
    # 12 shifted files, with distinct ages (oldest = original -6)
    plan = [("original", k) for k in range(-6, 0)] + [("novocals", k) for k in range(1, 7)]
    base = time.time() - 1000
    for i, (m, k) in enumerate(plan):
        p = MIX.ensure(d, m, k)
        os.utime(p, (base + i, base + i))
    assert len(MIX.shifted_files(d)) == 12
    # using the oldest one again makes it the newest
    MIX.ensure(d, "original", -6)
    assert (d / "play_t-6.m4a").stat().st_mtime > base + 100
    # the 13th pushes out the least recently used (original -5 now)
    MIX.ensure(d, "harmony", 3)
    names = sorted(p.name for p in MIX.shifted_files(d))
    assert len(names) == 12
    assert "play_t-5.m4a" not in names and "play_t-6.m4a" in names and "play_harmony_t+3.m4a" in names
    assert (d / "play_novocals.m4a").exists() and (d / "play.m4a").exists()   # plain files never pruned
    assert MIX.transposes_ready(d)["harmony"] == [3]


def _decode_wav(m4a: Path, wav: Path) -> tuple[np.ndarray, int]:
    import subprocess

    subprocess.run([str(config.FFMPEG), "-nostdin", "-y", "-loglevel", "error", "-i", str(m4a), str(wav)],
                   check=True, capture_output=True, creationflags=config.NO_WINDOW)
    y, sr = sf.read(wav)
    return (y.mean(axis=1) if y.ndim > 1 else y), sr


def _peak_hz(y: np.ndarray, sr: int) -> float:
    seg = y[sr // 2 : sr // 2 + sr]
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return float(np.fft.rfftfreq(len(seg), 1 / sr)[int(sp.argmax())])


@pytest.mark.skipif(not config.FFMPEG.exists(), reason="tools/ffmpeg not installed")
def test_real_ffmpeg_shift_doubles_the_pitch_at_plus_12_and_keeps_the_length(tmp_path):
    """A 2 s 440 Hz tone moved up 12 semitones peaks at 880 Hz; the duration stays within 2 %."""
    import subprocess

    d = tmp_path / "song"
    d.mkdir()
    sr = 44100
    t = np.arange(2 * sr) / sr
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(d / "tone.wav", np.stack([y, y], axis=1), sr, subtype="PCM_16")
    subprocess.run([str(config.FFMPEG), "-nostdin", "-y", "-loglevel", "error", "-i", str(d / "tone.wav"),
                    "-c:a", "aac", "-b:a", "192k", str(d / "play.m4a")], check=True, capture_output=True,
                   creationflags=config.NO_WINDOW)
    out = d / "play_t+12.m4a"
    MIX.shift(d / "play.m4a", out, 12)                 # +12 is outside the route's range but a clean check
    assert out.exists() and b"ftyp" in out.read_bytes()[:64]
    y0, sr0 = _decode_wav(d / "play.m4a", d / "a.wav")
    y1, sr1 = _decode_wav(out, d / "b.wav")
    assert abs(_peak_hz(y0, sr0) - 440) < 3
    assert abs(_peak_hz(y1, sr1) - 880) < 6
    assert abs(len(y1) / sr1 - len(y0) / sr0) <= 0.02 * (len(y0) / sr0)
    # the route's own range: -3 lands on 440 * 2^(-3/12) = 370 Hz
    out2 = MIX.ensure(d, "original", -3)
    assert out2.name == "play_t-3.m4a"
    y2, sr2 = _decode_wav(out2, d / "c.wav")
    assert abs(_peak_hz(y2, sr2) - 369.99) < 6
