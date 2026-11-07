"""WP9 server tests: FastAPI TestClient with the GPU worker replaced by the fixture stub (no GPU).

Real-GPU integration tests are skipped unless NCHORD_GPU_TESTS=1.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nathan_chordify import ENGINE_VERSION, config, schema
from nathan_chordify.export import lab_lines
from nathan_chordify.server import storage, stub
from nathan_chordify.server.app import STUB_ANALYZE, ServerSettings, create_app

REAL_CACHE = config.CACHE_DIR
HUIJIA = "yt-Y7boUFbM76k"
HUIJIA_URL = "https://www.youtube.com/watch?v=Y7boUFbM76k&list=RDY7boUFbM76k&start_radio=1&pp=ygUM5Zue5a62IGNob3Jk"
GPU = os.environ.get("NCHORD_GPU_TESTS") == "1"
gpu = pytest.mark.skipif(not GPU, reason="real-GPU test: set NCHORD_GPU_TESTS=1")


# ---- helpers ------------------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp cache/data/logs; the stub pipeline sleeps 10 ms per stage."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path / "eval")      # GT exports never touch the real eval/
    monkeypatch.setenv("NCHORD_STUB_DELAY", "0.01")
    return tmp_path


def settings(**kw) -> ServerSettings:
    kw.setdefault("worker", "thread")
    kw.setdefault("analyze_fn", stub.analyze)
    kw.setdefault("prewarm", False)
    kw.setdefault("allowed_hosts", ("testserver", "127.0.0.1"))
    return ServerSettings(**kw)


def wait_ready(c: TestClient, timeout: float = 30.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        h = c.get("/health").json()
        if h["worker"] == "ready":
            return h
        time.sleep(0.05)
    raise AssertionError(f"worker not ready: {h}")


@contextmanager
def client(**kw):
    with TestClient(create_app(settings(**kw))) as c:
        wait_ready(c, 60.0)
        yield c


def wav_bytes(seconds: float = 1.0, sr: int = 22050, freq: float = 440.0) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    y = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())
    return buf.getvalue()


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name:
            out.append((name, data))
    return out


def events(c: TestClient, job_id: str, timeout: float = 60.0) -> list[tuple[str, dict]]:
    """The whole SSE stream of a job (it closes after done/error); fails instead of hanging."""
    with ThreadPoolExecutor(1) as ex:
        r = ex.submit(c.get, f"/api/jobs/{job_id}/events").result(timeout)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    return parse_sse(r.text)


def upload(c: TestClient, data: bytes | None = None, name: str = "我的歌.wav") -> dict:
    r = c.post("/api/uploads", files={"file": (name, data or wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()


def put_cached_song(key: str, audio: bytes | None = None, engine_version: str = ENGINE_VERSION) -> schema.Analysis:
    """A finished song in the temp cache: the fixture re-keyed, plus play.m4a."""
    fa, fd = stub.fixture_paths()
    a = schema.Analysis.model_validate_json(fa.read_text("utf-8"))
    a.engine_version = engine_version
    a.mode = "full"
    a.source = a.source.model_copy(update={"key": key})
    d = config.CACHE_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "analysis.json").write_text(a.model_dump_json(by_alias=True), "utf-8")
    shutil.copyfile(fd, d / "display.json")
    (d / "play.m4a").write_bytes(audio if audio is not None else os.urandom(10_000))
    (d / "meta.json").write_text(json.dumps({"source_type": "youtube", "source_url": a.source.url}), "utf-8")
    return a


# ---- jobs, SSE, tracks --------------------------------------------------------------------------

def test_health_and_system_info(env):
    with client() as c:
        h = c.get("/health").json()
        assert h == {"status": "ok", "worker": "ready", "queue": 0, "engine_version": ENGINE_VERSION}
        info = c.get("/api/system/info").json()
        for k in ("engine_version", "schema_version", "python", "packages", "gpu", "ffmpeg", "deno", "yt_dlp",
                  "disk", "worker"):
            assert k in info
        assert info["disk"]["free_bytes"] > 0
        assert info["stub"] is True


def test_upload_job_sse_done_and_track(env):
    with client() as c:
        up = upload(c)
        assert up["key"].startswith("sha256-") and up["title"] == "我的歌" and up["cached"] is False
        r = c.post("/api/jobs", json={"upload_id": up["upload_id"]})
        assert r.status_code == 200, r.text
        job = r.json()
        assert job["key"] == up["key"] and job["cached"] is False

        evs = events(c, job["job_id"])
        names = [e for e, _ in evs]
        assert names[-1] == "done" and names.count("done") == 1 and "error" not in names
        progress = [d for e, d in evs if e == "progress"]
        stages = {d["stage"] for d in progress}
        assert {"separate", "rhythm", "evidence", "decode", "output"} <= stages
        overall = [d["overall"] for d in progress]
        assert overall == sorted(overall) and all(0.0 <= o <= 1.0 for o in overall)
        assert all({"stage", "pct", "msg", "eta_s"} <= set(d) for d in progress)
        done = evs[-1][1]
        assert done["analysis_url"] == f"/api/tracks/{up['key']}"
        assert done["display_url"] == f"/api/tracks/{up['key']}/display"

        a = schema.Analysis.model_validate(c.get(done["analysis_url"]).json())
        assert a.source.key == up["key"] and a.source.type == "upload" and len(a.chords) > 0
        assert a.audio.play == f"/api/tracks/{up['key']}/audio"
        d = schema.Display.model_validate(c.get(done["display_url"]).json())
        assert set(d.variants["0:auto"].chords) == {ch.id for ch in a.chords}

        tracks = c.get("/api/tracks").json()["tracks"]
        assert [(t["key"], t["status"], t["title"]) for t in tracks] == [(up["key"], "ready", "我的歌")]
        assert tracks[0]["last_opened"] is not None            # GET /api/tracks/{key} marks it opened
        assert not (config.CACHE_DIR / "_uploads" / up["upload_id"]).exists()
        snap = c.get(f"/api/jobs/{job['job_id']}").json()
        assert snap["state"] == "done" and snap["attempts"] == 1 and snap["source"].startswith("upload:")

        # the stream of a finished job replays `done`
        assert [e for e, _ in events(c, job["job_id"])] == ["done"]


def test_cached_song_returns_a_finished_job(env):
    put_cached_song(HUIJIA)

    def must_not_run(*a, **k):
        raise AssertionError("the worker must not run for a cached song")

    with client(analyze_fn=must_not_run) as c:
        r = c.post("/api/jobs", json={"source": HUIJIA_URL}).json()
        assert r["key"] == HUIJIA and r["cached"] is True and r["state"] == "done"
        evs = events(c, r["job_id"])
        assert [e for e, _ in evs] == ["done"] and evs[0][1]["cached"] is True
        assert c.get(f"/api/tracks/{HUIJIA}").status_code == 200
        # a song from an older engine version is re-analysed
        put_cached_song(HUIJIA, engine_version="0.0.1")
        r2 = c.post("/api/jobs", json={"source": HUIJIA_URL}).json()
        assert r2["cached"] is False


def test_library_reconciles_existing_cache_on_start(env):
    put_cached_song("yt-AAAAAAAAAAA")
    with client() as c:
        keys = [t["key"] for t in c.get("/api/tracks").json()["tracks"]]
        assert keys == ["yt-AAAAAAAAAAA"]


def test_audio_range_requests(env):
    data = os.urandom(50_000)
    key = "yt-AAAAAAAAAAA"
    put_cached_song(key, audio=data)
    url = f"/api/tracks/{key}/audio"
    with client() as c:
        r = c.get(url)
        assert r.status_code == 200 and r.content == data
        assert r.headers["content-type"] == "audio/mp4"
        assert r.headers["accept-ranges"] == "bytes"
        r = c.get(url, headers={"Range": "bytes=0-99"})
        assert r.status_code == 206 and r.content == data[:100]
        assert r.headers["content-range"] == f"bytes 0-99/{len(data)}"
        assert r.headers["content-type"] == "audio/mp4"
        r = c.get(url, headers={"Range": "bytes=1000-"})
        assert r.status_code == 206 and r.content == data[1000:]
        assert r.headers["content-range"] == f"bytes 1000-{len(data) - 1}/{len(data)}"
        r = c.get(url, headers={"Range": "bytes=-500"})
        assert r.status_code == 206 and r.content == data[-500:]
        r = c.get(url, headers={"Range": f"bytes={len(data) + 10}-"})
        assert r.status_code == 416
        r = c.head(url)
        assert r.status_code == 200 and int(r.headers["content-length"]) == len(data)
        assert c.get(url + "?mix=novocals").status_code == 409        # no stems in this fixture (test_server_mixes)
        assert c.get(url + "?mix=karaoke").status_code == 400
        assert c.get("/api/tracks/yt-BBBBBBBBBBB/audio").status_code == 404
        assert c.get("/api/tracks/..%2F..%2Fsecret/audio").status_code in (400, 404)


def test_delete_track(env):
    key = "yt-AAAAAAAAAAA"
    put_cached_song(key)
    with client() as c:
        assert c.get(f"/api/tracks/{key}").status_code == 200
        r = c.delete(f"/api/tracks/{key}")
        assert r.status_code == 200 and r.json() == {"deleted": key}
        assert not (config.CACHE_DIR / key).exists()
        assert c.get(f"/api/tracks/{key}").status_code == 404
        assert c.get("/api/tracks").json()["tracks"] == []
        assert c.delete(f"/api/tracks/{key}").status_code == 404
        assert c.delete("/api/tracks/not-a-key").status_code == 400


def test_rename_track(env):
    key = "yt-AAAAAAAAAAA"
    put_cached_song(key)
    with client() as c:
        r = c.patch(f"/api/tracks/{key}", json={"title": "回家 (Eternity Girls)", "artist": "永恆音樂事工"})
        assert r.status_code == 200 and r.json()["renamed"] is True
        src = c.get(f"/api/tracks/{key}").json()["source"]
        assert src["title"] == "回家 (Eternity Girls)" and src["artist"] == "永恆音樂事工"
        c.app.state.ctx.sync_track(key)                         # a re-analysis keeps the rename
        assert c.get("/api/tracks").json()["tracks"][0]["title"] == "回家 (Eternity Girls)"
        assert c.patch(f"/api/tracks/{key}", json={"title": "  "}).status_code == 400
        assert c.patch("/api/tracks/yt-BBBBBBBBBBB", json={"title": "x"}).status_code == 404


def test_cancel_queued_and_running_jobs(env):
    started = threading.Event()

    def slow(source, mode="full", params=None, progress=None, device="cuda", write=True):
        started.set()
        for i in range(3000):
            progress("separate", i / 3000, "Separating stems")
            time.sleep(0.01)
        raise AssertionError("should have been cancelled")

    with client(analyze_fn=slow) as c:
        a = c.post("/api/jobs", json={"source": "https://youtu.be/AAAAAAAAAAA"}).json()
        b = c.post("/api/jobs", json={"source": "https://youtu.be/BBBBBBBBBBB"}).json()
        dup = c.post("/api/jobs", json={"source": "https://www.youtube.com/watch?v=BBBBBBBBBBB"}).json()
        assert dup["job_id"] == b["job_id"]                     # no duplicate job for the same song
        assert started.wait(10)
        assert c.get(f"/api/jobs/{b['job_id']}").json()["position"] == 1
        assert c.delete("/api/tracks/yt-AAAAAAAAAAA").status_code == 409

        assert c.delete(f"/api/jobs/{b['job_id']}").json()["state"] == "cancelled"
        assert events(c, b["job_id"])[-1] == ("error", {"job_id": b["job_id"], "code": "cancelled",
                                                        "msg": "The analysis was cancelled.", "hint": ""})
        c.delete(f"/api/jobs/{a['job_id']}")
        evs = events(c, a["job_id"], timeout=10)
        assert evs[-1][0] == "error" and evs[-1][1]["code"] == "cancelled"
        wait_ready(c)
        assert c.get("/api/tracks").json()["tracks"] == []      # cancelled new songs leave no rows


def test_worker_crash_is_recovered_and_job_retried(env):
    calls = {"n": 0}

    def crash_once(source, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SystemExit(3)                                 # ends the worker thread like a dead process
        return stub.analyze(source, **kw)

    with client(analyze_fn=crash_once) as c:
        up = upload(c)
        job = c.post("/api/jobs", json={"upload_id": up["upload_id"]}).json()
        evs = events(c, job["job_id"])
        assert evs[-1][0] == "done"
        assert any("retrying" in d["msg"] for e, d in evs if e == "progress")
        snap = c.get(f"/api/jobs/{job['job_id']}").json()
        assert snap["attempts"] == 2 and snap["state"] == "done"
        assert calls["n"] == 2
        assert wait_ready(c)["worker"] == "ready"


def test_worker_crashing_twice_fails_the_job(env):
    def always_crash(source, **kw):
        raise SystemExit(3)

    with client(analyze_fn=always_crash) as c:
        job = c.post("/api/jobs", json={"source": "https://youtu.be/AAAAAAAAAAA"}).json()
        evs = events(c, job["job_id"])
        assert evs[-1][0] == "error" and evs[-1][1]["code"] == "worker_crashed"
        assert wait_ready(c)["worker"] == "ready"               # the next job gets a fresh worker


def test_ingest_errors_reach_the_client(env):
    with client() as c:                                         # stub: uncached YouTube links → ingest error
        job = c.post("/api/jobs", json={"source": "https://youtu.be/AAAAAAAAAAA"}).json()
        name, data = events(c, job["job_id"])[-1]
        assert name == "error" and data["code"] == "unavailable" and data["hint"]
        assert "vpn" not in (data["msg"] + data["hint"]).lower()
        assert c.get("/api/tracks").json()["tracks"][0]["status"] == "failed"


def test_process_worker_crash_recovery(env, monkeypatch):
    """The real spawned worker process (stub pipeline): os._exit on the first job, then a retry."""
    monkeypatch.setenv("NCHORD_STUB_CRASH_ONCE", str(env / "crash.marker"))
    with client(worker="process", analyze=STUB_ANALYZE, analyze_fn=None) as c:
        up = upload(c)
        job = c.post("/api/jobs", json={"upload_id": up["upload_id"]}).json()
        evs = events(c, job["job_id"], timeout=120)
        assert evs[-1][0] == "done", evs[-3:]
        assert (env / "crash.marker").exists()
        assert c.get(f"/api/jobs/{job['job_id']}").json()["attempts"] == 2
        a = schema.Analysis.model_validate(c.get(f"/api/tracks/{up['key']}").json())
        assert a.source.key == up["key"]


def test_bad_requests(env):
    with client() as c:
        def post(body):
            r = c.post("/api/jobs", json=body)
            return r.status_code, r.json().get("code")
        assert post({"source": "C:\\Users\\User\\secret.wav"}) == (400, "bad_source")
        assert post({"source": "file:///C:/secret.wav"}) == (400, "bad_source")
        assert post({"source": "https://www.deezer.com/track/1"}) == (400, "drm")
        assert post({}) == (400, "bad_source")
        assert post({"source": "https://youtu.be/AAAAAAAAAAA", "params": {"nope": 1}}) == (400, "bad_params")
        assert post({"upload_id": "0123456789abcdef"}) == (404, "no_upload")
        assert post({"upload_id": "../../x"}) == (400, "bad_upload")
        assert c.get("/api/jobs/nope/events").status_code == 404
        assert c.delete("/api/jobs/nope").status_code == 404
        r = c.post("/api/uploads", content=b"raw", headers={"content-type": "audio/wav"})
        assert r.status_code == 415
        r = c.post("/api/uploads", files={"other": ("a.wav", b"123", "audio/wav")})
        assert r.status_code == 400 and r.json()["code"] == "bad_upload"


def test_disk_full_refuses_new_jobs(env):
    put_cached_song(HUIJIA)
    full = storage.Policy(refuse_free_bytes=10**18, warn_free_bytes=10**18)
    with client(policy=full) as c:
        r = c.post("/api/jobs", json={"source": "https://youtu.be/AAAAAAAAAAA"})
        assert r.status_code == 507 and r.json()["code"] == "disk_full"
        assert c.post("/api/uploads", files={"file": ("a.wav", wav_bytes(), "audio/wav")}).status_code == 507
        assert c.post("/api/jobs", json={"source": HUIJIA_URL}).json()["cached"] is True   # nothing to compute
        assert c.get("/api/system/info").json()["disk"]["warn"] is True


def test_exports(env):
    key = "yt-AAAAAAAAAAA"
    a = put_cached_song(key)
    with client() as c:
        r = c.get(f"/api/tracks/{key}/export/lab?detail=0.8")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
        assert "attachment" in r.headers["content-disposition"]
        assert r.text.splitlines() == lab_lines(a, 0.8)
        assert len(c.get(f"/api/tracks/{key}/export/lab?detail=0").text.splitlines()) <= len(lab_lines(a, 0.8))
        j = c.get(f"/api/tracks/{key}/export/json")
        assert schema.Analysis.model_validate(j.json()).source.key == key
        assert c.get(f"/api/tracks/{key}/export/midi-daw").status_code == 501
        assert c.get(f"/api/tracks/{key}/export/lab?transpose=2").status_code == 501
        assert c.get(f"/api/tracks/{key}/export/lab?detail=3").status_code == 422
        assert c.get(f"/api/tracks/{key}/export/xyz").status_code == 404


def test_redecode_calls_pipeline_when_present(env, monkeypatch):
    from nathan_chordify import pipeline
    key = "yt-AAAAAAAAAAA"
    a = put_cached_song(key)
    with client() as c:
        monkeypatch.delattr(pipeline, "redecode", raising=False)
        r = c.post(f"/api/tracks/{key}/redecode", json={"params": {"lam_scale": 0.5}})
        assert r.status_code == 501 and r.json()["code"] == "not_implemented"
        seen = {}

        def fake(k, params, region, *, write=False):
            seen.update(key=k, params=params, region=region, write=write)
            return a
        monkeypatch.setattr(pipeline, "redecode", fake, raising=False)
        r = c.post(f"/api/tracks/{key}/redecode", json={"params": {"lam_scale": 0.5}, "region": [10.0, 20.0]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["analysis"]["source"]["key"] == key
        d = schema.Display.model_validate(body["display"])
        assert set(d.variants["3:auto"].chords) == {ch.id for ch in a.chords}
        assert seen == {"key": key, "params": {"lam_scale": 0.5}, "region": (10.0, 20.0), "write": False}
        c.post(f"/api/tracks/{key}/redecode", json={"write": True})
        assert seen["write"] is True and seen["params"] is None and seen["region"] is None
        assert c.post(f"/api/tracks/{key}/redecode", json={"params": {"bogus": 1}}).status_code == 400
        assert c.post(f"/api/tracks/{key}/redecode", json={"region": [20, 10]}).status_code == 400
        assert c.post("/api/tracks/yt-BBBBBBBBBBB/redecode", json={}).status_code == 404

        def no_state(k, params, region):
            raise FileNotFoundError("no decode state")
        monkeypatch.setattr(pipeline, "redecode", no_state)
        r = c.post(f"/api/tracks/{key}/redecode", json={})
        assert r.status_code == 409 and r.json()["code"] == "no_decode_state"


def test_later_milestone_routes_are_501(env):
    with client() as c:
        for method, url in (("GET", "/api/search?q=x"),
                            ("GET", "/api/tracks/yt-AAAAAAAAAAA/lyrics"), ("GET", "/api/setlists"),
                            ("DELETE", "/api/setlists/1"), ("GET", "/api/lyrics/lrclib")):
            assert c.request(method, url).status_code == 501, url


def test_update_downloader_restarts_the_worker(env):
    with client(pip_cmd=[sys.executable, "-c", "print('pip ok')"]) as c:
        r = c.post("/api/system/update-downloader").json()
        assert r["ok"] is True and "pip ok" in r["log"] and r["after"] == r["before"]
        assert wait_ready(c)["worker"] == "ready"
        up = upload(c)                                          # jobs run again afterwards
        job = c.post("/api/jobs", json={"upload_id": up["upload_id"]}).json()
        assert events(c, job["job_id"])[-1][0] == "done"


def test_web_ui_spa_fallback_and_headers(env):
    dist = env / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>", "utf-8")
    (dist / "assets" / "app-abc123.js").write_text("console.log(1)", "utf-8")
    with client(web_dist=dist) as c:
        r = c.get("/")
        assert r.status_code == 200 and "<title>app</title>" in r.text
        csp = r.headers["content-security-policy"]
        assert "'wasm-unsafe-eval'" in csp and "media-src 'self' blob: data:" in csp
        assert c.get("/library/some/deep/link").text == r.text
        js = c.get("/assets/app-abc123.js")
        assert js.headers["content-type"].startswith("text/javascript")
        assert "immutable" in js.headers["cache-control"]
        assert c.get("/missing.js").status_code == 404
        assert c.get("/api/unknown").json()["code"] == "not_found"
        assert c.get("/health", headers={"host": "evil.example"}).status_code == 400   # DNS-rebinding guard


# ---- storage ------------------------------------------------------------------------------------

def _song(cache: Path, key: str, stems: int, evidence: int, wav: int) -> Path:
    d = cache / key
    (d / "stems").mkdir(parents=True)
    (d / "evidence").mkdir()
    (d / "stems" / "vocals.flac").write_bytes(b"s" * stems)
    (d / "evidence" / "lv@mix.v1.npz").write_bytes(b"e" * evidence)
    (d / "audio.wav").write_bytes(b"w" * wav)
    for keep in ("analysis.json", "display.json", "play.m4a", "meta.json", "source.webm"):
        (d / keep).write_bytes(b"k" * 10)
    return d


def test_storage_retention_and_cap(tmp_path):
    cache = tmp_path / "cache"
    now = time.time()
    day = 86400.0
    old = _song(cache, "yt-OLDOLDOLDOL", 1000, 500, 800)       # not opened for 40 days
    a = _song(cache, "yt-AAAAAAAAAAA", 1000, 500, 800)         # opened 5 days ago
    b = _song(cache, "yt-BBBBBBBBBBB", 1000, 500, 800)         # opened 1 day ago
    busy = _song(cache, "yt-BUSYBUSYBUS", 1000, 500, 800)      # analysing now: protected
    (cache / "_uploads" / "old").mkdir(parents=True)
    (cache / "_uploads" / "old" / "x.wav").write_bytes(b"u" * 100)
    os.utime(cache / "_uploads" / "old", (now - 2 * day, now - 2 * day))
    (cache / "_uploads" / "new").mkdir()
    (cache / "_trash" / "gone").mkdir(parents=True)
    (cache / "_trash" / "gone" / "f").write_bytes(b"t" * 50)
    opened = {"yt-OLDOLDOLDOL": now - 40 * day, "yt-AAAAAAAAAAA": now - 5 * day, "yt-BBBBBBBBBBB": now - 1 * day,
              "yt-BUSYBUSYBUS": now - 90 * day}
    # 4 songs × 2350 B + upload 100 + trash 50 = 9550. Retention frees old's stems + evidence (1500),
    # clean-up 150 → 7900; the 6000 B cap then takes a's stems, then b's (5900).
    pol = storage.Policy(retention_days=30, cap_bytes=6000)
    rep = storage.enforce(cache, opened, pol, protect={"yt-BUSYBUSYBUS"}, now=now)

    assert not (old / "stems").exists() and not (old / "evidence").exists()          # retention
    reasons = {(r["key"], r["tier"]): r["reason"] for r in rep.removed}
    assert reasons[("yt-OLDOLDOLDOL", "stems")] == "retention"
    assert reasons[("yt-AAAAAAAAAAA", "stems")] == reasons[("yt-BBBBBBBBBBB", "stems")] == "cap"
    assert (old / "audio.wav").exists()                                               # audio is the last tier
    assert (a / "evidence").exists() and (b / "evidence").exists()
    assert rep.total_after == 5900 <= pol.cap_bytes
    assert all((busy / p).exists() for p in ("stems", "evidence", "audio.wav"))      # protected
    for d in (old, a, b, busy):                                                       # always kept
        assert all((d / k).exists() for k in ("analysis.json", "display.json", "play.m4a", "meta.json",
                                              "source.webm"))
    assert not (cache / "_uploads" / "old").exists() and (cache / "_uploads" / "new").exists()
    assert not (cache / "_trash" / "gone").exists()


def test_storage_cap_evicts_tiers_in_order(tmp_path):
    cache = tmp_path / "cache"
    now = time.time()
    a = _song(cache, "yt-AAAAAAAAAAA", 1000, 1000, 1000)
    b = _song(cache, "yt-BBBBBBBBBBB", 1000, 1000, 1000)
    opened = {"yt-AAAAAAAAAAA": now - 3 * 86400, "yt-BBBBBBBBBBB": now - 86400}
    keep = 2 * 50
    # cap allows both songs' evidence + audio: only stems go (a's first, then b's)
    rep = storage.enforce(cache, opened, storage.Policy(cap_bytes=4000 + keep + 1000), now=now)
    assert [(r["key"], r["tier"]) for r in rep.removed] == [("yt-AAAAAAAAAAA", "stems")]
    rep = storage.enforce(cache, opened, storage.Policy(cap_bytes=2000 + keep), now=now)
    assert [(r["key"], r["tier"]) for r in rep.removed] == [("yt-BBBBBBBBBBB", "stems"),
                                                           ("yt-AAAAAAAAAAA", "evidence"),
                                                           ("yt-BBBBBBBBBBB", "evidence")]
    rep = storage.enforce(cache, opened, storage.Policy(cap_bytes=1000 + keep), now=now)
    assert [(r["key"], r["tier"]) for r in rep.removed] == [("yt-AAAAAAAAAAA", "audio")]
    assert (b / "audio.wav").exists() and (a / "play.m4a").exists()


def test_storage_routes(env):
    put_cached_song("yt-AAAAAAAAAAA")
    with client(policy=storage.Policy(cap_bytes=10**12)) as c:
        s = c.get("/api/system/storage").json()
        assert s["cap_bytes"] == 10**12 and s["retention_days"] == 30
        assert [t["key"] for t in s["tracks"]] == ["yt-AAAAAAAAAAA"]
        rep = c.post("/api/system/storage/cleanup").json()
        assert rep["freed"] == 0 and rep["total_after"] > 0


# ---- real GPU (NCHORD_GPU_TESTS=1) --------------------------------------------------------------

def _real_settings(**kw) -> ServerSettings:
    # Never evict anything from the real cache.
    kw.setdefault("policy", storage.Policy(retention_days=1e6, cap_bytes=10**15, refuse_free_bytes=0))
    return ServerSettings(storage_on_start=False, allowed_hosts=("testserver",), **kw)


@gpu
def test_gpu_cached_song_through_real_worker(tmp_path, monkeypatch):
    """The real pre-warmed GPU worker; 回家 is already analysed, so the job finishes via the cache."""
    if not (REAL_CACHE / HUIJIA / "analysis.json").exists():
        pytest.skip("cache/yt-Y7boUFbM76k is not analysed")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")      # temp library, real cache
    t0 = time.monotonic()
    with TestClient(create_app(_real_settings())) as c:
        wait_ready(c, 300)
        warm = time.monotonic() - t0
        info = c.get("/api/system/info").json()
        assert info["gpu"] and "RTX" in info["gpu"]["name"], info
        assert not info["worker"]["prewarm_errors"], info["worker"]
        t1 = time.monotonic()
        r = c.post("/api/jobs", json={"source": HUIJIA_URL}).json()
        assert r["cached"] is True and r["key"] == HUIJIA
        assert [e for e, _ in events(c, r["job_id"])] == ["done"]
        a = schema.Analysis.model_validate(c.get(f"/api/tracks/{HUIJIA}").json())
        rng = c.get(f"/api/tracks/{HUIJIA}/audio", headers={"Range": "bytes=0-1023"})
        assert rng.status_code == 206 and len(rng.content) == 1024
        print(f"\npre-warm {warm:.1f}s ({info['worker']['prewarm_s']}s in worker), GPU {info['gpu']['name']}, "
              f"cached job {time.monotonic() - t1:.2f}s, {len(a.chords)} chords")
        before = (REAL_CACHE / HUIJIA / "analysis.json").stat().st_mtime
        t2 = time.monotonic()
        r = c.post(f"/api/tracks/{HUIJIA}/redecode", json={"region": [60.0, 75.0]})   # preview: no write
        if r.status_code != 501:
            assert r.status_code == 200, r.text
            red = schema.Analysis.model_validate(r.json()["analysis"])
            schema.Display.model_validate(r.json()["display"])
            assert (REAL_CACHE / HUIJIA / "analysis.json").stat().st_mtime == before
            print(f"redecode preview {time.monotonic() - t2:.1f}s: {len(red.chords)} chords, "
                  f"{sum(ch.suggestion for ch in red.chords)} suggestions in 60-75 s")


@gpu
def test_gpu_full_pipeline_on_cache_copy(tmp_path, monkeypatch):
    """A real analysis through the worker (demucs, Beat This!, lv-chordia on the GPU) on a copy of
    回家's downloaded audio, so the real cache is untouched."""
    src = REAL_CACHE / HUIJIA
    if not (src / "audio.wav").exists():
        pytest.skip("cache/yt-Y7boUFbM76k has no audio")
    cache = tmp_path / "cache"
    (cache / HUIJIA).mkdir(parents=True)
    for name in ("meta.json", "audio.wav", "play.m4a", *(p.name for p in src.glob("source.*"))):
        shutil.copyfile(src / name, cache / HUIJIA / name)
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    with TestClient(create_app(_real_settings())) as c:
        wait_ready(c, 300)
        t0 = time.monotonic()
        r = c.post("/api/jobs", json={"source": HUIJIA_URL}).json()
        assert r["cached"] is False
        evs = events(c, r["job_id"], timeout=600)
        assert evs[-1][0] == "done", evs[-3:]
        stages = {d["stage"] for e, d in evs if e == "progress"}
        assert {"separate", "rhythm", "evidence", "decode", "output"} <= stages
        a = schema.Analysis.model_validate(c.get(f"/api/tracks/{HUIJIA}").json())
        assert a.engine_version == ENGINE_VERSION and len(a.chords) > 50
        print(f"\nfull analysis through the worker: {time.monotonic() - t0:.1f}s, runtime {a.stats.runtime_s}, "
              f"key {a.key.global_.tonic} {a.key.global_.mode}, {a.tempo.bpm} BPM, {len(a.chords)} chords")
