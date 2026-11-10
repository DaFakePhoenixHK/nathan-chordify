"""RemoteGuard: a phone must show the secret code; this computer doesn't have to."""
from __future__ import annotations

import queue
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from nathan_chordify import config
from nathan_chordify.server import remote, tunnel
from nathan_chordify.server.app import create_app

from test_server import env, put_cached_song, settings  # noqa: F401  (env is a fixture)

KEY = "yt-PHONETEST01"
TUNNEL = {"host": "abc.example.test", "cf-connecting-ip": "203.0.113.9"}
ORIGIN = "https://dafakephoenixhk.github.io"


@pytest.fixture(autouse=True)
def app_published(monkeypatch):
    """No network in tests: the phone app page counts as published unless a test says otherwise."""
    monkeypatch.setattr(remote, "app_online", lambda timeout=4.0: True)


@pytest.fixture
def web(env):
    s = settings(public_url="https://abc.example.test", allowed_hosts=("testserver", "127.0.0.1", "abc.example.test"))
    with TestClient(create_app(s)) as c:
        yield c


def code() -> str:
    return remote.phone_token(config.DATA_DIR)


def bearer() -> dict:
    return {**TUNNEL, "authorization": f"Bearer {code()}"}


def test_this_computer_needs_no_code(web):
    assert web.get("/api/tracks").status_code == 200


def test_tunnel_without_code_is_refused(web):
    r = web.get("/api/tracks", headers=TUNNEL)
    assert r.status_code == 401 and r.json()["code"] == "need_code"


def test_wrong_code_is_refused(web):
    assert web.get("/api/tracks", headers={**TUNNEL, "authorization": "Bearer nope"}).status_code == 401


def test_right_code_is_accepted(web):
    assert web.get("/api/tracks", headers=bearer()).status_code == 200


def test_loopback_host_cannot_hide_a_forwarded_request(web):
    assert web.get("/api/tracks", headers={"host": "127.0.0.1", "cf-connecting-ip": "203.0.113.9"}).status_code == 401


def test_foreign_page_is_not_local(web):
    assert web.get("/api/tracks", headers={"origin": "https://evil.example"}).status_code == 401


def test_health_and_app_files_are_open(web):
    assert web.get("/health", headers=TUNNEL).status_code == 200
    assert web.get("/", headers=TUNNEL).status_code == 200


@pytest.mark.parametrize("path", ["/api/phone/status", "/api/system/info", "/api/docs", "/api/openapi.json"])
def test_machine_routes_stay_local_even_with_the_code(web, path):
    r = web.get(path, headers=bearer())
    assert r.status_code == 403 and r.json()["code"] == "local_only"


def test_update_and_cleanup_stay_local(web):
    assert web.post("/api/system/update-downloader", headers=bearer()).status_code == 403
    assert web.post("/api/system/storage/cleanup", headers=bearer()).status_code == 403


def test_query_code_works_for_audio_only(web):
    put_cached_song(KEY)
    q = {"token": code()}
    assert web.get(f"/api/tracks/{KEY}/audio", params=q, headers=TUNNEL).status_code == 200
    assert web.get(f"/api/tracks/{KEY}", params=q, headers=TUNNEL).status_code == 401
    assert web.post(f"/api/tracks/{KEY}/audio", params=q, headers=TUNNEL).status_code == 401


def test_audio_answers_range_requests_to_a_phone(web):
    put_cached_song(KEY)
    r = web.get(f"/api/tracks/{KEY}/audio", params={"token": code()}, headers={**TUNNEL, "range": "bytes=0-1"})
    assert r.status_code == 206 and r.headers["content-range"].startswith("bytes 0-1/")


def test_preflight_from_the_phone_app(web):
    r = web.options("/api/tracks", headers={**TUNNEL, "origin": ORIGIN, "access-control-request-method": "GET",
                                            "access-control-request-headers": "authorization"})
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == ORIGIN
    assert "authorization" in r.headers["access-control-allow-headers"]


def test_preflight_from_another_site_gets_no_permission(web):
    r = web.options("/api/tracks", headers={**TUNNEL, "origin": "https://evil.example",
                                            "access-control-request-method": "GET"})
    assert "access-control-allow-origin" not in r.headers


def test_refusal_is_readable_by_the_phone_app(web):
    r = web.get("/api/tracks", headers={**TUNNEL, "origin": ORIGIN})
    assert r.status_code == 401 and r.headers["access-control-allow-origin"] == ORIGIN


def test_responses_are_readable_by_the_phone_app(web):
    r = web.get("/api/tracks", headers={**bearer(), "origin": ORIGIN})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == ORIGIN


def test_status_gives_this_computer_the_pairing_link(web):
    d = web.get("/api/phone/status").json()
    assert d["token"] == code() and d["public_url"] == "https://abc.example.test"
    u = urlsplit(d["pair_link"])
    q = parse_qs(u.fragment.split("?", 1)[1])
    assert u.netloc == "dafakephoenixhk.github.io"
    assert q["s"] == ["https://abc.example.test"] and q["t"] == [code()]


def test_reset_makes_the_old_code_useless(web):
    old = code()
    new = web.post("/api/phone/reset").json()["token"]
    assert new != old
    assert web.get("/api/tracks", headers={**TUNNEL, "authorization": f"Bearer {old}"}).status_code == 401
    assert web.get("/api/tracks", headers={**TUNNEL, "authorization": f"Bearer {new}"}).status_code == 200


def test_code_survives_a_restart(env):
    assert remote.phone_token(config.DATA_DIR) == remote.phone_token(config.DATA_DIR)


def test_no_public_url_means_no_link(env):
    with TestClient(create_app(settings())) as c:
        d = c.get("/api/phone/status").json()
    assert d["public_url"] is None and d["pair_link"] is None


# ---- the phone link (Cloudflare quick tunnel), with a stand-in for cloudflared -------------------

class FakeCloudflared:
    """Prints what cloudflared prints, then runs until stopped (or ends by itself with end=True)."""

    def __init__(self, lines, end=False):
        self.q = queue.Queue()
        for line in lines:
            self.q.put(line)
        if end:
            self.q.put(None)
        self.returncode = None

    @property
    def stderr(self):
        while True:
            line = self.q.get()
            if line is None:
                self.returncode = self.returncode if self.returncode is not None else 1
                return
            yield line

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        self.q.put(None)

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


BANNER = ["INF Requesting new quick Tunnel on trycloudflare.com...",
          "INF |  https://brave-lion-sings-loud.trycloudflare.com  |",
          "INF Registered tunnel connection connIndex=0 location=syd01 protocol=http2"]


def fake_factory(runs: list, lines=BANNER, end=False):
    def make(port):
        def popen(cmd, **kw):
            assert "--url" in cmd and cmd[cmd.index("--url") + 1] == f"http://127.0.0.1:{port}"
            proc = FakeCloudflared(lines, end=end)
            runs.append(proc)
            return proc
        exe = config.DATA_DIR / "cloudflared.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"")
        return tunnel.Tunnel(port, exe=exe, popen=popen, retry_s=0.05)
    return make


def wait_for(fn, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError("timed out")


def link_client(runs, **kw):
    s = settings(port=8764, tunnel_factory=fake_factory(runs, **{k: v for k, v in kw.items() if k in ("lines", "end")}),
                 **{k: v for k, v in kw.items() if k not in ("lines", "end")})
    return TestClient(create_app(s))


def test_turning_the_link_on_gives_a_code_to_scan(env):
    runs = []
    with link_client(runs) as c:
        assert c.get("/api/phone/status").json()["link"]["state"] == "off"
        c.post("/api/phone/link", json={"on": True})
        d = wait_for(lambda: (lambda j: j if j["link"]["state"] == "on" else None)(c.get("/api/phone/status").json()))
        assert d["public_url"] == "https://brave-lion-sings-loud.trycloudflare.com"
        assert "s=https%3A%2F%2Fbrave-lion-sings-loud.trycloudflare.com" in d["pair_link"]


def test_the_address_only_counts_once_cloudflare_says_it_is_connected(env):
    runs = []
    with link_client(runs, lines=BANNER[:2]) as c:
        c.post("/api/phone/link", json={"on": True})
        wait_for(lambda: runs)
        time.sleep(0.1)
        d = c.get("/api/phone/status").json()
        assert d["link"]["state"] == "starting" and d["pair_link"] is None


def test_the_link_answers_phones_with_the_code_only(env):
    runs = []
    with link_client(runs) as c:
        c.post("/api/phone/link", json={"on": True})
        wait_for(lambda: c.get("/api/phone/status").json()["public_url"])
        phone = {"host": "brave-lion-sings-loud.trycloudflare.com", "cf-connecting-ip": "203.0.113.9"}
        assert c.get("/api/tracks", headers=phone).status_code == 401
        assert c.get("/api/tracks", headers={**phone, "authorization": f"Bearer {code()}"}).status_code == 200


def test_turning_it_off_stops_cloudflared_and_is_remembered(env):
    runs = []
    with link_client(runs) as c:
        c.post("/api/phone/link", json={"on": True})
        wait_for(lambda: c.get("/api/phone/status").json()["public_url"])
        d = c.post("/api/phone/link", json={"on": False}).json()
        assert d["link"]["state"] == "off" and d["public_url"] is None and runs[0].returncode == 0
    assert not tunnel.wanted(config.DATA_DIR)


def test_the_link_comes_back_after_a_restart(env):
    runs = []
    with link_client(runs) as c:
        c.post("/api/phone/link", json={"on": True})
        wait_for(lambda: c.get("/api/phone/status").json()["public_url"])
    assert runs[0].returncode == 0                      # the server stopping ends cloudflared
    with link_client(runs) as c:
        assert wait_for(lambda: c.get("/api/phone/status").json()["public_url"])
    assert len(runs) == 2


def test_cloudflared_is_restarted_when_it_stops_by_itself(env):
    runs = []
    with link_client(runs, end=True) as c:
        c.post("/api/phone/link", json={"on": True})
        wait_for(lambda: len(runs) >= 2)


def test_missing_program_is_explained(env):
    t = tunnel.Tunnel(8764, exe=config.DATA_DIR / "nothing.exe")
    t.start()
    assert t.status()["state"] == "error" and "missing" in t.status()["error"]


def test_without_a_published_app_the_computer_serves_it(env, monkeypatch):
    monkeypatch.setattr(remote, "app_online", lambda timeout=4.0: False)
    with TestClient(create_app(settings(public_url="https://abc.example.test"))) as c:
        d = c.get("/api/phone/status").json()
    assert d["app_ready"] is False
    assert d["pair_link"] == f"https://abc.example.test/#/pair?t={code()}"
