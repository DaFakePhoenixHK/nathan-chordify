"""WP1 tests: URL normalisation, upload keys, ffmpeg decode (no network)."""
from __future__ import annotations

import wave

import numpy as np
import pytest

from nathan_chordify import config, ingest
from nathan_chordify.ingest import IngestError, resolve, youtube_id

ID = "Y7boUFbM76k"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={ID}",
    f"https://www.youtube.com/watch?v={ID}&list=RD{ID}&start_radio=1&pp=ygUM5Zue5a62IGNob3JkoAcB",
    f"https://www.youtube.com/watch?v={ID}&list=RD{ID}&start_radio=1",
    f"https://youtu.be/{ID}?si=abc",
    f"https://www.youtube.com/shorts/{ID}",
    f"https://music.youtube.com/watch?v={ID}&feature=share",
    f"https://m.youtube.com/watch?t=30&v={ID}",
    f"https://www.youtube.com/embed/{ID}",
    f"youtube.com/watch?v={ID}",
])
def test_youtube_forms_resolve_to_one_key(url):
    key, meta = resolve(url)
    assert key == f"yt-{ID}"
    assert meta["source_url"] == f"https://www.youtube.com/watch?v={ID}"


def test_non_youtube_hosts_are_not_parsed_as_youtube():
    assert youtube_id("https://example.com/watch?v=Y7boUFbM76k") is None
    assert youtube_id("https://www.youtube.com/watch?v=short") is None


def test_deezer_is_rejected():
    with pytest.raises(IngestError) as e:
        resolve("https://www.deezer.com/track/12345")
    assert e.value.code == "drm"


def test_garbage_is_rejected():
    with pytest.raises(IngestError) as e:
        resolve("not a link")
    assert e.value.code == "bad_source"


def _write_sine_wav(path, sr=22050, seconds=2.0):
    t = np.arange(int(sr * seconds)) / sr
    y = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def test_upload_key_is_sha256_and_fetch_decodes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    src = tmp_path / "my song.wav"
    _write_sine_wav(src)
    key, meta = resolve(str(src))
    assert key.startswith("sha256-") and len(key) == len("sha256-") + 64
    assert meta["source_type"] == "upload"

    asset = ingest.fetch(str(src))
    assert asset.key == key
    assert asset.wav.exists() and asset.play.exists()
    assert abs(asset.duration_s - 2.0) < 0.05
    with wave.open(str(asset.wav)) as w:
        assert w.getframerate() == 44100 and w.getnchannels() == 2 and w.getsampwidth() == 2
    # second call is served from the cache
    again = ingest.fetch(str(src))
    assert again.wav == asset.wav
