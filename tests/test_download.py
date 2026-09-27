from pathlib import Path

from medisense.data.loaders import _download


def test_download_discards_invalid_cached_file_before_retry(tmp_path, monkeypatch):
    destination = tmp_path / "archive.zip"
    destination.write_bytes(b"partial")

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, _size):
            if hasattr(self, "sent"): return b""
            self.sent = True
            return b"complete"

    monkeypatch.setattr("medisense.data.loaders.urlopen", lambda *_args, **_kwargs: Response())
    downloaded = _download("https://example.invalid/data", destination, validator=lambda path: path.read_bytes() == b"complete")
    assert downloaded.read_bytes() == b"complete"
    assert not Path(f"{destination}.part").exists()
