from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from inzwa_asr.release import ReleaseError, hydrate_release, verify_release


class FakeS3:
    def __init__(self, source: Path) -> None:
        self.source = source

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        relative = key.split("prefix/", 1)[1]
        shutil.copyfile(self.source / relative, destination)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_release(root: Path) -> None:
    (root / "manifests").mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "audio").mkdir()
    with gzip.open(root / "manifests/train.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write('{"example": true}\n')
    (root / "metadata/summary.json").write_text('{"rows": 1}\n')
    with tarfile.open(root / "audio/audio.tar", "w") as archive:
        payload = root / "sample.flac"
        payload.write_bytes(b"audio")
        archive.add(payload, arcname="sample.flac")
        payload.unlink()
    files = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    ready = {
        "status": "ready",
        "format_version": "shona-parakeet-full-corpus-v2",
        "mode": "full",
        "destination": "s3://bucket/prefix",
        "files": files,
    }
    (root / "READY.json").write_text(json.dumps(ready))


def test_manifest_only_hydration_is_not_train_ready(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "cache"
    make_release(source)
    result = hydrate_release(
        destination,
        uri="s3://bucket/prefix",
        manifest_only=True,
        client=FakeS3(source),
    )
    assert result["train_ready"] is False
    assert (destination / "manifests/train.jsonl.gz").is_file()
    assert not (destination / "audio/audio.tar").exists()
    with pytest.raises(ReleaseError, match="Missing release file"):
        verify_release(destination, require_complete=True)


def test_full_verification_rejects_checksum_mismatch(tmp_path: Path) -> None:
    make_release(tmp_path)
    assert verify_release(tmp_path, require_complete=True)["train_ready"] is True
    (tmp_path / "metadata/summary.json").write_text("corrupt")
    with pytest.raises(ReleaseError, match="mismatch"):
        verify_release(tmp_path, require_complete=True)


def test_ready_rejects_parent_path(tmp_path: Path) -> None:
    make_release(tmp_path)
    ready = json.loads((tmp_path / "READY.json").read_text())
    ready["files"][0]["path"] = "../secret"
    (tmp_path / "READY.json").write_text(json.dumps(ready))
    with pytest.raises(ReleaseError, match="Unsafe"):
        verify_release(tmp_path)
