"""Hydrate and verify the immutable full-corpus release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .constants import RELEASE_FORMAT, RELEASE_URI


class ReleaseError(RuntimeError):
    """The release is absent, malformed, incomplete, or corrupt."""


@dataclass(frozen=True)
class ReleaseFile:
    path: str
    size: int
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an S3 URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ReleaseError(f"Unsafe release path: {value!r}")
    return path.as_posix()


def load_ready(root: Path) -> dict[str, Any]:
    path = root / "READY.json"
    if not path.is_file():
        raise ReleaseError(f"Missing release gate: {path}")
    try:
        ready = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ReleaseError(f"Cannot read {path}: {exc}") from exc
    if ready.get("status") != "ready":
        raise ReleaseError("READY.json status must be 'ready'")
    if ready.get("format_version") != RELEASE_FORMAT:
        raise ReleaseError(
            f"Unsupported format {ready.get('format_version')!r}; "
            f"expected {RELEASE_FORMAT!r}"
        )
    if ready.get("mode") != "full":
        raise ReleaseError("Training requires the full release")
    files = ready.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseError("READY.json has no file inventory")
    seen: set[str] = set()
    for item in files:
        path_value = _safe_relative_path(str(item.get("path", "")))
        if path_value in seen:
            raise ReleaseError(f"Duplicate file in READY.json: {path_value}")
        seen.add(path_value)
        size = item.get("bytes")
        checksum = str(item.get("sha256", ""))
        if not isinstance(size, int) or size < 0 or len(checksum) != 64:
            raise ReleaseError(f"Invalid file record for {path_value}")
    return ready


def release_files(ready: dict[str, Any]) -> list[ReleaseFile]:
    return [
        ReleaseFile(str(item["path"]), int(item["bytes"]), str(item["sha256"]))
        for item in ready["files"]
    ]


def manifest_files(files: Iterable[ReleaseFile]) -> list[ReleaseFile]:
    return [
        item
        for item in files
        if item.path.startswith("manifests/") or item.path.startswith("metadata/")
    ]


def verify_file(root: Path, item: ReleaseFile) -> None:
    path = root / item.path
    if not path.is_file():
        raise ReleaseError(f"Missing release file: {item.path}")
    actual_size = path.stat().st_size
    if actual_size != item.size:
        raise ReleaseError(
            f"Size mismatch for {item.path}: expected {item.size}, got {actual_size}"
        )
    actual_hash = file_sha256(path)
    if actual_hash != item.sha256:
        raise ReleaseError(
            f"Checksum mismatch for {item.path}: expected {item.sha256}, "
            f"got {actual_hash}"
        )


def verify_release(root: Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Verify the local cache, requiring every READY file for training."""

    root = root.resolve()
    ready = load_ready(root)
    files = release_files(ready)
    selected = files if require_complete else manifest_files(files)
    for item in selected:
        verify_file(root, item)
    if require_complete:
        expected = {item.path for item in files}
        verified = expected
    else:
        expected = {item.path for item in files}
        verified = {item.path for item in selected}
    return {
        "status": "verified",
        "train_ready": require_complete and verified == expected,
        "format_version": ready["format_version"],
        "verified_files": len(verified),
        "release_files": len(expected),
        "release_destination": ready.get("destination"),
        "audit": ready.get("audit", {}),
    }


def _download(client: Any, bucket: str, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        client.download_file(bucket, key, str(partial))
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def hydrate_files(
    root: Path,
    paths: Iterable[str],
    *,
    uri: str = RELEASE_URI,
    client: Any | None = None,
) -> None:
    """Hydrate selected READY-listed files and verify each one."""

    if client is None:
        import boto3

        client = boto3.client(
            "s3", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
    bucket, prefix = parse_s3_uri(uri)
    ready = load_ready(root)
    inventory = {item.path: item for item in release_files(ready)}
    for requested in sorted(set(paths)):
        safe_path = _safe_relative_path(requested)
        if safe_path not in inventory:
            raise ReleaseError(f"File is not listed by READY.json: {safe_path}")
        item = inventory[safe_path]
        destination = root / safe_path
        if destination.is_file():
            verify_file(root, item)
            continue
        _download(client, bucket, f"{prefix}/{safe_path}", destination)
        verify_file(root, item)


def hydrate_release(
    destination: Path,
    *,
    uri: str = RELEASE_URI,
    manifest_only: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Download a release atomically, then verify the hydrated scope."""

    if client is None:
        import boto3

        client = boto3.client(
            "s3", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
    bucket, prefix = parse_s3_uri(uri)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _download(client, bucket, f"{prefix}/READY.json", destination / "READY.json")
    ready = load_ready(destination)
    files = release_files(ready)
    selected = manifest_files(files) if manifest_only else files
    for item in selected:
        path = destination / item.path
        if path.is_file() and path.stat().st_size == item.size:
            try:
                verify_file(destination, item)
                continue
            except ReleaseError:
                pass
        _download(client, bucket, f"{prefix}/{item.path}", path)
        verify_file(destination, item)
    result = verify_release(destination, require_complete=not manifest_only)
    result.update({"mode": "manifest-only" if manifest_only else "full", "uri": uri})
    hydration_path = destination / "HYDRATION.json"
    hydration_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    hydrate = subparsers.add_parser("hydrate")
    hydrate.add_argument("destination", type=Path)
    hydrate.add_argument("--uri", default=RELEASE_URI)
    hydrate.add_argument("--manifest-only", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("destination", type=Path)
    verify.add_argument("--manifest-only", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "hydrate":
        result = hydrate_release(
            args.destination, uri=args.uri, manifest_only=args.manifest_only
        )
    else:
        result = verify_release(
            args.destination, require_complete=not args.manifest_only
        )
    print(json.dumps(result, indent=2))
