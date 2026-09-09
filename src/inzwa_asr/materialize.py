"""Convert verified tar manifests into ordinary NeMo manifests."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

from .constants import VALIDATION_MANIFESTS
from .manifests import read_manifest
from .release import ReleaseError, verify_release


def _extract_rows(root: Path, rows: list[dict[str, Any]], audio_root: Path) -> None:
    by_tar: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tar.setdefault(str(row["tar_file"]), []).append(row)
    for relative_tar, tar_rows in sorted(by_tar.items()):
        expected = {str(row["audio_filepath"]): row for row in tar_rows}
        with tarfile.open(root / relative_tar, "r") as archive:
            for member_name, row in expected.items():
                member = archive.getmember(member_name)
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ReleaseError(
                        f"Unsafe tar member: {relative_tar}:{member.name}"
                    )
                destination = audio_root / member.name
                if not destination.is_file():
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise ReleaseError(
                            f"Cannot extract {relative_tar}:{member.name}"
                        )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload.read())
                row["audio_filepath"] = str(destination.resolve())


def _write_nemo_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_nemo_manifests(
    root: Path,
    destination: Path,
    *,
    release_verified: bool = False,
    train_source: str | None = None,
) -> dict[str, Path]:
    if not release_verified:
        verify_release(root, require_complete=True)
    destination = destination.resolve()
    audio_root = destination / "audio"
    outputs: dict[str, Path] = {}
    manifests = {"train": "manifests/train.jsonl.gz", **VALIDATION_MANIFESTS}
    for name, relative in manifests.items():
        rows = read_manifest(root / relative)
        if name == "train" and train_source:
            rows = [row for row in rows if str(row["source"]) == train_source]
            if not rows:
                raise ReleaseError(
                    f"No training rows found for source {train_source!r}"
                )
        _extract_rows(root, rows, audio_root)
        output = destination / f"{name}.jsonl"
        _write_nemo_manifest(output, rows)
        outputs[name] = output
    return outputs
