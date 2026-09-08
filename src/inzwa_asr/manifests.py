"""Read and validate full-corpus manifests."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .constants import PSEUDO_TIERS, TRAIN_SOURCES
from .release import ReleaseError

REQUIRED_FIELDS = {
    "id",
    "source",
    "split",
    "tier",
    "text",
    "transcript_type",
    "duration",
    "duration_bin",
    "original_sample_rate",
    "sample_rate",
    "audio_filepath",
    "tar_file",
}


def iter_manifest(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseError(f"Invalid JSON at {path}:{line_number}") from exc
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ReleaseError(f"Missing {sorted(missing)} at {path}:{line_number}")
            if not row["text"] or float(row["duration"]) <= 0:
                raise ReleaseError(f"Invalid text or duration at {path}:{line_number}")
            yield row


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return list(iter_manifest(path))


def validate_training_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sources = Counter(str(row["source"]) for row in rows)
    tiers = Counter(str(row["tier"]) for row in rows if row.get("tier"))
    if set(sources) != TRAIN_SOURCES:
        raise ReleaseError(
            f"Training sources are {sorted(sources)}, expected {sorted(TRAIN_SOURCES)}"
        )
    if set(tiers) != PSEUDO_TIERS:
        raise ReleaseError(
            f"Pseudo tiers are {sorted(tiers)}, expected {sorted(PSEUDO_TIERS)}"
        )
    if any(row.get("tier") for row in rows if row["source"] != "waxal-pseudo"):
        raise ReleaseError("Only WAXAL pseudo-label rows may carry a pseudo tier")
    return {"rows": len(rows), "sources": dict(sources), "pseudo_tiers": dict(tiers)}
