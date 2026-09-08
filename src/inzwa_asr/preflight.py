"""Run a source-stratified audio loader preflight."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

from .audio import decode_audio_bytes
from .constants import PSEUDO_TIERS, RELEASE_URI, TARGET_SAMPLE_RATE, TRAIN_SOURCES
from .manifests import read_manifest, validate_training_rows
from .release import hydrate_files, verify_release


def _stratum(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["source"]), str(row.get("tier") or "none")


def representative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = {(source, "none") for source in TRAIN_SOURCES - {"waxal-pseudo"}}
    targets.update({("waxal-pseudo", tier) for tier in PSEUDO_TIERS})
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _stratum(row)
        if key in targets and key not in selected:
            selected[key] = row
    missing = targets - selected.keys()
    if missing:
        raise RuntimeError(f"Manifest does not cover strata: {sorted(missing)}")
    return [selected[key] for key in sorted(selected)]


def _read_member(tar_path: Path, member_name: str) -> bytes:
    with tarfile.open(tar_path, "r") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise RuntimeError(f"Missing {member_name} in {tar_path}") from exc
        if not member.isfile():
            raise RuntimeError(f"Expected a regular file: {tar_path}:{member_name}")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Cannot read {tar_path}:{member_name}")
        return extracted.read()


def run_preflight(
    root: Path, *, uri: str = RELEASE_URI, hydrate_audio: bool = True
) -> dict[str, Any]:
    root = root.resolve()
    verification = verify_release(root, require_complete=False)
    rows = read_manifest(root / "manifests/train.jsonl.gz")
    manifest_summary = validate_training_rows(rows)
    selected = representative_rows(rows)
    tar_paths = {str(row["tar_file"]) for row in selected}
    if hydrate_audio:
        hydrate_files(root, tar_paths, uri=uri)
    checks = []
    for row in selected:
        payload = _read_member(root / row["tar_file"], str(row["audio_filepath"]))
        waveform, decoded_rate, output_rate = decode_audio_bytes(payload)
        if output_rate != TARGET_SAMPLE_RATE or not len(waveform):
            raise RuntimeError(f"Audio decode failed for {row['id']}")
        if decoded_rate != int(row["sample_rate"]):
            raise RuntimeError(
                f"Recorded sample rate mismatch for {row['id']}: "
                f"manifest={row['sample_rate']} decoded={decoded_rate}"
            )
        checks.append(
            {
                "id": row["id"],
                "source": row["source"],
                "tier": row.get("tier"),
                "duration_bin": row["duration_bin"],
                "transcript_type": row["transcript_type"],
                "decoded_sample_rate": decoded_rate,
                "output_sample_rate": output_rate,
                "samples": len(waveform),
            }
        )
    return {
        "status": "passed",
        "release": verification,
        "manifest": manifest_summary,
        "audio_shards": sorted(tar_paths),
        "checks": checks,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("release_root", type=Path)
    result.add_argument("--uri", default=RELEASE_URI)
    result.add_argument("--no-hydrate-audio", action="store_true")
    result.add_argument("--report", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    report = run_preflight(
        args.release_root,
        uri=args.uri,
        hydrate_audio=not args.no_hydrate_audio,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
