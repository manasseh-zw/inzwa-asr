import json
from pathlib import Path

import numpy as np
import soundfile

from inzwa_asr.speed_benchmark import SOURCES, build_manifest


def test_build_manifest_is_deterministic_and_has_no_duplicates(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for source in SOURCES:
        manifest = manifest_dir / f"test-{source}.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for index in range(3):
                audio = tmp_path / f"{source}-{index}.wav"
                soundfile.write(audio, np.zeros(16_000, dtype=np.float32), 16_000)
                handle.write(
                    json.dumps(
                        {
                            "id": f"{source}-{index}",
                            "audio_filepath": str(audio),
                            "text": "test",
                        }
                    )
                    + "\n"
                )

    first = tmp_path / "speed-a.jsonl"
    second = tmp_path / "speed-b.jsonl"
    first_metadata = build_manifest(manifest_dir, first, target_hours=4 / 3600, seed=42)
    second_metadata = build_manifest(
        manifest_dir, second, target_hours=4 / 3600, seed=42
    )
    first_rows = [json.loads(line) for line in first.read_text().splitlines()]
    second_rows = [json.loads(line) for line in second.read_text().splitlines()]

    assert first_rows == second_rows
    assert len({row["id"] for row in first_rows}) == len(first_rows)
    assert {row["source"] for row in first_rows} == set(SOURCES)
    assert first_metadata["audio_hours"] == second_metadata["audio_hours"]
