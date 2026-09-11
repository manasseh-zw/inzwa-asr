from pathlib import Path

import pytest

from inzwa_asr.evaluation import read_evaluation_manifest
from inzwa_asr.export_model import export_model
from inzwa_asr.test_evaluation import _column
from inzwa_asr.training import TrainConfig, _select_longest_rows


def test_column_selects_first_supported_name() -> None:
    assert (
        _column({"sentence", "audio"}, ("transcription", "sentence", "text"))
        == "sentence"
    )


def test_dataset_revisions_are_pinned() -> None:
    from inzwa_asr.test_evaluation import DATASETS

    assert all(len(spec["revision"]) == 40 for spec in DATASETS.values())
    assert Path("test.parquet").suffix == ".parquet"


def test_evaluation_manifest_does_not_require_training_fields(tmp_path: Path) -> None:
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"placeholder")
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(
        '{"id":"one","audio_filepath":"' + str(audio) + '","text":"mhoro"}\n',
        encoding="utf-8",
    )
    assert read_evaluation_manifest(manifest)[0]["text"] == "mhoro"


def test_adaptation_configuration_is_explicit(tmp_path: Path) -> None:
    config = TrainConfig(
        release_root=tmp_path,
        output_dir=tmp_path / "output",
        output_prefix="s3://bucket/run",
        run_id="run",
        initial_model_uri="s3://bucket/model.nemo",
        initial_model_sha256="a" * 64,
        train_source="fleurs",
        selection_objective="fleurs",
        max_steps=10,
    )
    assert config.train_source == "fleurs"
    assert config.selection_objective == "fleurs"


def test_deferred_export_requires_training_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.json"):
        export_model(tmp_path, "s3://bucket/run")


def test_longest_example_smoke_selection() -> None:
    rows = [
        {"id": "short", "duration": 4.0},
        {"id": "longest", "duration": 53.4},
        {"id": "long", "duration": 38.82},
    ]
    selected = _select_longest_rows(rows, 2)
    assert [row["id"] for row in selected] == ["longest", "long"]
