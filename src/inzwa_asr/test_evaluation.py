"""Evaluate a frozen Inzwa model on the public Shona test suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile

from .audio import decode_audio_bytes
from .evaluation import evaluate_manifest
from .release import file_sha256, parse_s3_uri
from .training import _disable_cuda_graphs, _disable_prediction_logging, _git_commit

MODEL_SHA256 = "08bb22eb1050f165529c25d9306c45679b10e54a9b6c568a113e33622b3a8da1"
DATASETS = {
    "waxal": {
        "revision": "f91b1a79cbac15520d3c808b56e5192bf903f280",
        "expected_rows": 1565,
        "s3_prefix": "datasets/waxal/annotated/v1/data/",
        "pattern": re.compile(r"test-.*\.parquet$"),
    },
    "fleurs": {
        "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        "expected_rows": 925,
        "s3_prefix": "datasets/evaluations/fleurs/sn_zw/v1/parquet-data/sn_zw/",
        "pattern": re.compile(r"test-.*\.parquet$"),
    },
    "bible": {
        "revision": "36f7967da6e2a92037a73287082fccf5481fe495",
        "expected_rows": 1593,
        "s3_prefix": "datasets/evaluations/bible/book-disjoint/v1/data/",
        "pattern": re.compile(r"test-.*\.parquet$"),
    },
    "curated-150": {
        "revision": "4c72a535218164cab117126e5bc9c4a4799d3101",
        "expected_rows": 150,
        "s3_prefix": "datasets/evaluations/curated-150/v1/data/",
        "pattern": re.compile(r"train-.*\.parquet$"),
    },
}


def _audio_bytes(value: Any) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, list):
        value = bytes((int(item) + 256) % 256 for item in value)
    if not isinstance(value, bytes):
        raise RuntimeError(f"Unsupported embedded audio value: {type(value)!r}")
    return value


def _column(names: set[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise RuntimeError(f"None of {candidates} found in {sorted(names)}")


def _download_sources(root: Path, bucket: str) -> dict[str, list[Path]]:
    import boto3

    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    result: dict[str, list[Path]] = {}
    for name, spec in DATASETS.items():
        target = root / "parquet" / name
        target.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        if "s3_prefix" in spec:
            paginator = client.get_paginator("list_objects_v2")
            keys = [
                item["Key"]
                for page in paginator.paginate(Bucket=bucket, Prefix=spec["s3_prefix"])
                for item in page.get("Contents", [])
                if spec["pattern"].search(item["Key"])
            ]
            for key in sorted(keys):
                path = target / Path(key).name
                if not path.is_file():
                    client.download_file(bucket, key, str(path))
                paths.append(path)
        else:
            path = target / "test.parquet"
            if not path.is_file():
                urllib.request.urlretrieve(str(spec["url"]), path)
            paths.append(path)
        if not paths:
            raise RuntimeError(f"No test shards found for {name}")
        result[name] = paths
    return result


def materialize_dataset(name: str, shards: list[Path], root: Path) -> Path:
    import pyarrow.parquet as pq

    audio_root = root / "audio" / name
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifests" / f"test-{name}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        names = set(parquet.schema_arrow.names)
        audio_col = _column(names, ("audio", "audio_bytes", "bytes"))
        text_col = _column(names, ("transcription", "sentence", "text", "reference"))
        id_col = next(
            (item for item in ("id", "source_id", "path") if item in names), None
        )
        columns = [audio_col, text_col] + ([id_col] if id_col else [])
        offset = 0
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            for local_index, record in enumerate(batch.to_pylist()):
                index = offset + local_index
                row_id = str(record.get(id_col) if id_col else f"{shard.stem}:{index}")
                safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", row_id)[:180]
                path = audio_root / f"{shard.stem}-{index:06d}-{safe_id}.flac"
                payload = _audio_bytes(record[audio_col])
                waveform, source_rate, target_rate = decode_audio_bytes(payload)
                if not path.is_file():
                    soundfile.write(path, waveform, target_rate, format="FLAC")
                rows.append(
                    {
                        "id": f"{name}:{shard.stem}:{index}:{row_id}",
                        "audio_filepath": str(path),
                        "text": str(record[text_col]).strip(),
                        "source_sample_rate": source_rate,
                        "sample_rate": target_rate,
                    }
                )
            offset += batch.num_rows
    expected = int(DATASETS[name]["expected_rows"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} {name} rows, found {len(rows)}")
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


def _upload(root: Path, uri: str) -> list[str]:
    import boto3

    bucket, prefix = parse_s3_uri(uri)
    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    uploaded = []
    for path in sorted(root.iterdir()):
        if path.is_file():
            key = f"{prefix}/{path.name}"
            client.upload_file(str(path), bucket, key)
            uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def run(args: argparse.Namespace) -> dict[str, Any]:
    import boto3

    root = args.work_root.resolve()
    output = root / "results"
    output.mkdir(parents=True, exist_ok=True)
    bucket, model_key = parse_s3_uri(args.model_uri)
    model_path = root / "model" / "inzwa.nemo"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        boto3.client(
            "s3", region_name=os.environ.get("AWS_REGION", "us-east-1")
        ).download_file(bucket, model_key, str(model_path))
    if file_sha256(model_path) != args.model_sha256:
        raise RuntimeError("Model checksum mismatch")
    aws_identity = boto3.client("sts").get_caller_identity()
    sources = _download_sources(root, bucket)
    manifests = {
        name: materialize_dataset(name, shards, root)
        for name, shards in sources.items()
    }
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location="cpu")
    _disable_cuda_graphs(model)
    _disable_prediction_logging(model)
    model = model.cuda().eval()
    results = {
        name: evaluate_manifest(
            model,
            manifest,
            name=f"test-{name}",
            batch_size=args.batch_size,
            output_dir=output,
        )
        for name, manifest in manifests.items()
    }
    summary = {
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "model_uri": args.model_uri,
        "model_sha256": args.model_sha256,
        "aws_account": aws_identity["Account"],
        "normalization": (
            "Unicode NFKC + casefold + punctuation-to-space + whitespace collapse"
        ),
        "datasets": {
            name: {"revision": spec["revision"], **results[name]}
            for name, spec in DATASETS.items()
        },
        "comparability_note": (
            "Test sets were not used for training, checkpoint selection, or tuning."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary["uploaded"] = _upload(output, args.output_prefix)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--model-sha256", default=MODEL_SHA256)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    print(json.dumps(run(parser.parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
