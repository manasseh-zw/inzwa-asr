"""Mirror the frozen curated-150 Hugging Face dataset into S3."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .release import file_sha256, parse_s3_uri

REPO_ID = "manassehzw/sna-manasseh-150-raw"
REVISION = "4c72a535218164cab117126e5bc9c4a4799d3101"
FILES = ("README.md", "data/train-00000-of-00001.parquet")
EXPECTED_ROWS = 150


def run(destination: str, work_root: Path) -> dict[str, object]:
    import boto3
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    bucket, prefix = parse_s3_uri(destination)
    client = boto3.client("s3")
    work_root.mkdir(parents=True, exist_ok=True)
    objects = []
    for filename in FILES:
        cached = Path(
            hf_hub_download(
                REPO_ID,
                filename,
                repo_type="dataset",
                revision=REVISION,
                local_dir=work_root,
            )
        )
        relative = cached.relative_to(work_root).as_posix()
        key = f"{prefix}/{relative}"
        digest = file_sha256(cached)
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except client.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey"}:
                raise
            head = None
        if head is None or int(head["ContentLength"]) != cached.stat().st_size:
            client.upload_file(str(cached), bucket, key)
        objects.append(
            {
                "path": relative,
                "bytes": cached.stat().st_size,
                "sha256": digest,
            }
        )
    parquet = work_root / "data/train-00000-of-00001.parquet"
    rows = pq.ParquetFile(parquet).metadata.num_rows
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {rows}")
    ready = {
        "status": "ready",
        "dataset": REPO_ID,
        "revision": REVISION,
        "source_split": "train",
        "evaluation_role": "private-curated-test",
        "rows": rows,
        "destination": destination.rstrip("/"),
        "objects": objects,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/READY.json",
        Body=(json.dumps(ready, indent=2) + "\n").encode(),
        ContentType="application/json",
    )
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
