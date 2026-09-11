"""Evaluate pinned Sunbird Whisper on the frozen Inzwa test suite."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from .evaluation import read_evaluation_manifest, score
from .test_evaluation import DATASETS, _download_sources, _upload, materialize_dataset
from .training import _git_commit

MODEL_ID = "Sunbird/asr-whisper-51-african-languages"
MODEL_REVISION = "213531767f739bc5b1e5dcb3e2ec9e112674ab67"
SAMPLE_RATE = 16_000
SHONA_LANGUAGE_TOKEN = 50324


def _load_audio(path: str) -> np.ndarray:
    waveform, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"Expected {SAMPLE_RATE} Hz audio at {path}, got {sample_rate}"
        )
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import boto3
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    root = args.work_root.resolve()
    output = root / "sunbird-results"
    output.mkdir(parents=True, exist_ok=False)
    bucket = boto3.client("sts").get_caller_identity()["Account"]
    if bucket != "102431378819":
        raise RuntimeError(f"Unexpected AWS account: {bucket}")
    data_bucket = "teleagents-research-102431378819-us-east-1"
    sources = _download_sources(root, data_bucket)
    manifests = {
        name: materialize_dataset(name, shards, root)
        for name, shards in sources.items()
    }

    device = torch.device("cuda")
    dtype = torch.float16
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    transcribe_token = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    no_timestamps_token = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    forced_decoder_ids = [
        (1, SHONA_LANGUAGE_TOKEN),
        (2, transcribe_token),
        (3, no_timestamps_token),
    ]

    results: dict[str, Any] = {}
    for name, manifest in manifests.items():
        rows = read_evaluation_manifest(manifest)
        hypotheses: list[str] = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            inputs = processor(
                [_load_audio(str(row["audio_filepath"])) for row in batch],
                sampling_rate=SAMPLE_RATE,
                do_normalize=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: (
                    value.to(device=device, dtype=dtype)
                    if torch.is_floating_point(value)
                    else value.to(device)
                )
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                predicted = model.generate(
                    **inputs,
                    forced_decoder_ids=forced_decoder_ids,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=256,
                )
            hypotheses.extend(
                processor.batch_decode(
                    predicted,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        references = [str(row["text"]) for row in rows]
        metrics = score(references, hypotheses)
        prediction_path = output / f"predictions-{name}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row, hypothesis in zip(rows, hypotheses, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "reference": row["text"],
                            "hypothesis": hypothesis,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        results[name] = {
            "name": f"test-{name}",
            "rows": len(rows),
            "metrics": metrics,
        }
        print(json.dumps({"dataset": name, **results[name]}), flush=True)

    summary = {
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers-whisper",
        "decoding": {
            "language": "Shona",
            "language_token": SHONA_LANGUAGE_TOKEN,
            "task": "transcribe",
            "num_beams": 1,
        },
        "normalization": (
            "Unicode NFKC + casefold + punctuation-to-space + whitespace collapse"
        ),
        "datasets": {
            name: {"revision": DATASETS[name]["revision"], **results[name]}
            for name in DATASETS
        },
        "comparability_note": "Same frozen test inputs and scoring as Inzwa.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary["uploaded"] = _upload(output, args.output_prefix)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    print(json.dumps(evaluate(parser.parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
