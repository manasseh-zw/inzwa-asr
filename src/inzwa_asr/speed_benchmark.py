"""Build and run the frozen Shona ASR speed-v1 benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import soundfile

from .evaluation import read_evaluation_manifest
from .release import file_sha256, parse_s3_uri
from .sunbird_evaluation import MODEL_ID as SUNBIRD_MODEL_ID
from .sunbird_evaluation import MODEL_REVISION as SUNBIRD_MODEL_REVISION
from .sunbird_evaluation import SAMPLE_RATE, SHONA_LANGUAGE_TOKEN, _load_audio
from .training import _disable_cuda_graphs, _disable_prediction_logging, _git_commit

BENCHMARK_VERSION = "shona-speed-v1"
SOURCES = ("waxal", "fleurs", "bible", "curated-150")


class Backend(Protocol):
    def transcribe(self, paths: list[str], batch_size: int) -> list[str]: ...


def _duration(path: str) -> float:
    return float(soundfile.info(path).duration)


def _digest(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_to_duration(
    rows: list[dict[str, Any]], target_seconds: float, seed: int
) -> list[dict[str, Any]]:
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    selected: list[dict[str, Any]] = []
    elapsed = 0.0
    for row in candidates:
        selected.append(row)
        elapsed += float(row["duration"])
        if elapsed >= target_seconds:
            break
    return selected


def build_manifest(
    manifest_dir: Path,
    output: Path,
    *,
    target_hours: float = 6.0,
    seed: int = 42,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Speed manifest already exists: {output}")
    by_source: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCES:
        source_rows = read_evaluation_manifest(manifest_dir / f"test-{source}.jsonl")
        by_source[source] = [
            {
                "id": str(row["id"]),
                "audio_filepath": str(row["audio_filepath"]),
                "source": source,
                "duration": _duration(str(row["audio_filepath"])),
            }
            for row in source_rows
        ]

    target_seconds = target_hours * 3600
    equal_share = target_seconds / len(SOURCES)
    curated_total = sum(row["duration"] for row in by_source["curated-150"])
    curated_target = min(curated_total, equal_share)
    selected = _select_to_duration(by_source["curated-150"], curated_target, seed + 3)
    remaining = max(0.0, target_seconds - sum(row["duration"] for row in selected))
    public_target = remaining / 3
    for offset, source in enumerate(SOURCES[:3]):
        selected.extend(
            _select_to_duration(by_source[source], public_target, seed + offset)
        )
    selected.sort(key=lambda row: (row["source"], row["id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    totals = {
        source: {
            "rows": sum(row["source"] == source for row in selected),
            "audio_hours": sum(
                row["duration"] for row in selected if row["source"] == source
            )
            / 3600,
        }
        for source in SOURCES
    }
    metadata = {
        "benchmark": BENCHMARK_VERSION,
        "seed": seed,
        "target_hours": target_hours,
        "selection": (
            "all curated audio up to an equal share; residual duration split "
            "equally across WAXAL, FLEURS, and Bible; no duplicated clips"
        ),
        "rows": len(selected),
        "audio_hours": sum(row["duration"] for row in selected) / 3600,
        "sources": totals,
        "manifest_sha256": file_sha256(output),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _read_speed_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("Speed manifest is empty")
    for row in rows:
        if not Path(row["audio_filepath"]).is_file():
            raise FileNotFoundError(row["audio_filepath"])
    return rows


def _preflight_rows(rows: list[dict[str, Any]], seconds: float) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["duration"], row["id"]))
    stride = max(1, len(ordered) // 256)
    sampled = ordered[::stride]
    selected: list[dict[str, Any]] = []
    elapsed = 0.0
    for row in sampled:
        selected.append(row)
        elapsed += row["duration"]
        if elapsed >= seconds:
            break
    return selected


def _run_pass(
    backend: Backend, rows: list[dict[str, Any]], batch_size: int
) -> dict[str, Any]:
    import torch

    ordered = sorted(rows, key=lambda row: (row["duration"], row["id"]))
    hypotheses: list[str] = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(ordered), batch_size):
        paths = [row["audio_filepath"] for row in ordered[offset : offset + batch_size]]
        hypotheses.extend(backend.transcribe(paths, batch_size))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    audio_seconds = sum(row["duration"] for row in ordered)
    return {
        "wall_seconds": elapsed,
        "audio_seconds": audio_seconds,
        "rtf": elapsed / audio_seconds,
        "audio_hours_per_hour": audio_seconds / elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "hypothesis_sha256": _digest(hypotheses),
    }


class ParakeetBackend:
    def __init__(self, model_path: Path) -> None:
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.restore_from(
            str(model_path), map_location="cpu"
        )
        _disable_cuda_graphs(self.model)
        _disable_prediction_logging(self.model)
        self.model = self.model.cuda().eval()

    def transcribe(self, paths: list[str], batch_size: int) -> list[str]:
        values = self.model.transcribe(paths, batch_size=batch_size, verbose=False)
        return [str(getattr(value, "text", value) or "") for value in values]


class SunbirdBackend:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(
            SUNBIRD_MODEL_ID, revision=SUNBIRD_MODEL_REVISION
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            SUNBIRD_MODEL_ID,
            revision=SUNBIRD_MODEL_REVISION,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        tokenizer = self.processor.tokenizer
        self.forced_decoder_ids = [
            (1, SHONA_LANGUAGE_TOKEN),
            (2, tokenizer.convert_tokens_to_ids("<|transcribe|>")),
            (3, tokenizer.convert_tokens_to_ids("<|notimestamps|>")),
        ]

    def transcribe(self, paths: list[str], batch_size: int) -> list[str]:
        inputs = self.processor(
            [_load_audio(path) for path in paths],
            sampling_rate=SAMPLE_RATE,
            do_normalize=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: (
                value.to(device=self.device, dtype=self.dtype)
                if self.torch.is_floating_point(value)
                else value.to(self.device)
            )
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            predicted = self.model.generate(
                **inputs,
                forced_decoder_ids=self.forced_decoder_ids,
                num_beams=1,
                do_sample=False,
                max_new_tokens=256,
            )
        return self.processor.batch_decode(
            predicted,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def _download_model(uri: str, sha256: str, target: Path) -> Path:
    import boto3

    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        bucket, key = parse_s3_uri(uri)
        boto3.client(
            "s3", region_name=os.environ.get("AWS_REGION", "us-east-1")
        ).download_file(bucket, key, str(target))
    if file_sha256(target) != sha256:
        raise RuntimeError("Parakeet model checksum mismatch")
    return target


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"Benchmark output already exists: {args.output}")
    rows = _read_speed_manifest(args.manifest)
    model_started = time.perf_counter()
    if args.backend == "parakeet":
        model_path = _download_model(
            args.model_uri, args.model_sha256, args.work_root / "models" / "inzwa.nemo"
        )
        backend: Backend = ParakeetBackend(model_path)
        model_revision = args.model_sha256
    else:
        backend = SunbirdBackend()
        model_revision = SUNBIRD_MODEL_REVISION
    model_load_seconds = time.perf_counter() - model_started

    warmup = sorted(rows, key=lambda row: row["duration"])[: args.warmup_rows]
    backend.transcribe([row["audio_filepath"] for row in warmup], len(warmup))
    preflight = _preflight_rows(rows, args.preflight_minutes * 60)
    candidates = []
    reference_digest = None
    for batch_size in args.batch_sizes:
        try:
            result = _run_pass(backend, preflight, batch_size)
            if reference_digest is None:
                reference_digest = result["hypothesis_sha256"]
            result["outputs_match_reference"] = (
                result["hypothesis_sha256"] == reference_digest
            )
            candidates.append({"batch_size": batch_size, "status": "ok", **result})
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            candidates.append(
                {"batch_size": batch_size, "status": "oom", "error": str(error)}
            )
            import torch

            torch.cuda.empty_cache()
    eligible = [
        item
        for item in candidates
        if item["status"] == "ok" and item["outputs_match_reference"]
    ]
    if not eligible:
        raise RuntimeError("No batch size passed the speed preflight")
    selected_batch_size = min(eligible, key=lambda item: item["rtf"])["batch_size"]
    repetitions = [
        _run_pass(backend, rows, selected_batch_size) for _ in range(args.repetitions)
    ]
    digests = {item["hypothesis_sha256"] for item in repetitions}
    if len(digests) != 1:
        raise RuntimeError("Full-run hypotheses changed between repetitions")
    summary = {
        "status": "complete",
        "benchmark": BENCHMARK_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "backend": args.backend,
        "model_revision": model_revision,
        "gpu": __import__("torch").cuda.get_device_name(0),
        "manifest_sha256": file_sha256(args.manifest),
        "audio_hours": sum(row["duration"] for row in rows) / 3600,
        "ordering": "ascending duration for padding-efficient offline throughput",
        "timing_scope": (
            "end-to-end cached-local-audio transcription; model load separate"
        ),
        "model_load_seconds": model_load_seconds,
        "preflight": candidates,
        "selected_batch_size": selected_batch_size,
        "repetitions": repetitions,
        "median_rtf": statistics.median(item["rtf"] for item in repetitions),
        "median_audio_hours_per_hour": statistics.median(
            item["audio_hours_per_hour"] for item in repetitions
        ),
        "median_wall_seconds": statistics.median(
            item["wall_seconds"] for item in repetitions
        ),
        "peak_vram_gib": max(item["peak_vram_gib"] for item in repetitions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_prefix:
        bucket, prefix = parse_s3_uri(args.output_prefix)
        import boto3

        boto3.client(
            "s3", region_name=os.environ.get("AWS_REGION", "us-east-1")
        ).upload_file(str(args.output), bucket, f"{prefix}/{args.output.name}")
    del backend
    gc.collect()
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--manifest-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--target-hours", type=float, default=6.0)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--output-prefix")
    run = subparsers.add_parser("run")
    run.add_argument("--backend", choices=("parakeet", "sunbird"), required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--work-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--output-prefix")
    run.add_argument("--model-uri")
    run.add_argument("--model-sha256")
    run.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    run.add_argument("--preflight-minutes", type=float, default=20.0)
    run.add_argument("--warmup-rows", type=int, default=8)
    run.add_argument("--repetitions", type=int, default=3)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        value = build_manifest(
            args.manifest_dir,
            args.output,
            target_hours=args.target_hours,
            seed=args.seed,
        )
        if args.output_prefix:
            import boto3

            bucket, prefix = parse_s3_uri(args.output_prefix)
            client = boto3.client(
                "s3", region_name=os.environ.get("AWS_REGION", "us-east-1")
            )
            for path in (args.output, args.output.with_suffix(".metadata.json")):
                client.upload_file(str(path), bucket, f"{prefix}/{path.name}")
    else:
        if args.backend == "parakeet" and not (args.model_uri and args.model_sha256):
            raise ValueError("Parakeet requires --model-uri and --model-sha256")
        value = run_benchmark(args)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
