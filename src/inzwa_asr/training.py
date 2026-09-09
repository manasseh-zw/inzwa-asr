"""Fine-tune Parakeet TDT on the verified v2 Shona release."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import BASE_MODEL, RELEASE_URI
from .evaluation import evaluate_manifest
from .manifests import read_manifest, validate_training_rows
from .materialize import materialize_nemo_manifests
from .release import file_sha256, verify_release
from .sampler import DurationBatchSampler

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}\Z")


@dataclass(frozen=True)
class TrainConfig:
    release_root: Path
    output_dir: Path
    output_prefix: str
    run_id: str
    model_id: str = BASE_MODEL
    model_revision: str = "main"
    seed: int = 42
    max_epochs: int = 10
    batch_size: int = 2
    max_batch_duration: float = 60.0
    sort_window: int = 256
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    num_workers: int = 2
    validation_batch_size: int = 16
    validation_interval_steps: int = 2000
    bible_validation_interval_steps: int = 10_000
    early_stopping_patience_epochs: int = 2
    early_stopping_min_delta: float = 0.001
    exposure_interval: int = 1000
    limit_train_batches: int | None = None
    validation_limit: int | None = None
    reuse_prepared_output: bool = False


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _disable_prediction_logging(model: Any) -> None:
    for metric_name in ("wer", "validation_wer", "test_wer"):
        metric = getattr(model, metric_name, None)
        if metric is not None and hasattr(metric, "log_prediction"):
            metric.log_prediction = False
    for config_name in ("validation_ds", "test_ds"):
        config = getattr(model.cfg, config_name, None)
        if config is not None and "log_prediction" in config:
            config["log_prediction"] = False


def _disable_cuda_graphs(model: Any) -> None:
    decoding = getattr(model, "decoding", None)
    if decoding is None:
        return
    strategy = getattr(decoding, "decoding", decoding)
    if hasattr(strategy, "use_cuda_graph_decoder"):
        strategy.use_cuda_graph_decoder = False


def _resolve_model(model_id: str, revision: str, cache: Path) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download

    resolved = HfApi().model_info(model_id, revision=revision).sha
    snapshot = Path(
        snapshot_download(repo_id=model_id, revision=resolved, cache_dir=cache)
    )
    candidates = sorted(snapshot.glob("*.nemo"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one .nemo file in {snapshot}, found {candidates}")
    return candidates[0], resolved


def _upload_artifacts(output_dir: Path, s3_uri: str) -> list[str]:
    import boto3

    from .release import parse_s3_uri

    bucket, prefix = parse_s3_uri(s3_uri)
    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    excluded_roots = {"materialized", "model-cache"}
    uploaded = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative.split("/", 1)[0] in excluded_roots:
            continue
        key = f"{prefix}/{relative}"
        client.upload_file(str(path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def train(config: TrainConfig) -> dict[str, Any]:
    if not RUN_ID.fullmatch(config.run_id):
        raise ValueError("run_id contains unsupported characters")
    if not config.output_prefix.rstrip("/").endswith(config.run_id):
        raise ValueError("output_prefix must be versioned and end with run_id")
    if config.batch_size < 1 or config.max_batch_duration <= 0:
        raise ValueError("batch_size and max_batch_duration must be positive")
    if (
        min(
            config.validation_interval_steps,
            config.bible_validation_interval_steps,
            config.early_stopping_patience_epochs,
            config.exposure_interval,
        )
        < 1
    ):
        raise ValueError(
            "validation, patience, and telemetry intervals must be positive"
        )
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must not be negative")
    release = verify_release(config.release_root, require_complete=True)
    if not release["train_ready"]:
        raise RuntimeError("The full release did not pass verification")
    if config.output_dir.exists() and not config.reuse_prepared_output:
        raise RuntimeError(f"Output directory already exists: {config.output_dir}")
    if config.output_dir.exists():
        completed_markers = [
            config.output_dir / "best-weights.pt",
            config.output_dir / "config.json",
            config.output_dir / "summary.json",
            config.output_dir / "inzwa-parakeet-tdt-0.6b-v3.nemo",
        ]
        if any(path.exists() for path in completed_markers):
            raise RuntimeError(
                "Refusing to reuse an output directory that contains run artifacts"
            )
    config.output_dir.mkdir(parents=True, exist_ok=config.reuse_prepared_output)

    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    import torch
    from lightning.pytorch.loggers import CometLogger
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    pl.seed_everything(config.seed, workers=True)
    manifests = materialize_nemo_manifests(
        config.release_root,
        config.output_dir / "materialized",
        release_verified=True,
    )
    train_rows = read_manifest(manifests["train"])
    validate_training_rows(train_rows)
    sampler = DurationBatchSampler(
        train_rows,
        max_batch_duration=config.max_batch_duration,
        max_batch_size=config.batch_size,
        seed=config.seed,
        sort_window=config.sort_window,
    )
    model_path, resolved_revision = _resolve_model(
        config.model_id, config.model_revision, config.output_dir / "model-cache"
    )
    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location="cpu")
    _disable_cuda_graphs(model)
    _disable_prediction_logging(model)
    model.cfg.train_ds = OmegaConf.create(
        {
            "manifest_filepath": str(manifests["train"]),
            "sample_rate": 16_000,
            "batch_size": config.batch_size,
            "shuffle": False,
            "num_workers": config.num_workers,
            "pin_memory": True,
            "use_lhotse": False,
            "min_duration": 0.1,
            "max_duration": max(float(row["duration"]) for row in train_rows) + 0.1,
        }
    )
    model.setup_training_data(model.cfg.train_ds)
    dataset = model._train_dl.dataset
    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        persistent_workers=config.num_workers > 0,
    )
    model.cfg.optim = OmegaConf.create(
        {
            "name": "adamw",
            "lr": config.learning_rate,
            "betas": [0.9, 0.98],
            "weight_decay": 0.001,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_ratio": config.warmup_ratio,
                "min_lr": 1e-6,
            },
        }
    )
    comet = CometLogger(
        project_name=os.environ.get("COMET_PROJECT_NAME", "inzwa-asr"),
        workspace=os.environ.get("COMET_WORKSPACE") or None,
        experiment_name=config.run_id,
        save_dir=str(config.output_dir),
    )
    provenance = {
        "run_id": config.run_id,
        "git_commit": _git_commit(),
        "dataset_release": RELEASE_URI,
        "dataset_format": release["format_version"],
        "model_id": config.model_id,
        "model_revision": resolved_revision,
        "seed": config.seed,
        "sampler": {
            "type": "deterministic-sortish-duration-batching",
            "max_batch_duration": config.max_batch_duration,
            "max_batch_size": config.batch_size,
            "sort_window": config.sort_window,
            "natural_mixture": True,
        },
        "validation": {
            "selection_datasets": ["waxal", "fleurs"],
            "diagnostic_dataset": "bible",
            "selection_interval_steps": config.validation_interval_steps,
            "bible_interval_steps": config.bible_validation_interval_steps,
            "limit_per_dataset": config.validation_limit,
        },
        "early_stopping": {
            "patience_epoch_equivalents": config.early_stopping_patience_epochs,
            "min_delta_wer": config.early_stopping_min_delta,
        },
        "output_prefix": config.output_prefix,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (config.output_dir / "config.json").write_text(
        json.dumps({**asdict(config), **provenance}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    comet.log_hyperparams(provenance)

    class ExperimentCallback(pl.Callback):
        def __init__(self) -> None:
            self.best_score = float("inf")
            self.meaningful_best = float("inf")
            self.last_meaningful_step = 0
            self.last_validation_step = -1
            self.best_path = config.output_dir / "best-weights.pt"
            self.steps_per_epoch = len(sampler)
            self.patience_steps = (
                config.early_stopping_patience_epochs * self.steps_per_epoch
            )

        def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            sampler.set_epoch(trainer.current_epoch)

        def on_train_batch_end(
            self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
        ) -> None:
            if (
                trainer.global_step
                and trainer.global_step % config.exposure_interval == 0
            ):
                metrics = sampler.exposure_metrics()
                pl_module.log_dict(metrics, logger=True)
                print(json.dumps({"step": trainer.global_step, **metrics}), flush=True)

            if (
                trainer.global_step
                and trainer.global_step % config.validation_interval_steps == 0
                and trainer.global_step != self.last_validation_step
            ):
                include_bible = (
                    trainer.global_step % config.bible_validation_interval_steps == 0
                )
                self._validate(trainer, pl_module, include_bible=include_bible)

        def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            if trainer.global_step != self.last_validation_step:
                self._validate(trainer, pl_module, include_bible=True)

        def _validate(
            self, trainer: Any, pl_module: Any, *, include_bible: bool
        ) -> None:
            _disable_prediction_logging(pl_module)
            names = ["waxal", "fleurs"]
            if include_bible:
                names.append("bible")
            results = {
                name: evaluate_manifest(
                    pl_module,
                    manifests[name],
                    name=name,
                    batch_size=config.validation_batch_size,
                    output_dir=config.output_dir,
                    limit=config.validation_limit,
                )
                for name in names
            }
            metrics = {
                f"validation/{name}/{metric}": value
                for name, result in results.items()
                for metric, value in result["metrics"].items()
            }
            selection = (
                results["waxal"]["metrics"]["wer"] + results["fleurs"]["metrics"]["wer"]
            ) / 2
            metrics["validation/selection_wer"] = selection
            comet.log_metrics(metrics, step=trainer.global_step)
            with (config.output_dir / "validation-history.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": trainer.global_step,
                            "selection_wer": selection,
                            "datasets": results,
                        }
                    )
                    + "\n"
                )
            if selection < self.best_score:
                self.best_score = selection
                state = {
                    key: value.detach().cpu()
                    for key, value in pl_module.state_dict().items()
                }
                torch.save(state, self.best_path)
            if selection <= self.meaningful_best - config.early_stopping_min_delta:
                self.meaningful_best = selection
                self.last_meaningful_step = trainer.global_step
            elif trainer.global_step - self.last_meaningful_step >= self.patience_steps:
                trainer.should_stop = True
            self.last_validation_step = trainer.global_step

    callback = ExperimentCallback()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=config.max_epochs,
        gradient_clip_val=1.0,
        logger=comet,
        callbacks=[callback],
        enable_checkpointing=False,
        limit_train_batches=config.limit_train_batches or 1.0,
        log_every_n_steps=25,
        default_root_dir=str(config.output_dir),
    )
    trainer.fit(model, train_dataloaders=train_loader)
    if not callback.best_path.is_file():
        raise RuntimeError("Training finished without a best weights checkpoint")
    model.load_state_dict(
        torch.load(callback.best_path, map_location="cpu", weights_only=True)
    )
    final_model = config.output_dir / "inzwa-parakeet-tdt-0.6b-v3.nemo"
    model.save_to(str(final_model))
    summary = {
        "status": "complete",
        **provenance,
        "best_selection_wer": callback.best_score,
        "best_weights": {
            "path": str(callback.best_path),
            "sha256": file_sha256(callback.best_path),
        },
        "final_model": {"path": str(final_model), "sha256": file_sha256(final_model)},
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary["uploaded"] = _upload_artifacts(config.output_dir, config.output_prefix)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--release-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--output-prefix", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--model-id", default=BASE_MODEL)
    result.add_argument("--model-revision", default="main")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--max-epochs", type=int, default=10)
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--max-batch-duration", type=float, default=60.0)
    result.add_argument("--sort-window", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=5e-5)
    result.add_argument("--warmup-ratio", type=float, default=0.1)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--validation-batch-size", type=int, default=16)
    result.add_argument("--validation-interval-steps", type=int, default=2000)
    result.add_argument("--bible-validation-interval-steps", type=int, default=10_000)
    result.add_argument("--early-stopping-patience-epochs", type=int, default=2)
    result.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    result.add_argument("--exposure-interval", type=int, default=1000)
    result.add_argument("--limit-train-batches", type=int)
    result.add_argument("--validation-limit", type=int)
    result.add_argument("--reuse-prepared-output", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(train(TrainConfig(**vars(args))), indent=2))
