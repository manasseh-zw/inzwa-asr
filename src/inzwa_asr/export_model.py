"""Export selected Inzwa weights to NeMo in a fresh, low-memory process."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from .release import file_sha256
from .training import (
    _disable_cuda_graphs,
    _disable_prediction_logging,
    _upload_artifacts,
)


def export_model(output_dir: Path, output_prefix: str) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    weights_path = output_dir / "best-weights.pt"
    initial_model = output_dir / "model-cache" / "initial.nemo"
    for path in (config_path, summary_path, weights_path, initial_model):
        if not path.is_file():
            raise FileNotFoundError(f"Required export input is missing: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not bool(config.get("defer_final_model_export")):
        raise RuntimeError("Run was not configured for deferred model export")
    if str(config.get("output_prefix")) != output_prefix:
        raise RuntimeError("Export prefix does not match the training configuration")

    import nemo.collections.asr as nemo_asr
    import torch

    model = nemo_asr.models.ASRModel.restore_from(
        str(initial_model), map_location="cpu"
    )
    _disable_cuda_graphs(model)
    _disable_prediction_logging(model)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Best checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    del state
    gc.collect()

    final_model = output_dir / "inzwa-parakeet-tdt-0.6b-v3.nemo"
    model.save_to(str(final_model))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "complete"
    summary["final_model"] = {
        "path": str(final_model),
        "sha256": file_sha256(final_model),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["uploaded"] = _upload_artifacts(output_dir, output_prefix)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    print(json.dumps(export_model(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
