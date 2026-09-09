"""Text normalization and named validation for checkpoint selection."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any


def read_evaluation_manifest(path: Path) -> list[dict[str, Any]]:
    """Read the deliberately smaller inference-only manifest contract."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "audio_filepath", "text"} - row.keys()
            if missing:
                raise ValueError(f"Missing {sorted(missing)} at {path}:{line_number}")
            if not Path(str(row["audio_filepath"])).is_file():
                raise FileNotFoundError(
                    f"Missing audio at {path}:{line_number}: {row['audio_filepath']}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Evaluation manifest is empty: {path}")
    return rows


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    return " ".join(text.split())


def score(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    from jiwer import cer, wer

    normalized_references = [normalize_text(value) for value in references]
    normalized_hypotheses = [normalize_text(value) for value in hypotheses]
    return {
        "wer": float(wer(normalized_references, normalized_hypotheses)),
        "cer": float(cer(normalized_references, normalized_hypotheses)),
        "raw_wer": float(wer(references, hypotheses)),
    }


def evaluate_manifest(
    model: Any,
    manifest: Path,
    *,
    name: str,
    batch_size: int,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_evaluation_manifest(manifest)
    if limit is not None:
        if limit < 1:
            raise ValueError("validation limit must be positive")
        rows = rows[:limit]
    paths = [str(row["audio_filepath"]) for row in rows]
    hypotheses = model.transcribe(paths, batch_size=batch_size, verbose=False)
    texts = [
        str(getattr(hypothesis, "text", hypothesis) or "") for hypothesis in hypotheses
    ]
    metrics = score([str(row["text"]) for row in rows], texts)
    prediction_path = output_dir / f"validation-{name}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row, hypothesis in zip(rows, texts, strict=True):
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
    return {"name": name, "rows": len(rows), "metrics": metrics}
