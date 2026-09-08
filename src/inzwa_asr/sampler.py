"""Deterministic dynamic duration-aware batching and exposure telemetry."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any


class DurationBatchSampler:
    """Visit every row once per epoch using deterministic sortish batches."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        max_batch_duration: float,
        max_batch_size: int = 2,
        seed: int = 42,
        sort_window: int = 256,
    ) -> None:
        if not rows:
            raise ValueError("rows must not be empty")
        if max_batch_duration <= 0 or max_batch_size < 1 or sort_window < 1:
            raise ValueError("batch limits and sort_window must be positive")
        self.rows = rows
        self.max_batch_duration = max_batch_duration
        self.max_batch_size = max_batch_size
        self.seed = seed
        self.sort_window = sort_window
        self.epoch = 0
        self.exposure: Counter[tuple[str, str]] = Counter()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.exposure.clear()

    def _ordered_indices(self) -> list[int]:
        indices = list(range(len(self.rows)))
        random.Random(self.seed + self.epoch).shuffle(indices)
        windows = [
            indices[start : start + self.sort_window]
            for start in range(0, len(indices), self.sort_window)
        ]
        for number, window in enumerate(windows):
            window.sort(
                key=lambda index: float(self.rows[index]["duration"]),
                reverse=bool(number % 2),
            )
        random.Random(self.seed + self.epoch + 1_000_003).shuffle(windows)
        return [index for window in windows for index in window]

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        duration = 0.0
        for index in self._ordered_indices():
            item_duration = float(self.rows[index]["duration"])
            if batch and (
                len(batch) >= self.max_batch_size
                or duration + item_duration > self.max_batch_duration
            ):
                yield batch
                batch, duration = [], 0.0
            batch.append(index)
            duration += item_duration
            row = self.rows[index]
            self.exposure[(str(row["source"]), str(row.get("tier") or "none"))] += 1
        if batch:
            yield batch

    def __len__(self) -> int:
        return sum(1 for _ in self._batches_without_telemetry())

    def _batches_without_telemetry(self) -> Iterator[list[int]]:
        batch: list[int] = []
        duration = 0.0
        for index in self._ordered_indices():
            item_duration = float(self.rows[index]["duration"])
            if batch and (
                len(batch) >= self.max_batch_size
                or duration + item_duration > self.max_batch_duration
            ):
                yield batch
                batch, duration = [], 0.0
            batch.append(index)
            duration += item_duration
        if batch:
            yield batch

    def exposure_metrics(self) -> dict[str, int]:
        return {
            f"exposure/{source}/{tier}": count
            for (source, tier), count in sorted(self.exposure.items())
        }
