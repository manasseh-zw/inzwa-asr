from __future__ import annotations

from inzwa_asr.sampler import DurationBatchSampler


def rows() -> list[dict[str, object]]:
    return [
        {"id": str(index), "duration": duration, "source": source, "tier": tier}
        for index, (duration, source, tier) in enumerate(
            [
                (2.0, "bible", None),
                (8.0, "fleurs", None),
                (6.0, "waxal-human", None),
                (4.0, "waxal-pseudo", "high"),
                (3.0, "waxal-pseudo", "very_high"),
                (7.0, "bible", None),
            ]
        )
    ]


def test_sampler_is_deterministic_and_losss_duration_bounded() -> None:
    first = DurationBatchSampler(
        rows(), max_batch_duration=10.0, max_batch_size=2, seed=42, sort_window=3
    )
    second = DurationBatchSampler(
        rows(), max_batch_duration=10.0, max_batch_size=2, seed=42, sort_window=3
    )
    first_batches = list(first)
    assert first_batches == list(second)
    assert sorted(index for batch in first_batches for index in batch) == list(range(6))
    assert all(len(batch) <= 2 for batch in first_batches)
    assert all(
        sum(float(rows()[index]["duration"]) for index in batch) <= 10
        for batch in first_batches
    )


def test_sampler_changes_each_epoch_without_changing_natural_mixture() -> None:
    sampler = DurationBatchSampler(
        rows(), max_batch_duration=10.0, max_batch_size=2, seed=42, sort_window=2
    )
    epoch_zero = list(sampler)
    exposure_zero = sampler.exposure.copy()
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert epoch_zero != epoch_one
    assert sampler.exposure == exposure_zero
    assert sum(sampler.exposure.values()) == len(rows())


def test_sampler_keeps_an_individually_long_example() -> None:
    sampler = DurationBatchSampler(
        [{"duration": 11.0, "source": "bible", "tier": None}],
        max_batch_duration=10.0,
    )
    assert list(sampler) == [[0]]
