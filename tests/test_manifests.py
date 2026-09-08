from __future__ import annotations

from inzwa_asr.manifests import validate_training_rows
from inzwa_asr.preflight import representative_rows


def test_representative_rows_cover_sources_and_pseudo_tiers() -> None:
    rows = [
        {"id": "1", "source": "bible", "tier": None},
        {"id": "2", "source": "fleurs", "tier": None},
        {"id": "3", "source": "waxal-human", "tier": None},
        {"id": "4", "source": "waxal-pseudo", "tier": "high"},
        {"id": "5", "source": "waxal-pseudo", "tier": "very_high"},
    ]
    selected = representative_rows(rows)
    assert len(selected) == 5
    summary = validate_training_rows(rows)
    assert summary["sources"]["waxal-pseudo"] == 2
