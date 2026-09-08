"""Frozen public experiment defaults."""

RELEASE_URI = (
    "s3://teleagents-research-102431378819-us-east-1/"
    "datasets/asr/shona-parakeet-full-corpus/releases/v2/full/"
)
RELEASE_FORMAT = "shona-parakeet-full-corpus-v2"
BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
TARGET_SAMPLE_RATE = 16_000
TRAIN_SOURCES = frozenset({"waxal-pseudo", "waxal-human", "bible", "fleurs"})
PSEUDO_TIERS = frozenset({"very_high", "high"})
VALIDATION_MANIFESTS = {
    "waxal": "manifests/validation-waxal-human.jsonl.gz",
    "fleurs": "manifests/validation-fleurs.jsonl.gz",
    "bible": "manifests/validation-bible.jsonl.gz",
}
