# Inzwa ASR

Inzwa ASR is the training code for an open Shona speech recognizer based on
[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
This repository contains the reproducible training path. Corpus construction,
private evaluation data, and historical experiments are intentionally kept out.

The first baseline trains on 82,110 examples, or 331.86 hours, and validates on
4,307 named examples. It uses the natural release mixture without source
weighting, oversampling, quality weighting, or a curriculum.

## Data

Training uses the immutable full-corpus v2 release:

```text
s3://teleagents-research-102431378819-us-east-1/datasets/asr/shona-parakeet-full-corpus/releases/v2/full/
```

The release contains WAXAL `very_high` and `high` pseudo-labels, human WAXAL
train, book-disjoint Bible train, and FLEURS `sn_zw` train. Its WAXAL, FLEURS,
and Bible validation manifests are separate. WAXAL and FLEURS choose the
checkpoint; Bible is diagnostic. Test sets and the private 150-example set are
never loaded by this training code.

Every run checks `READY.json` and all listed SHA-256 checksums. S3 is the source
of truth. A local disk is only a cache.

## Local setup

Python 3.11 is the supported version.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python -m compileall -q src
```

Copy `.env.example` into your own ignored environment file if useful. Do not
commit AWS, Hugging Face, or Comet credentials.

## Release preflight

Hydrate only manifests and metadata first. This checks each downloaded file but
does not mark the cache as ready for training because audio shards are absent.

```bash
export AWS_PROFILE=teleagents-admin
export AWS_REGION=us-east-1
uv run inzwa-release hydrate data/full-v2 --manifest-only
uv run inzwa-release verify data/full-v2 --manifest-only
uv run inzwa-preflight data/full-v2 --report outputs/loader-preflight.json
```

The loader preflight selects one example from Bible, FLEURS, and human WAXAL,
plus one WAXAL pseudo-label from each allowed tier. It downloads only the needed
tar shards, verifies them against `READY.json`, decodes the FLAC members at the
recorded rate, and explicitly produces 16 kHz audio.

For a training cache, hydrate and verify the entire release:

```bash
uv run inzwa-release hydrate /data/inzwa-full-v2
uv run inzwa-release verify /data/inzwa-full-v2
```

## Training

Install the GPU dependencies with `uv sync --extra train`. The baseline uses
deterministic duration-aware batches and visits every row once per epoch. The
safe L40S fallback is two examples per batch. Source and pseudo-tier exposure is
sent to Comet every 1,000 steps through Lightning's `CometLogger`. WAXAL and
FLEURS validation runs every 2,000 steps. Bible validation runs every 10,000
steps and at epoch end.

```bash
export COMET_API_KEY=...
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run inzwa-train \
  --release-root /data/inzwa-full-v2 \
  --output-dir outputs/full-corpus-v2-seed42 \
  --run-id full-corpus-v2-seed42 \
  --output-prefix s3://YOUR-BUCKET/asr/inzwa/runs/full-corpus-v2-seed42
```

The run records the dataset release, resolved model revision, seed, git commit,
run ID, sampler settings, and output prefix. It keeps a best weights-only state
file and a final NeMo artifact. Metrics, configuration, provenance, and run
metadata are uploaded with them. No optimizer checkpoint is written.

SkyPilot tasks live in `sky/`. Run `manifest-preflight.yaml` before the L40S
smoke task. Freeze the run ID, git commit, resolved model revision, sampler seed,
and output prefix before using `full-training.yaml`. The example S3 destinations
must be replaced first.

## Evaluation

Validation reports raw WER plus punctuation-insensitive WER and CER. The latter
uses Unicode NFKC, case folding, punctuation-to-space replacement, and
whitespace collapse. WAXAL and FLEURS WER have equal weight in the selection
score. Bible results are logged but cannot change the selected checkpoint. The
early stopper requires a 0.001 absolute WER improvement and waits two full
epoch-equivalents without such an improvement. The hard limit is 10 epochs.

Final test evaluation is deliberately outside the training command. This keeps
test and private data out of checkpoint selection.

The optional FLEURS adaptation starts from the selected full-corpus model and
uses only FLEURS train. A 10-step smoke test is required before the full task.
On a `g5.xlarge`, the A10G has enough VRAM for batch-size-1 training but the VM
has only 16 GB of host RAM. Its tasks therefore finish weights-only training,
exit to release trainer memory, and perform NeMo export in a fresh process.
They also use zero loader workers and smaller validation batches; neither
changes the examples, gradients, checkpoint objective, or resulting weights.
The adaptation uses a `3e-6` learning rate and FLEURS-validation early stopping.
It still logs WAXAL validation so later checkpoint interpolation can be selected
on the equal WAXAL/FLEURS validation mean. Test and curated data remain
inaccessible to training and interpolation selection.

Before the private pass, `sky/stage-curated-150.yaml` mirrors the frozen
`manassehzw/sna-manasseh-150-raw` revision into S3 and writes a checksum-bearing
`READY.json`. The A10G evaluator includes this set when that gate is present.

Run the frozen public test suite (WAXAL, FLEURS `sn_zw`, and the book-disjoint
Bible test) with `sky/a10g-test-evaluation.yaml`. The evaluator pins each
dataset revision, decodes at the recorded sample rate, explicitly resamples to
16 kHz, and uploads per-example predictions plus raw WER, normalized WER, and
CER. The private curated-150 set remains a separate robustness evaluation and
is not part of the reproducible public score.

## License

No license has been added yet. The repository code is newly written, but the
intended license still needs confirmation against the NVIDIA model terms and
the source-dataset licenses. Until that review is complete, normal copyright
rules apply.
