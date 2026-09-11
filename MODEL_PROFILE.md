# Inzwa ASR model profile

This page records the current best Inzwa ASR checkpoint and the measurements
available as of 11 September 2026. The results support a strong Shona model.
They do not, by themselves, prove a best-in-class claim across every Shona use
case.

## Model identity

| Field | Value |
| --- | --- |
| Architecture | NVIDIA Parakeet TDT 0.6B v3 |
| Base model | `nvidia/parakeet-tdt-0.6b-v3` |
| Current checkpoint | FLEURS adaptation, step 5,750 |
| Model SHA-256 | `f6510ce38ebb9071eadce047afe7b4964c846833e5a8afc0c87c6d1bdbf1de1f` |
| Training seed | 42 |
| Output format | NeMo `.nemo` archive |
| Artifact size | 2,509,332,480 bytes |

Internal artifact location:

```text
s3://teleagents-research-102431378819-us-east-1/models/asr/inzwa/adaptation/fleurs-adaptation-lr3e6-8bit-20260911a/inzwa-parakeet-tdt-0.6b-v3.nemo
```

## Training history

The first stage fine-tuned Parakeet on the immutable `full-corpus-v2` release.
It contains 82,110 examples and 331.86 hours from human WAXAL, high-confidence
and very-high-confidence WAXAL pseudo-labels, book-disjoint Bible training
audio, and the Shona FLEURS training split. Sampling used the natural mixture,
without source weights or oversampling. Step 82,000 produced the selected
first-stage model.

The second stage continued from that model using FLEURS training data only. It
used a `3e-6` learning rate, batch size 1, 8-bit AdamW optimizer states, and
FLEURS validation for checkpoint selection. Step 5,750 was selected. The run
was stopped by the user after the validation at step 6,500.

Test data and the private curated set were not available to either training
stage, checkpoint selection, or early stopping.

## Frozen test evaluation

The evaluator decoded every source at its recorded sample rate and explicitly
resampled audio to 16 kHz. Each table entry is a corpus-level score for that
test set.

| Test set | Rows | Normalized WER | Raw WER | CER |
| --- | ---: | ---: | ---: | ---: |
| WAXAL | 1,565 | 20.77% | 27.40% | 3.74% |
| FLEURS `sn_zw` | 925 | 26.85% | 29.27% | 6.22% |
| Bible, book-disjoint | 1,593 | 5.52% | 19.10% | 1.11% |
| Curated 150 | 150 | 11.21% | 19.30% | 2.04% |

Normalized scoring applies Unicode NFKC, case folding, punctuation-to-space
replacement, and whitespace collapse. Raw WER uses the references and
hypotheses without that normalization. CER uses normalized text.

### Aggregate views

| Aggregate | Normalized WER | Raw WER | CER |
| --- | ---: | ---: | ---: |
| Public macro, equal WAXAL/FLEURS/Bible weight | 17.71% | 25.26% | 3.69% |
| Four-domain macro, including curated 150 | 16.09% | 23.77% | 3.28% |

These are unweighted means of the domain-level scores. The four-domain macro
is useful internally, but it is not a fully reproducible public benchmark
because curated 150 is private. Neither macro represents a production traffic
distribution.

Dataset revisions:

| Test set | Revision |
| --- | --- |
| WAXAL | `f91b1a79cbac15520d3c808b56e5192bf903f280` |
| FLEURS | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` |
| Bible | `36f7967da6e2a92037a73287082fccf5481fe495` |
| Curated 150 | `4c72a535218164cab117126e5bc9c4a4799d3101` |

## Change from the first-stage model

The FLEURS adaptation made the largest difference to raw FLEURS WER. It also
left Bible performance nearly unchanged.

| Test set | First-stage normalized WER | Current normalized WER | First-stage raw WER | Current raw WER |
| --- | ---: | ---: | ---: | ---: |
| WAXAL | 20.73% | 20.77% | 27.66% | 27.40% |
| FLEURS | 28.47% | 26.85% | 38.14% | 29.27% |
| Bible | 5.65% | 5.52% | 19.19% | 19.10% |
| Curated 150 | 10.81% | 11.21% | 19.62% | 19.30% |

The public normalized macro improved from 18.28% to 17.71%. The private
four-domain normalized macro improved from 16.42% to 16.09%.

## Sunbird comparison

`Sunbird/asr-whisper-51-african-languages` at revision
`213531767f739bc5b1e5dcb3e2ec9e112674ab67` was evaluated on the same frozen
test inputs and with the same scoring code.

| Test set | Inzwa normalized WER | Sunbird normalized WER |
| --- | ---: | ---: |
| WAXAL | 20.77% | 15.77% |
| FLEURS | 26.85% | 18.60% |
| Bible | 5.52% | 33.52% |
| Curated 150 | 11.21% | 12.89% |
| Four-domain macro | 16.09% | 20.19% |

Sunbird is better on WAXAL and FLEURS. Inzwa is much better on Bible and is
modestly better on curated 150. The macro should therefore accompany the full
domain table, not replace it.

## A10G inference benchmark

`shona-speed-v1` contains 1,765 clips and 6.007 hours of audio. It uses the
same frozen manifest for both models:

```text
SHA-256 6d7f927dc9af78d287de7794d77f04c2b2e1bc9485c1e340b8263be8c1a4bb80
```

| Source | Clips | Audio hours |
| --- | ---: | ---: |
| WAXAL | 373 | 1.918 |
| FLEURS | 459 | 1.922 |
| Bible | 783 | 1.919 |
| Curated 150 | 150 | 0.248 |

The benchmark ran on one NVIDIA A10G in an AWS `g5.xlarge`. Both models used
batch size 16 and ascending-duration ordering. Timing includes local audio
loading, preprocessing, inference, and decoding. It excludes model loading.
Audio was already cached on the instance. The run used each framework's native
inference path, without TensorRT, ONNX Runtime, `torch.compile`, quantization,
or another inference engine.

| Metric | Inzwa | Sunbird |
| --- | ---: | ---: |
| Timed audio | 6.007 h | 6.007 h |
| Timed wall time | 42.96 s, median of 3 | 335.60 s, 1 pass |
| Real-time factor | 0.001986 | 0.015519 |
| Audio hours per wall-clock hour | 503.42 | 64.44 |
| Real-time speed | 503.42x | 64.44x |
| Model load time | 47.75 s | 37.53 s |
| Peak allocated VRAM | 5.60 GiB | 7.10 GiB |
| Pooled normalized WER on speed set | 15.98% | 22.67% |
| Pooled raw WER on speed set | 24.49% | 32.62% |
| Pooled CER on speed set | 3.24% | 4.78% |

Inzwa was 7.81 times faster in steady-state throughput on this setup. Its three
passes produced the same hypothesis checksum. Sunbird was run once after its
batch-size sweep, so its timing does not have the same repetition count.

The pooled speed-set WER is not the public macro score. It weights the words in
the selected six-hour corpus and exists mainly as an inference-integrity check.
Use the frozen test table for accuracy claims.

## What this profile does not measure

The current evaluation does not cover streaming latency, CPU inference,
concurrent request serving, long-form segmentation, noisy far-field speech,
English or Ndebele code-switching, names and entities, or deployment-engine
optimizations. Curated 150 contains one speaker and should not be described as
a population-level robustness test.

The next benchmark release should add speaker and environment diversity,
code-switching, named entities, long-form audio, and latency under concurrent
requests. Those additions matter more than compressing the current results
into one leaderboard number.

## Provenance

Evaluation summary:

```text
s3://teleagents-research-102431378819-us-east-1/models/asr/inzwa/evaluations/fleurs-adaptation-step5750-20260911a/full-test-v1/summary.json
```

Speed result:

```text
s3://teleagents-research-102431378819-us-east-1/models/asr/inzwa/benchmarks/shona-speed-v1/parakeet-step5750.json
```

Training summary:

```text
s3://teleagents-research-102431378819-us-east-1/models/asr/inzwa/adaptation/fleurs-adaptation-lr3e6-8bit-20260911a/summary.json
```
