---
license: cc-by-4.0
language:
- sn
pipeline_tag: automatic-speech-recognition
library_name: nemo
base_model: nvidia/parakeet-tdt-0.6b-v3
tags:
- shona
- asr
- parakeet
- tdt
- nemo
---

# Inzwa ASR 0.6B

This is a private preview of Inzwa ASR, a Shona speech recognizer based on
NVIDIA Parakeet TDT 0.6B v3. The current checkpoint is the step-5,750 FLEURS
adaptation of the full-corpus model.

The preview exists for clean-environment testing and demo preparation. The
weights, loading instructions, and documentation may change before the public
release.

## Results

| Test set | Normalized WER | Raw WER | CER |
| --- | ---: | ---: | ---: |
| WAXAL | 20.77% | 27.40% | 3.74% |
| FLEURS `sn_zw` | 26.85% | 29.27% | 6.22% |
| Bible, book-disjoint | 5.52% | 19.10% | 1.11% |
| Curated 150 | 11.21% | 19.30% | 2.04% |

The equal-weight public macro across WAXAL, FLEURS, and Bible is 17.71%
normalized WER. Curated 150 is private and was not used for training or
checkpoint selection.

On one NVIDIA A10G, native NeMo inference processed 6.007 hours in a median of
42.96 seconds at batch size 16. That is 503.42 times real time with a 0.001986
real-time factor. The timing used cached local audio ordered by duration and did
not use TensorRT, ONNX Runtime, quantization, or `torch.compile`.

See `MODEL_PROFILE.md` for the complete evaluation contract, provenance,
comparison results, and limitations.

## Loading the model

Install a NeMo version compatible with the repository's locked training
environment, then restore the archive:

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from("inzwa-parakeet-tdt-0.6b-v3.nemo")
model.eval()
transcriptions = model.transcribe(["audio.wav"], batch_size=1)
print(transcriptions[0].text)
```

Decode audio at its source sample rate and resample explicitly when preparing
custom inputs. The training and evaluation pipeline uses 16 kHz model input.

## Files

- `inzwa-parakeet-tdt-0.6b-v3.nemo` contains the selected model.
- `SHA256SUMS` authenticates the model archive.
- `MODEL_PROFILE.md` contains the detailed report.
- `training-summary.json` and `training-config.json` record training provenance.
- `evaluation-summary.json` records the frozen test results.
- `speed-summary.json` records the A10G benchmark.

## Limitations

The current evaluation does not establish performance for streaming,
long-form segmentation, noisy far-field speech, named entities, or English and
Ndebele code-switching. Curated 150 contains one speaker. Do not treat that set
as a population-level robustness measurement.

