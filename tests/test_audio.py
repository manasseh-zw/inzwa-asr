from __future__ import annotations

import io

import numpy as np
import soundfile

from inzwa_asr.audio import decode_audio_bytes


def test_decode_uses_header_rate_and_resamples_to_16khz() -> None:
    source_rate = 24_000
    seconds = 0.25
    waveform = np.sin(
        2 * np.pi * 440 * np.arange(int(source_rate * seconds)) / source_rate
    ).astype(np.float32)
    encoded = io.BytesIO()
    soundfile.write(encoded, waveform, source_rate, format="FLAC")

    decoded, decoded_rate, output_rate = decode_audio_bytes(encoded.getvalue())

    assert decoded_rate == source_rate
    assert output_rate == 16_000
    assert abs(len(decoded) - int(16_000 * seconds)) <= 1
