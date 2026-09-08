"""Decode release audio at its recorded rate and resample to 16 kHz."""

from __future__ import annotations

import io
import math
from typing import BinaryIO

import numpy as np
import soundfile
from scipy.signal import resample_poly

from .constants import TARGET_SAMPLE_RATE


def decode_audio(
    payload: bytes | BinaryIO, target_rate: int = TARGET_SAMPLE_RATE
) -> tuple[np.ndarray, int, int]:
    waveform, source_rate = soundfile.read(payload, dtype="float32", always_2d=True)
    source_rate = int(source_rate)
    waveform = waveform.mean(axis=1)
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        waveform = resample_poly(
            waveform, target_rate // divisor, source_rate // divisor
        ).astype(np.float32)
    return waveform, source_rate, target_rate


def decode_audio_bytes(payload: bytes) -> tuple[np.ndarray, int, int]:
    return decode_audio(io.BytesIO(payload))
