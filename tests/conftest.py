from __future__ import annotations

import numpy as np
import pytest


def tone(seconds: float, sample_rate: int = 16_000, freq: float = 440.0, amp: float = 0.3):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, sample_rate: int = 16_000):
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


@pytest.fixture
def sr() -> int:
    return 16_000
