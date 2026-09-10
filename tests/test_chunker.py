from __future__ import annotations

import numpy as np

from meetingminutes.audio.chunker import Chunker
from tests.conftest import silence, tone


def _blocks(signal: np.ndarray, block: int):
    for i in range(0, len(signal), block):
        yield signal[i : i + block]


def test_fixed_windows_have_contiguous_timestamps(sr):
    signal = tone(25.0, sr)
    chunks = list(Chunker(sr, chunk_seconds=10).chunks(_blocks(signal, sr // 2)))

    assert [c.start_seconds for c in chunks] == [0.0, 10.0, 20.0]
    assert [round(c.duration, 3) for c in chunks] == [10.0, 10.0, 5.0]  # final partial flushed
    assert all(len(c.samples) == len(c.samples.reshape(-1)) for c in chunks)


def test_silence_is_skipped_but_time_still_advances(sr):
    signal = np.concatenate([tone(10, sr), silence(10, sr), tone(10, sr)])
    chunks = list(Chunker(sr, chunk_seconds=10).chunks(_blocks(signal, sr)))

    assert [c.start_seconds for c in chunks] == [0.0, 20.0]


def test_short_tail_is_dropped(sr):
    signal = tone(10.4, sr)
    chunks = list(Chunker(sr, chunk_seconds=10, min_chunk_seconds=1.0).chunks(_blocks(signal, sr)))
    assert len(chunks) == 1


def test_blocks_larger_than_chunk_are_split(sr):
    signal = tone(35, sr)
    chunks = list(Chunker(sr, chunk_seconds=10).chunks([signal]))
    assert [c.start_seconds for c in chunks] == [0.0, 10.0, 20.0, 30.0]


def test_stereo_blocks_are_rejected_upstream_not_here(sr):
    # Chunker takes whatever 1-D data it is given; to_mono lives in capture.
    from meetingminutes.audio.capture import to_mono

    stereo = np.stack([tone(1, sr), tone(1, sr, amp=0.1)], axis=1)
    mono = to_mono(stereo)
    assert mono.ndim == 1 and mono.dtype == np.float32
    assert np.isclose(mono.max(), 0.2, atol=0.01)
