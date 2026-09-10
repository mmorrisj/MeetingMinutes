"""Audio sources: anything that yields blocks of float32 PCM."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from meetingminutes.audio.devices import LoopbackDevice, _soundcard


@runtime_checkable
class AudioSource(Protocol):
    """A stream of mono float32 blocks in the range [-1, 1]."""

    sample_rate: int

    def blocks(self) -> Iterator[np.ndarray]:
        """Yield 1-D float32 arrays until the source is exhausted or ``stop()`` is called."""
        ...

    def stop(self) -> None: ...


def to_mono(block: np.ndarray) -> np.ndarray:
    """Down-mix an ``(frames, channels)`` block to a 1-D float32 array."""
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    return block.mean(axis=1, dtype=np.float32)


class SoundcardLoopbackSource:
    """Read what the system is playing via a loopback/monitor device.

    Produces mono float32 blocks of ``block_seconds`` length. ``blocks()`` runs until
    ``stop()`` is called from another thread or the generator is closed.
    """

    def __init__(
        self,
        device: LoopbackDevice,
        sample_rate: int = 16_000,
        block_seconds: float = 0.5,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.block_frames = max(1, int(sample_rate * block_seconds))
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def blocks(self) -> Iterator[np.ndarray]:
        sc = _soundcard()
        mic = sc.get_microphone(self.device.id, include_loopback=True)
        self._stop.clear()
        # channels=1 asks the backend to down-mix; we still guard with to_mono in case a
        # backend ignores it (PulseAudio honours it, WASAPI does too).
        with mic.recorder(samplerate=self.sample_rate, channels=1) as rec:
            while not self._stop.is_set():
                data = rec.record(numframes=self.block_frames)
                yield to_mono(np.asarray(data))
