"""Turn a stream of small audio blocks into transcription-sized chunks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioChunk:
    samples: np.ndarray  # 1-D float32
    sample_rate: int
    start_seconds: float  # offset from the start of the session

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration

    @property
    def rms(self) -> float:
        if len(self.samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))


class Chunker:
    """Accumulate blocks into fixed-length chunks, tracking absolute time offsets.

    Chunks below ``silence_rms`` are dropped so we do not waste transcriber time on silence.
    The final partial chunk is flushed when the input stream ends.

    Fixed windows are deliberately simple. They can cut a word in half at the boundary;
    the planned follow-up is VAD-aligned boundaries (see README roadmap).
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_seconds: float = 10.0,
        silence_rms: float = 0.005,
        min_chunk_seconds: float = 1.0,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_seconds)
        self.min_frames = int(sample_rate * min_chunk_seconds)
        self.silence_rms = silence_rms

    def chunks(self, blocks: Iterable[np.ndarray]) -> Iterator[AudioChunk]:
        buffer: list[np.ndarray] = []
        buffered = 0
        consumed = 0  # total frames consumed so far -> gives each chunk its start time

        for block in blocks:
            block = np.asarray(block, dtype=np.float32).reshape(-1)
            buffer.append(block)
            buffered += len(block)
            while buffered >= self.chunk_frames:
                joined = np.concatenate(buffer)
                head, tail = joined[: self.chunk_frames], joined[self.chunk_frames :]
                chunk = AudioChunk(head, self.sample_rate, consumed / self.sample_rate)
                consumed += len(head)
                buffer = [tail] if len(tail) else []
                buffered = len(tail)
                if chunk.rms >= self.silence_rms:
                    yield chunk

        if buffered >= self.min_frames:
            joined = np.concatenate(buffer)
            chunk = AudioChunk(joined, self.sample_rate, consumed / self.sample_rate)
            if chunk.rms >= self.silence_rms:
                yield chunk
