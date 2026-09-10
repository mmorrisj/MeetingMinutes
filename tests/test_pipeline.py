from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from meetingminutes.audio.chunker import AudioChunk, Chunker
from meetingminutes.pipeline import Pipeline
from meetingminutes.sinks import JsonlSink, MultiSink
from meetingminutes.transcribe.base import Segment
from tests.conftest import tone


class FakeSource:
    """Replays a signal in blocks; supports stop() like the real source."""

    def __init__(self, signal: np.ndarray, sample_rate: int, block: int):
        self._signal, self.sample_rate, self._block = signal, sample_rate, block
        self._stopped = threading.Event()

    def blocks(self):
        for i in range(0, len(self._signal), self._block):
            if self._stopped.is_set():
                return
            yield self._signal[i : i + self._block]

    def stop(self):
        self._stopped.set()


class EchoTranscriber:
    """Emits one segment per chunk describing the chunk, so we can check offsets."""

    def transcribe(self, chunk: AudioChunk) -> list[Segment]:
        return [Segment(f"chunk@{chunk.start_seconds:g}", chunk.start_seconds, chunk.end_seconds)]


class Collect:
    def __init__(self):
        self.segments: list[Segment] = []
        self.closed = False

    def write(self, segments):
        self.segments.extend(segments)

    def close(self):
        self.closed = True


def test_end_to_end_offsets_and_jsonl(tmp_path: Path, sr):
    source = FakeSource(tone(32, sr), sr, sr // 2)
    collect = Collect()
    out = tmp_path / "t.jsonl"
    pipeline = Pipeline(
        source, Chunker(sr, 10), EchoTranscriber(), MultiSink(collect, JsonlSink(out))
    )

    pipeline.run()

    assert [s.text for s in collect.segments] == ["chunk@0", "chunk@10", "chunk@20", "chunk@30"]
    assert collect.closed
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[-1] == {
        "text": "chunk@30",
        "start_seconds": 30.0,
        "end_seconds": 32.0,
        "confidence": None,
    }
    assert pipeline.chunks_transcribed == 4 and pipeline.segments_written == 4


def test_capture_errors_propagate_after_cleanup(sr):
    class Boom(FakeSource):
        def blocks(self):
            yield tone(1, sr)
            raise OSError("device unplugged")

    collect = Collect()
    pipeline = Pipeline(Boom(tone(1, sr), sr, sr), Chunker(sr, 10), EchoTranscriber(), collect)
    try:
        pipeline.run()
    except OSError as exc:
        assert "unplugged" in str(exc)
    else:
        raise AssertionError("expected the capture error to surface")
    assert collect.closed
    assert [s.text for s in collect.segments] == ["chunk@0"]  # partial tail was still flushed
