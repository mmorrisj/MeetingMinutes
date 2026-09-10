"""Capture -> chunk -> transcribe -> sink.

Capture runs on its own thread feeding a bounded queue, so a slow transcriber never makes the
audio device drop frames; instead back-pressure shows up as a growing queue, which we log.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator

import numpy as np

from meetingminutes.audio.capture import AudioSource
from meetingminutes.audio.chunker import Chunker
from meetingminutes.sinks import Sink
from meetingminutes.transcribe.base import Transcriber

log = logging.getLogger(__name__)

_SENTINEL = object()


class Pipeline:
    def __init__(
        self,
        source: AudioSource,
        chunker: Chunker,
        transcriber: Transcriber,
        sink: Sink,
        max_queued_blocks: int = 600,  # 5 min at 0.5 s blocks before we start dropping
    ) -> None:
        self.source = source
        self.chunker = chunker
        self.transcriber = transcriber
        self.sink = sink
        self._queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=max_queued_blocks)
        self._stop = threading.Event()
        self._capture_error: BaseException | None = None
        self.segments_written = 0
        self.chunks_transcribed = 0

    # -- capture thread -------------------------------------------------------------------
    def _capture(self) -> None:
        try:
            for block in self.source.blocks():
                if self._stop.is_set():
                    break
                try:
                    self._queue.put(block, timeout=1.0)
                except queue.Full:
                    log.warning(
                        "transcriber is behind; dropping %.2fs of audio",
                        len(block) / self.source.sample_rate,
                    )
        except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread
            self._capture_error = exc
        finally:
            self._queue.put(_SENTINEL)

    def _drain(self) -> Iterator[np.ndarray]:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            yield item  # type: ignore[misc]

    # -- public API -----------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        self.source.stop()

    def run(self) -> None:
        """Block until the source ends or ``stop()`` is called. Safe to call once."""
        worker = threading.Thread(target=self._capture, name="mm-capture", daemon=True)
        worker.start()
        try:
            for chunk in self.chunker.chunks(self._drain()):
                if self._queue.qsize() > self._queue.maxsize // 2:
                    log.warning("transcription backlog: %d blocks queued", self._queue.qsize())
                segments = self.transcriber.transcribe(chunk)
                self.chunks_transcribed += 1
                if segments:
                    self.sink.write(segments)
                    self.segments_written += len(segments)
        finally:
            self.stop()
            worker.join(timeout=5)
            self.sink.close()
        if self._capture_error is not None:
            raise self._capture_error
