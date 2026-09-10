"""Local speech-to-text with faster-whisper (CTranslate2 port of OpenAI Whisper).

Chosen over the hosted APIs for the first cut because meeting audio is sensitive and
this keeps it on the machine. Swap in another backend by implementing ``Transcriber``.
"""

from __future__ import annotations

import math

from meetingminutes.audio.chunker import AudioChunk
from meetingminutes.transcribe.base import Segment


class FasterWhisperTranscriber:
    def __init__(
        self,
        model: str = "base.en",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
        vad_filter: bool = True,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run `pip install 'meetingminutes[whisper]'`."
            ) from exc
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        self.vad_filter = vad_filter

    def transcribe(self, chunk: AudioChunk) -> list[Segment]:
        if chunk.sample_rate != 16_000:
            raise ValueError("faster-whisper expects 16 kHz audio; capture at sample_rate=16000")
        segments, _info = self._model.transcribe(
            chunk.samples,
            language=self.language,
            vad_filter=self.vad_filter,
            beam_size=1,  # greedy: latency matters more than the last % of accuracy live
            condition_on_previous_text=False,  # chunks are independent; avoids hallucination loops
        )
        out: list[Segment] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            # avg_logprob is a log-probability; exp() gives a rough 0..1 confidence.
            confidence = math.exp(seg.avg_logprob) if seg.avg_logprob is not None else None
            out.append(
                Segment(
                    text=text,
                    start_seconds=chunk.start_seconds + seg.start,
                    end_seconds=chunk.start_seconds + seg.end,
                    confidence=confidence,
                )
            )
        return out
